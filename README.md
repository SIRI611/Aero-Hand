# TetherIA Aero Hand Open — Grasp-and-Hold (Isaac Lab)

Isaac Lab port of the TetherIA Aero Hand Open for a grasp-and-hold task: the hand is
mounted pointing down at the ground, closes around an object, and has to hold it
against gravity for the length of the episode without dropping it.

This same codebase is instantiated once per graspable object — currently **mug**,
**bottle**, **card**, and **pen** — each as its own `<object>_cfg.py`
(`mug_cfg.py`, `bottle_cfg.py`, `card_cfg.py`, `pen_cfg.py`) paired with its own copy
of the grasp env. Everything below describes the architecture shared by all four;
[Per-object files](#per-object-files) covers what's actually different between them.

The action space, observation space, and reward *structure* are built to mirror
TetherIA's own official RL environment in `mujoco_playground`. (Their published env
is a cube-rotation task, not grasp-hold — they don't have a grasp task released — but
the core conventions, tendon-space control and torque/energy-aware rewards, carry
over regardless of task or object.)

There is also a **hardware path**, kept entirely in `real_hand/`:
`real_hand/aero_hand_bridge.py` drives the physical hand over ROS2 using the same
7-actuator convention the sim policy outputs, so a policy trained here can be pushed
to the real hand without changing units or ordering. Nothing in `real_hand/` imports
Isaac Lab, and nothing in the scene variants imports ROS2 — the split is clean in
both directions. See [Running on the real hand](#running-on-the-real-hand).

## Shared vs. per-object, at a glance

| Shared across every object | Per-object (tuned separately for mug/bottle/card/pen) |
|---|---|
| `aero_hand_cfg.py` (hand asset, actuator gains, orientation) | `<object>_cfg.py` (USD path, mass, spawn offset) |
| 7-tendon action space & joint-grouping map | `grasp_joint_pos` (per-object finger curl) |
| 68-dim observation layout | Object spawn offset relative to the hand |
| Reward *structure* (which terms exist) | `*_drop_height_thresh`, and the two object-specific reward scales |
| Hand orientation tuning workflow | `rew_scale_torques` / `rew_scale_energy` / `rew_scale_pose` tuning |

## Files

| File | What it is |
|---|---|
| `aero_hand_cfg.py` | Hand asset: USD path, base pose/orientation, per-joint actuator gains. Identical across every object variant. |
| `aero_hand_scene_cfg.py` | Thin `InteractiveSceneCfg` wrapper — ground plane, light, N cloned hands. Not object-specific. |
| `<object>_cfg.py` | The graspable object for one variant: USD path, mass, spawn offset relative to the hand. One per object — `mug_cfg.py`, `bottle_cfg.py`, `card_cfg.py`, `pen_cfg.py`. |
| `<object>_grasp_env.py` | The `DirectRLEnv` for that variant — action space, observation space, reward, reset logic. Same structure in every variant, with the object's name substituted into the object-specific fields (see [Rewards](#rewards)). |
| `test.py` | Standalone script that spawns N envs and holds the tuned grasp pose in the viewport for whichever variant's env module it imports. No RL loop, just for visually checking a pose. **In `mug/` this is named `test_sim.py`**, to keep it distinct from the real-hand `real_hand/test_ros.py`; `pen/`, `card/`, and `bottle/` still use `test.py`. Worth renaming the other three to match. |
| `run_aero_hand_parallel.py` | Standalone hand-*only* sanity check — no object, no `DirectRLEnv`, no RL action space. Builds the scene directly from `AeroHandSceneCfg` with the raw `InteractiveScene`/`SimulationContext` APIs and drives all 16 joints with one shared sinusoidal open/close curl, ignoring the 7-tendon grouping and per-joint limits entirely. See [Sanity-checking the scene](#sanity-checking-the-scene). |

### Hardware — `real_hand/`

Everything that talks to the **physical** hand over ROS2 lives here, and nothing
else does. No Isaac Lab imports in this directory, no ROS2 imports in the scene
variants. It is object-agnostic — the hand doesn't care what it's holding — so it
sits at the top level rather than being copied into each variant.

| File | What it is |
|---|---|
| `real_hand/aero_hand_bridge.py` | Abstraction between 7-actuator control values (radians) and the real hand over ROS2. Hides joint-space expansion, unit conversion, and ROS2 plumbing. Also reconstructs joint positions from motor angles by inverting the tendon model — the hardware has no joint encoders and publishes only 7 motor angles. Its coupling constants must stay identical to the ones in every `<object>_grasp_env.py`. |
| `real_hand/test_ros.py` | Choreographed demo on the real hand — finger wave, counting 1→5, thumb opposition, piano taps, slow clench. Poses are interpolated and streamed at 100 Hz to match the hand's own firmware loop. Doubles as an end-to-end check that commands and feedback round-trip. |

Variant scripts run from their own directory, so import the bridge with:

```python
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "real_hand"))
from aero_hand_bridge import AeroHandBridge
```

Setup, ROS2 plumbing, control rates, and troubleshooting for the physical hand are
in [Running on the real hand](#running-on-the-real-hand).

## The 7-tendon action space

The real hand has 7 motors driving 16 joints through tendons — it's under-actuated,
so groups of joints are mechanically forced to move together. This is a property of
the *hand*, not the object being grasped, so it's identical across every variant. The
action space matches it: the policy outputs 7 numbers (one per actuator, normalized
to `[-1, 1]`), and each one is broadcast to every joint it physically drives:

| Actuator | Joints it drives |
|---|---|
| `right_thumb_cmc_abd_act` | `right_thumb_cmc_abd` |
| `right_thumb_cmc_flex_act` | `right_thumb_cmc_flex` |
| `right_thumb_tendon_act` | `right_thumb_mcp`, `right_thumb_ip` |
| `right_index_tendon_act` | `right_index_mcp_flex`, `right_index_pip`, `right_index_dip` |
| `right_middle_tendon_act` | `right_middle_mcp_flex`, `right_middle_pip`, `right_middle_dip` |
| `right_ring_tendon_act` | `right_ring_mcp_flex`, `right_ring_pip`, `right_ring_dip` |
| `right_pinky_tendon_act` | `right_pinky_mcp_flex`, `right_pinky_pip`, `right_pinky_dip` |

This is exactly TetherIA's own "compact representation" from their SDK docs
(`docs.tetheria.ai/docs/sdk/#compact-joint-representation`), reimplemented in Isaac
Lab as a broadcast matrix in each `<object>_grasp_env.py`'s `__init__`
(`_act_to_joint_matrix`), since PhysX doesn't model MuJoCo-style spatial tendons
directly.

### Which joints are still independent vs. forced equal

Out of the 16 joints, only 2 stay fully independent — the rest get grouped into 5
clusters where every joint in the cluster is forced to the identical commanded
angle (a straight 1:1 copy, no scaling or ratio involved — safe to do because every
joint inside a group shares the same min/max range). This grouping is fixed by the
hand's mechanical design, so it's the same table for every object — only the
specific angles being averaged differ, since each object has its own tuned
`grasp_joint_pos`. Worked example using the **mug** variant's tuned values:

| Group | Joints | Original per-joint values (example values) | Now |
|---|---|---|---|
| — | thumb base ab/adduction | `1.4` | **Independent** — own actuator, unaffected |
| — | thumb base flexion | `0.0` | **Independent** — own actuator, unaffected |
| thumb tip | mcp / ip | `0.4` / `0.8` | **Forced equal** → both `0.6` |
| index | mcp / pip / dip | `0.9` / `1.0` / `1.0` | **Forced equal** → all `0.967` |
| middle | mcp / pip / dip | `0.8` / `0.9` / `0.9` | **Forced equal** → all `0.867` |
| ring | mcp / pip / dip | `0.9` / `1.0` / `1.0` | **Forced equal** → all `0.967` |
| pinky | mcp / pip / dip | `0.9` / `1.0` / `1.0` | **Forced equal** → all `0.967` |

Re-derive this table for the bottle/card/pen variants by swapping in that
variant's own `grasp_joint_pos` — the *groupings* never change, only which numbers
end up averaged. A pinch grasp (card, pen) will generally have bigger gaps between
mcp and pip/dip than a power grasp (mug, bottle), since pinch grasps rely more on
independent fingertip placement — exactly the kind of precision this coupling gives
up.

One honest caveat: this matches TetherIA's documented compact representation, which
really is this simple 1:1 duplication for normal use. But their docs also mention
the *real* hardware coupling is more mechanically complex, especially for the
thumb — moving one thumb actuator physically tugs on the others too, in a way
that's nonlinear rather than a clean group of identical joints. That deeper
coupling lives in TetherIA's `joints_to_actuations.py`, which this port doesn't
replicate — and Isaac Lab doesn't simulate real tendons either, so this was already
an approximation on top of an approximation. Good enough to train a policy that
plays by the same 7-number rules the real hand does, not a physically exact
recreation of the cable mechanics.

The **reset pose** (`grasp_joint_pos` in the cfg) still specifies all 16 joints
individually — that's just where the hand starts each episode, written straight into
the sim, so it isn't limited by the 7-actuator coupling. Only the *ongoing control*
during the episode goes through the 7-dim action space.

## Observations (68-dim)

Identical layout across every variant — only the numbers coming from `object.data`
differ depending on which object is loaded.

| Slice | Size | Source |
|---|---|---|
| Joint positions | 16 | `hand.data.joint_pos` |
| Joint velocities | 16 | `hand.data.joint_vel` |
| Joint torques | 16 | `hand.data.applied_torque` (implicit-actuator PD estimate) |
| Previous action | 7 | last commanded tendon-space action |
| Object position (rel. to hand root) | 3 | `object.data.root_pos_w - hand.data.root_pos_w` |
| Object linear velocity | 3 | `object.data.root_lin_vel_w` |
| Object angular velocity | 3 | `object.data.root_ang_vel_w` |
| Object orientation (quat) | 4 | `object.data.root_quat_w` |

## Rewards

Named and structured after TetherIA's own reward terms in `rotate_z.py`
(`angvel`/`linvel`/`pose`/`torques`/`energy`/`action_rate`/`termination`), adapted
from "rotate as fast as possible" to "hold still without dropping." The *structure*
is shared; two of the scale names embed the object's name and get tuned per variant
(e.g. `rew_scale_mug_height` in the mug variant, `rew_scale_bottle_height` in the
bottle variant, and so on):

| Term (concept) | Field name pattern | Scale used in the mug variant | What it does |
|---|---|---|---|
| Held / not dropped | `rew_scale_<object>_height` | `5.0` | Reward each step the object hasn't dropped past the threshold. |
| Stillness | `rew_scale_<object>_still` | `-0.1` | Penalizes object linear speed — encourages a stable, non-wobbly grasp. |
| Action rate | `rew_scale_action_rate` | `-0.01` | Penalizes jerky tendon commands, `(action - last_action)²`. |
| Pose | `rew_scale_pose` | `-0.05` | Pulls the 16-joint hand shape back toward `grasp_joint_pos`. |
| Torques | `rew_scale_torques` | `-1e-3` | Penalizes actuator torque / cable tension. **Placeholder — tune per object.** |
| Energy | `rew_scale_energy` | `-1e-3` | Penalizes `\|vel\| * \|torque\|`, wasted mechanical power. **Placeholder — tune per object.** |
| Drop penalty | `rew_scale_drop_penalty` | `-10.0` | Penalty applied every step the object is in the "dropped" state. |

`<object>_drop_height_thresh` (meters of downward drift from the reset height)
decides when the object counts as dropped, for both the reward and episode
termination — this needs its own value per object (a pen slipping 15 cm means
something very different than a bottle slipping 15 cm).

## Hand orientation

The hand's base rotation is set via `_DOWNWARD_FACING_EULER_DEG` in
`aero_hand_cfg.py` (currently `(180, 0, 0)`, i.e. 180° about X) rather than a raw
quaternion, specifically so it's easy to nudge in-viewport. This is shared across
every variant — the hand's orientation doesn't depend on what it's holding. I don't
have visibility into `aero_hand.usda`'s rest pose, so this needs to be
verified/tuned by eye — same workflow you already use for `grasp_joint_pos` and
object spawn position.

## Per-object files

Each of the four variants (mug, bottle, card, pen) has its own:

1. **`<object>_cfg.py`** — `RigidObjectCfg` with that object's USD path, mass, and
   spawn offset.
2. **`grasp_joint_pos`** — a card, a bottle, and a pen need very different finger
   curl amounts than a mug. Tuned by eye with `test.py`.
3. **Spawn offset** — so the object actually starts inside the closed grasp for
   that object's geometry. This changes completely between something as flat as a
   card and something as long as a bottle.
4. **`<object>_drop_height_thresh`** — sized to that object.
5. **Reward scale tuning**, especially `rew_scale_torques`/`rew_scale_energy`/
   `rew_scale_pose` — a pinch grasp (card, pen) vs a power grasp (bottle, mug)
   wants different regularization strength.

Everything else — `aero_hand_cfg.py`, the 7-tendon action mapping, the observation
layout, and the reward *structure* — is object-agnostic and identical across all
four files.

If maintaining four near-duplicate env files gets annoying, the natural refactor is
to rename the object-specific bits in each `<object>_grasp_env.py` (`self.mug` /
`self.bottle` / etc., `MUG_CFG` / `BOTTLE_CFG` / etc., `mug_pos_rel`,
`rew_scale_mug_height`) to generic `self.obj` / `OBJECT_CFG` names, and pass the
object cfg plus its drop threshold and reward scales into a single shared
`AeroHandGraspEnvCfg` as fields — one env class, four cfg instances. Happy to do
that consolidation pass if it'd help.

## Known placeholders — check before training

- `aero_hand_cfg.py`: `_DOWNWARD_FACING_EULER_DEG`, `stiffness`, `effort_limit_sim`
  (shared — verify once, applies to all variants).
- `mug_cfg.py` / `bottle_cfg.py` / `card_cfg.py` / `pen_cfg.py`: object mass, spawn
  `pos` (per object).
- Each `<object>_grasp_env.py`: `rew_scale_torques`, `rew_scale_energy`
  (placeholder starting scales — verify against actual torque magnitudes once you
  can log them, per object).

## Running it

### Sanity-checking the scene

```bash
./isaaclab.sh -p run_aero_hand_parallel.py --num_envs 100
```

No object, no grasp env, no RL — just N cloned hands driven by a single shared
sinusoidal "curl" across all 16 joints, with a random-joint-offset reset every 300
steps. This doesn't exercise the 7-tendon action space, `grasp_joint_pos`, or any
object cfg at all; it's purely for confirming the cloning/spacing (`env_origins`),
joint control, and reset plumbing in `aero_hand_cfg.py`/`aero_hand_scene_cfg.py`
work *before* layering an object or the grasp env on top. Good first thing to run
when standing this up somewhere new. Its per-step `env_origins`/`root_pos_w` prints
are handy for debugging but noisy to leave on — trim them once things look right.

### Checking a tuned grasp pose (per object)

```bash
# visually check the tuned grasp pose across N envs, no RL loop
python test.py --num_envs 100

# fewer envs for faster iteration while tuning grasp_joint_pos / object spawn pos
python test.py --num_envs 9
```

Point `test.py`'s env import at whichever variant you want to check (or keep
separate `test_<object>.py` copies, same pattern as the env files). Unlike
`run_aero_hand_parallel.py`, this one goes through the real `AeroHandGraspEnv` —
object included, actions passed through the actual 7-tendon mapping — and holds the
tuned `grasp_joint_pos` so you can eyeball whether the grip actually holds the
object.

### Training

Every variant's `AeroHandGraspEnv` behaves like any other Isaac Lab `DirectRLEnv` —
`action_space = 7`, `observation_space = 68` — so each one drops into whatever
PPO/training script you're already using, just with a smaller action dimension than
the original 16-joint version.

## Running on the real hand

The bridge speaks the same 7-actuator, radians, `ACTUATOR_NAMES`-ordered convention
the sim policy outputs, so nothing needs converting between the two.

### One-time setup

Requires ROS2 Humble, the `aero-hand-open` workspace built with `colcon`, and the
`aero_open_sdk` Python package installed.

**The SDK needs a patch to talk to the hand over USB.** The Aero Hand enumerates as
an ESP32-S3 native USB-JTAG-Serial device, where **RTS drives the chip's reset
line** — and pyserial asserts both DTR and RTS when it opens a port. The chip
therefore sits held in reset and never answers, and every command dies with:

```
ACK (opcode 0x31) not received within 2.0s
```

Opening with RTS already low does *not* fix it; the chip has to actually see the
reset released, so the transition is what matters. `AeroHand.__init__` needs a wake
sequence before its first command:

```python
self.ser.dtr, self.ser.rts = True, True
time.sleep(0.3)
self.ser.dtr, self.ser.rts = False, False   # release EN -> chip boots
time.sleep(0.3)
self.ser.dtr, self.ser.rts = True, False    # run mode
time.sleep(1.0)                             # let it boot before talking
```

Baudrate is irrelevant here — it's native USB-CDC, and 115200 behaves identically to
921600.

Note the SDK installs as a **non-editable copy** into site-packages by default, so
editing the repo has no effect until you re-point the install:

```bash
pip install -e /path/to/aero-hand-open/sdk
```

### Starting the hardware node

```bash
ros2 run aero_hand_open aero_hand_node --ros-args \
    -p right_port:=auto -p control_space:=joint
```

`right_port:=auto` resolves through `/dev/serial/by-id/usb-Espressif_*`, which is
the robust choice — the wake sequence resets the chip, so it re-enumerates and the
`/dev/ttyACM<N>` number changes. Never hardcode `/dev/ttyACM0`.

**Only run one node at a time.** A second instance resets the chip out from under
the first, which then holds a dead device node.

### Network isolation

If you have a FastDDS unicast profile (`FASTRTPS_DEFAULT_PROFILES_FILE`) pointing
discovery at another machine, you will see *that* machine's hand instead of your
own — including being able to command it. The symptom is a topic list that looks
correct while no data ever arrives, because `avoid_builtin_multicast` plus an
`initialPeersList` aimed elsewhere lets participants match over shared memory
without endpoint discovery completing:

```
sequence size exceeds remaining buffer
RuntimeError: No feedback from hand — is aero_hand_node running?
```

For local-only operation:

```bash
unset FASTRTPS_DEFAULT_PROFILES_FILE
export ROS_LOCALHOST_ONLY=1
ros2 daemon stop        # the daemon caches discovery config
```

`.bashrc` makes this the default, with the remote peer available via
`AERO_REMOTE=1`. Shell startup files only apply to **newly opened** terminals — a
stale terminal is the usual cause of this failing after it was "fixed."

### The bridge API

```python
with AeroHandBridge() as hand:
    hand.send_actuator_positions([0.0, 0.3, 0.35, 0.6, 0.6, 0.6, 0.6])
    hand.spin_once(timeout_sec=0.0)

    actuators = hand.get_actuator_feedback()   # 7,  radians — same space as the command
    joints    = hand.get_joint_feedback()      # 16, radians, JOINT_NAMES order
    motors    = hand.get_motor_feedback()      # 7,  radians — raw hardware
    currents  = hand.get_actuator_current_feedback()   # 7, mA — torque proxy
```

Per-actuator travel, derived from the USD joint limits and the coupling ratios
(`send_actuator_positions` clamps to these, silently):

| Actuator | Index | Upper limit (rad) |
|---|---|---|
| `thumb_abduction_actuator` | 0 | 1.745 |
| `thumb_flex_actuator` | 1 | 0.956 |
| `thumb_tendon_actuator` | 2 | 2.618 |
| `index` / `middle` / `ring` / `pinky` | 3–6 | 1.571 |

**The hardware has no joint encoders.** It publishes only `ActuatorStates` — 7 raw
*motor* angles in degrees — and the SDK's `get_joint_positions()` is an
unimplemented stub. `get_joint_feedback()` reconstructs joint positions by
inverting the tendon model. Verified against the SDK's own forward model with a
worst round-trip error of **1.1e-16**, but it is still a model estimate: tendon
stretch, and an object physically blocking a finger, are both invisible to it. For
contact detection use `get_actuator_current_feedback()`, which is real sensing.

Motor space and actuator space are *not* interchangeable — the finger tendon gain
is `22.28386 / 9.0 ≈ 2.476`, so commanding `0.8` reads back `1.9592` in raw motor
radians. `get_actuator_feedback()` returns the value directly comparable to what you
sent; `get_motor_feedback()` returns the raw reading. Expect tracking residuals of
0.5–2% on the fingers and up to ~0.056 rad on the thumb, which is ordinary servo
error and tendon compliance rather than a units problem — the thumb is worse because
it is the cross-coupled chain with the tightest limit.

### Control rates

| | Rate | Where it's set |
|---|---|---|
| Real hand sensing | **100 Hz** | `feedback_frequency` param; firmware sync-read task at `pdMS_TO_TICKS(10)` |
| Real hand commands | event-driven, unthrottled | written on every `JointControl` message |
| Sim policy | **60 Hz** | `sim.dt = 1/120` with `decimation = 2` |

Measured 100.01 Hz feedback, holding steady even while commanding at 800 Hz. That
measures the ROS/USB path, *not* servo actuation — the node publishes whatever is in
the firmware's `gMetrics` buffer, so starved servo readings would go stale without
the publish rate ever dropping. Commands and sensing share a 1 Mbps servo bus behind
a mutex, and the read task uses a **try-lock** that skips its cycle when control
holds the bus, so commands have priority over sensing by design. There is nothing to
gain above 100 Hz.

The 60 Hz sim vs 100 Hz hardware gap matters for sim-to-real. Align them with
`decimation = 1` (120 Hz) or by dropping `feedback_frequency` to 60; commanding a
100 Hz hand at 60 Hz is otherwise fine, since commands latch until the next one.

### Demo

```bash
python3 real_hand/test_ros.py
```

Finger wave, counting 1→5, rock on, thumbs up, thumb opposition, piano taps, and a
slow clench — roughly 25 s. Poses are interpolated with `smoothstep` easing and
streamed at 100 Hz to match the hand's own loop, rather than sent as step targets
the servos have to chase. `Ctrl+C` glides back to an open palm instead of freezing
mid-pose.

The `PINCH` triples (thumb abduction/flexion/curl per finger) are geometric
estimates of fingertip contact, not measured — trim them if a pinch misses or
collides on your unit.

### Troubleshooting

| Symptom | Cause |
|---|---|
| `ACK (opcode 0x31) not received` | SDK missing the DTR/RTS wake sequence, or SDK installed non-editable so the patch isn't live |
| `No feedback from hand` + `sequence size exceeds remaining buffer` | Stale terminal still has the FastDDS unicast profile; open a new shell and `ros2 daemon stop` |
| Topics listed but no data | Same as above — participants match over shared memory while endpoint discovery never completes |
| Node starts, then reads fail | Two nodes running; the second reset the chip and it re-enumerated to a new `/dev/ttyACM<N>` |
| Feedback ~2.5× the commanded value | Reading `get_motor_feedback()` where `get_actuator_feedback()` was meant |

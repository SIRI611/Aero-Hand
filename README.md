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
| `test.py` | Standalone script that spawns N envs and holds the tuned grasp pose in the viewport for whichever variant's env module it imports. No RL loop, just for visually checking a pose. |
| `run_aero_hand_parallel.py` | Standalone hand-*only* sanity check — no object, no `DirectRLEnv`, no RL action space. Builds the scene directly from `AeroHandSceneCfg` with the raw `InteractiveScene`/`SimulationContext` APIs and drives all 16 joints with one shared sinusoidal open/close curl, ignoring the 7-tendon grouping and per-joint limits entirely. See [Sanity-checking the scene](#sanity-checking-the-scene). |

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

# run_aero_hand_parallel.py
"""
Usage:
    ./isaaclab.sh -p run_aero_hand_parallel.py --num_envs 100
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run N parallel Aero Hand instances.")
parser.add_argument("--num_envs", type=int, default=100, help="Number of hands to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

from aero_hand_scene_cfg import AeroHandSceneCfg

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["aero_hand"]
    print("is_fixed_base:", robot.is_fixed_base)
    sim_dt = sim.get_physics_dt()

    # Resolve joint indices by name once — never assume tensor column order
    joint_ids, joint_names = robot.find_joints("right_.*")
    print(f"Controlling {len(joint_ids)} joints per hand: {joint_names}")

    t = 0.0
    count = 0

    while simulation_app.is_running():
        # Reset every 300 steps, giving each hand a fresh (here: randomized) pose
        if (count % 300 == 0):
            count = 0

            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            robot.data.root_state_w[:] = root_state

            joint_pos = robot.data.default_joint_pos.clone()
            joint_pos[:, joint_ids] = torch.rand(
                scene.num_envs, len(joint_ids), device=joint_pos.device
            ) * 0.3   # small random offset from open — widen this once limits are tuned
            joint_vel = torch.zeros_like(robot.data.default_joint_vel)

            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            scene.reset()
            print("[INFO]: Reset all hands to a new initial joint state.")

        # Simple demo controller — all hands curl together, thumb stays neutral
        t += sim_dt
        curl = 0.5 * (1 + math.sin(t * 0.5))
        targets = robot.data.joint_pos.clone()
        targets[:, joint_ids] = curl

        robot.set_joint_position_target(targets)
        scene.write_data_to_sim()

        print("env_origins[:5]:\n", scene.env_origins[:5])
        print("root_pos_w[:5]:\n", robot.data.root_pos_w[:5])

        sim.step()
        count += 1
        scene.update(sim_dt)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([3.0, 3.0, 3.0], [0.0, 0.0, 0.3])

    scene_cfg = AeroHandSceneCfg(num_envs=args_cli.num_envs, env_spacing=1.0, replicate_physics=True)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()

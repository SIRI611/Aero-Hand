import gymnasium as gym

gym.register(
    id="Isaac-AeroHand-Grasp-Direct-v0",
    entry_point="aero_hand_grasp_env:AeroHandGraspEnv",
    kwargs={"env_cfg_entry_point": "aero_hand_grasp_env:AeroHandGraspEnvCfg"},
)

# to run: ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task=Isaac-AeroHand-Grasp-Direct-v0

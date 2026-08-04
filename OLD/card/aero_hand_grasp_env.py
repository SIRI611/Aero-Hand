# aero_hand_grasp_env.py
from __future__ import annotations
import math

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass

from aero_hand_cfg import AERO_HAND_CFG
from credit_card_cfg import CREDIT_CARD_CFG


@configclass
class AeroHandGraspEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    action_space = 16              # one normalized target per joint, in [-1, 1]
    observation_space = 42         # 16 joint_pos + 16 joint_vel + 3 credit_card_pos + 3 credit_card_vel + 4 credit_card_quat
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # assets
    hand_cfg: ArticulationCfg = AERO_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    credit_card_cfg: RigidObjectCfg = CREDIT_CARD_CFG.replace(prim_path="/World/envs/env_.*/CreditCard")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=100, env_spacing=0.6, replicate_physics=True
    )

    # joint limits (radians) — reused from your sim_to_real_isaac.py script;
    # replace with exact URDF values once confirmed TODO: check if these are correct for the Aero Hand Open
    # joint_lower = [-0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # joint_upper = [ 0.35, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57,
    #                 1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57]
    # joint_lower = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # joint_upper = [ 1.745, 0.96, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57,
    #                 1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.57]

    # joint limits (degrees)
    joint_limits_deg = {
        "right_index_mcp_flex":  (0.0, 90.000206),
        "right_index_pip":       (0.0, 90.000206),
        "right_index_dip":       (0.0, 90.000206),
        "right_middle_mcp_flex": (0.0, 90.000206),
        "right_middle_pip":      (0.0, 90.000206),
        "right_middle_dip":      (0.0, 90.000206),
        "right_ring_mcp_flex":   (0.0, 90.000206),
        "right_ring_pip":        (0.0, 90.000206),
        "right_ring_dip":        (0.0, 90.000206),
        "right_pinky_mcp_flex":  (0.0, 90.000206),
        "right_pinky_pip":       (0.0, 90.000206),
        "right_pinky_dip":       (0.0, 90.000206),
        "right_thumb_cmc_abd":   (0.0, 99.99833),
        "right_thumb_cmc_flex":  (0.0, 54.76903),
        "right_thumb_mcp":       (0.0, 90.000206),
        "right_thumb_ip":        (0.0, 90.000206),
    }

    # grasp / reset
    # grasp_joint_pos = 0.0            # radians — TUNE against your credit_card's actual diameter
    # grasp / reset — one value per joint (radians). Keyed by name so it's
    # immune to whatever internal order PhysX assigns to the joints.
    grasp_joint_pos = {
        "right_thumb_cmc_abd":  1.25, # base of thumb
        "right_thumb_cmc_flex": 0.1, # moves thumb across the palm
        "right_thumb_mcp":      0.7, # moves thumb up/down
        "right_thumb_ip":       1.0, # tip of thumb

        "right_index_mcp_flex": 1.6,
        "right_index_pip":      0.55,
        "right_index_dip":      0.6,

        "right_middle_mcp_flex": 0.0,
        "right_middle_pip":      0.0,
        "right_middle_dip":      0.0,

        "right_ring_mcp_flex": 0.0,
        "right_ring_pip":      0.0,
        "right_ring_dip":      0.0,

        "right_pinky_mcp_flex": 0.0,
        "right_pinky_pip":      0.0,
        "right_pinky_dip":      0.0,
    }
    # grasp_joint_pos = {
    #         "right_thumb_cmc_abd":  0.0, # base of thumb
    #         "right_thumb_cmc_flex": 0.0, # moves thumb across the palm
    #         "right_thumb_mcp":      0.4, # moves thumb up/down
    #         "right_thumb_ip":       0.8, # tip of thumb
    
    #         "right_index_mcp_flex": 0.9,
    #         "right_index_pip":      1.0,
    #         "right_index_dip":      1.0,
    
    #         "right_middle_mcp_flex": 0.8,
    #         "right_middle_pip":      0.9,
    #         "right_middle_dip":      0.9,
    
    #         "right_ring_mcp_flex": 0.9,
    #         "right_ring_pip":      1.0,
    #         "right_ring_dip":      1.0,
    
    #         "right_pinky_mcp_flex": 0.9,
    #         "right_pinky_pip":      1.0,
    #         "right_pinky_dip":      1.0,
    #     }
    credit_card_drop_height_thresh = 0.15    # meters of downward drift counted as "dropped"

    # reward scales
    rew_scale_credit_card_height = 5.0
    rew_scale_credit_card_still = -0.1
    rew_scale_action_rate = -0.01
    rew_scale_drop_penalty = -10.0


class AeroHandGraspEnv(DirectRLEnv):
    cfg: AeroHandGraspEnvCfg

    def __init__(self, cfg: AeroHandGraspEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._joint_ids, self._joint_names = self.hand.find_joints("right_.*")

        lower_deg = [self.cfg.joint_limits_deg[name][0] for name in self._joint_names]
        upper_deg = [self.cfg.joint_limits_deg[name][1] for name in self._joint_names]
        self._joint_lower = torch.tensor([math.radians(d) for d in lower_deg], device=self.device)
        self._joint_upper = torch.tensor([math.radians(d) for d in upper_deg], device=self.device)

        grasp_values = [self.cfg.grasp_joint_pos[name] for name in self._joint_names]
        self._grasp_joint_pos = torch.tensor(grasp_values, device=self.device)

        self.joint_pos = self.hand.data.joint_pos
        self.joint_vel = self.hand.data.joint_vel

        # per-env credit_card height captured at reset time, used to detect "has it dropped"
        self._initial_credit_card_height = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        self.hand = Articulation(self.cfg.hand_cfg)
        self.credit_card = RigidObject(self.cfg.credit_card_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["hand"] = self.hand
        self.scene.rigid_objects["credit_card"] = self.credit_card

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        # map normalized [-1, 1] actions to each joint's real radian range
        targets = self._joint_lower + (self.actions + 1.0) * 0.5 * (self._joint_upper - self._joint_lower)
        self.hand.set_joint_position_target(targets, joint_ids=self._joint_ids) # setting joint position targets (what RL learns)

    def _get_observations(self) -> dict:
        self.joint_pos = self.hand.data.joint_pos
        self.joint_vel = self.hand.data.joint_vel

        credit_card_pos_rel = self.credit_card.data.root_pos_w - self.scene.env_origins
        credit_card_lin_vel = self.credit_card.data.root_lin_vel_w
        credit_card_quat = self.credit_card.data.root_quat_w

        obs = torch.cat(
            (
                self.joint_pos[:, self._joint_ids],
                self.joint_vel[:, self._joint_ids],
                credit_card_pos_rel,
                credit_card_lin_vel,
                credit_card_quat,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        credit_card_height = self.credit_card.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        credit_card_speed = torch.norm(self.credit_card.data.root_lin_vel_w, dim=-1)

        dropped = (self._initial_credit_card_height - credit_card_height) > self.cfg.credit_card_drop_height_thresh
        held = ~dropped

        rew_height = self.cfg.rew_scale_credit_card_height * held.float()
        rew_still = self.cfg.rew_scale_credit_card_still * credit_card_speed
        rew_drop = self.cfg.rew_scale_drop_penalty * dropped.float()
        rew_action_rate = self.cfg.rew_scale_action_rate * torch.sum(torch.square(self.actions), dim=-1)

        return rew_height + rew_still + rew_drop + rew_action_rate

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        credit_card_height = self.credit_card.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        dropped = (self._initial_credit_card_height - credit_card_height) > self.cfg.credit_card_drop_height_thresh
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return dropped, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super()._reset_idx(env_ids)

        # --- Hand: spawn already closed around the credit_card ---
        # joint_pos = self.hand.data.default_joint_pos[env_ids].clone()
        # joint_pos[:, self._joint_ids] = self.cfg.grasp_joint_pos
        # joint_vel = torch.zeros_like(self.hand.data.default_joint_vel[env_ids])
        # self.hand.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        joint_pos = self.hand.data.default_joint_pos[env_ids].clone()
        joint_pos[:, self._joint_ids] = self._grasp_joint_pos   # broadcasts (16,) across the batch
        joint_vel = torch.zeros_like(self.hand.data.default_joint_vel[env_ids])
        self.hand.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_hand_root = self.hand.data.default_root_state[env_ids].clone()
        default_hand_root[:, :3] += self.scene.env_origins[env_ids]
        self.hand.write_root_pose_to_sim(default_hand_root[:, :7], env_ids)
        self.hand.write_root_velocity_to_sim(default_hand_root[:, 7:], env_ids)

        # --- Credit Card: spawn positioned inside the grasp ---
        default_credit_card_root = self.credit_card.data.default_root_state[env_ids].clone()
        default_credit_card_root[:, :3] += self.scene.env_origins[env_ids]
        self.credit_card.write_root_pose_to_sim(default_credit_card_root[:, :7], env_ids)
        self.credit_card.write_root_velocity_to_sim(default_credit_card_root[:, 7:], env_ids)

        self._initial_credit_card_height[env_ids] = default_credit_card_root[:, 2] - self.scene.env_origins[env_ids, 2]
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
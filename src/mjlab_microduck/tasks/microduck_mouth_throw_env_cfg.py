"""Microduck mouth-throw task.

One-shot episodic policy that: crouches so the beak (``mouth_tip``) reaches a pen
on the ground, closes the mouth to grip it, rears up (windup), then whips the
head/neck upward and OPENS the mouth to fling the pen up, finally returning to a
clean standing pose. This is the first task that drives the **new 15th actuated
``mouth`` joint** (bill + ``mouth_tip`` split out of ``jaw_soft`` by add_mouth.py).

⚠️ CONTRACT CHANGE (deliberate, scoped to this env): the mouth model has a 15th
actuator, so the BAM actuator regex picks it up and the ACTION becomes 15-D, and
the ``joint_pos``/``joint_vel`` obs each grow by one (mouth). This env is
STANDALONE — it does not share the 14-joint / 61-D hot-swap contract of the other
tasks (which never use this robot cfg). See docs/superpowers/specs/
2026-mouth-throw-design.md.

Phase encoding (5 segments, cyclic phase command in the ``twist`` slot,
``randomize_phase=False`` so every episode starts at phase 0):
    [0, LOWER_END)          crouch, mouth OPEN toward the ground
    [LOWER_END, GRAB_END)   mouth CLOSED (grip), held-pen weight turns ON
    [GRAB_END, WINDUP_END)  rise, rear the head back, KEEP mouth closed (hold)
    [WINDUP_END, THROW_END) whip head up + OPEN mouth -> payload OFF (release)
    [THROW_END, 1.0)        return to a clean stand

The pen is modeled as a per-env held weight at the beak (10–40 g, sampled each
episode) applied by the payload hook — the repo's established idiom for "holding
an object in the mouth". The throw itself is rewarded by the mouth_tip UPWARD
velocity + a clean (upright, near-HOME) return; no fragile contact-grasp physics.
"""

import math
from copy import deepcopy

# Symmetry — OFF (the throw is a forward/sagittal, non-symmetric maneuver).
ENABLE_SYMMETRY = False

# ── Domain randomisation toggles (matched to the velocity / ground-pick envs) ──
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = False  # head moves hard for the throw; keep CoM DR off
ENABLE_KP_RANDOMIZATION              = False
ENABLE_KD_RANDOMIZATION              = False
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True   # scales BAM friction budget per-env
ENABLE_JOINT_DAMPING_RANDOMIZATION   = False
ENABLE_ARMATURE_RANDOMIZATION        = True   # reflected rotor inertia (affects BAM)
ENABLE_VELOCITY_PUSHES               = False  # a throw is ballistic — no mid-episode pushes
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True
ENABLE_BASE_ORIENTATION_RANDOMIZATION = False
ENABLE_NECK_OFFSET_RANDOMIZATION     = False

# ── Ranges (matched to the velocity env) ──────────────────────────────────────
COM_RANDOMIZATION_RANGE          = 0.003
HEAD_COM_RANDOMIZATION_RANGE     = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
KP_RANDOMIZATION_RANGE           = (0.85, 1.15)
KD_RANDOMIZATION_RANGE           = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ARMATURE_RANDOMIZATION_RANGE     = (0.9, 1.1)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0
ENCODER_BIAS_RANGE               = (-0.015, 0.015)

# ── Throw segment boundaries (phase fractions; single period, not randomized) ──
#   LOWER | GRAB | RISE/WINDUP | THROW | RETURN
LOWER_END  = 0.20
GRAB_END   = 0.30
WINDUP_END = 0.60
THROW_END  = 0.85
GP_PERIOD  = 4.0
NUM_STEPS_PER_ENV = 24

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_MOUTH_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MICRODUCK_ROUGH_TERRAINS_CFG,
    HEAD_BODY_NAMES,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_mouth_throw_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the Microduck mouth-throw environment configuration."""

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # Head-on-ground impact sensor (neck subtree). Penalizes face-planting.
    head_impact_cfg = ContactSensorCfg(
        name="head_impact_contact",
        primary=ContactMatch(mode="subtree", pattern="neck", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("force",),
        reduce="netforce",
        num_slots=1,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_MOUTH_ROBOT_CFG}
    cfg.scene.sensors  = (feet_ground_cfg, self_collision_cfg, head_impact_cfg)
    cfg.viewer.body_name = "trunk_base"

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards: remove locomotion terms ──────────────────────────────────────
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Rewards: mouth-throw objectives ───────────────────────────────────────

    # LOWER+GRAB: mouth_tip close to the ground (it reaches the pen). No-touch
    # kept by head_impact_penalty (below) — the beak hovers just above the pen.
    # Gated to [0, GRAB_END) so it stops rewarding proximity once the rise begins.
    cfg.rewards["mouth_ground_proximity"] = RewardTermCfg(
        func=microduck_mdp.mouth_throw_ground_proximity,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=["mouth_tip"]),
            "std": 0.10,
            "target_height": 0.0,
            "command_name": "twist",
            "grab_end": GRAB_END,
        },
    )

    # Mouth OPEN while lowering (reach to the pen) and while throwing (release).
    cfg.rewards["mouth_open"] = RewardTermCfg(
        func=microduck_mdp.mouth_open_during_throw,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=["mouth_tip"]),
            "command_name": "twist",
            "open_target": 0.4,
            "std": 0.25,
            "lower_end": LOWER_END,
            "throw_end": THROW_END,
        },
    )

    # Mouth CLOSED while gripping + holding (grab..windup).
    cfg.rewards["mouth_closed_hold"] = RewardTermCfg(
        func=microduck_mdp.mouth_closed_during_hold,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=["mouth_tip"]),
            "command_name": "twist",
            "closed_target": 0.0,
            "std": 0.15,
            "grab_end": GRAB_END,
            "windup_end": WINDUP_END,
        },
    )

    # THROW: reward the mouth_tip UPWARD velocity during the throw window (the fling).
    cfg.rewards["mouth_throw_upward_velocity"] = RewardTermCfg(
        func=microduck_mdp.mouth_throw_upward_velocity,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=["mouth_tip"]),
            "command_name": "twist",
            "target": 1.2,
            "std": 0.6,
            "windup_end": WINDUP_END,
            "throw_end": THROW_END,
        },
    )

    # Whip quality: net upward speed through the throw window (potential, not a
    # rate-limited jackpot — nothing to park on).
    cfg.rewards["mouth_throw_peak"] = RewardTermCfg(
        func=microduck_mdp.mouth_throw_peak_upward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=["mouth_tip"]),
            "command_name": "twist",
            "grab_end": GRAB_END,
            "windup_end": WINDUP_END,
            "throw_end": THROW_END,
        },
    )

    # Payload hook (weight 0): applies the held-pen weight at the beak from grab,
    # released by throw (models "holding then throwing the pen").
    cfg.rewards["mouth_payload_release"] = RewardTermCfg(
        func=microduck_mdp.apply_mouth_payload_release,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["jaw_soft"], site_names=["mouth_tip"]
            ),
            "command_name": "twist",
            "grab_end": GRAB_END,
            "throw_end": THROW_END,
        },
    )

    # RETURN: legs + neck back to HOME (tight std on neck to avoid overshoot),
    # and the trunk upright, gated to the return window.
    # N.B. the mouth model has a 15th servo joint (mouth) between the neck and the
    # right leg, so the servo-relative index of the right leg is SHIFTED by +1 vs
    # the 14-joint layout: left leg [0-4], neck [5-8], mouth [9], right leg [10-14].
    # `ground_pick_return_pose_phased` indexes `_servo_joint_pos` (all 15
    # non-passive joints) with this list, so reusing the 14-joint indices here
    # would select `mouth` (9) and drop `right_ankle` (14) — the tests below pin
    # this exact head/leg split against the actual model.
    _LEG_JOINTS = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
    _NECK_JOINTS = [5, 6, 7, 8]
    cfg.rewards["mouth_throw_return_pose_legs"] = RewardTermCfg(
        func=microduck_mdp.ground_pick_return_pose_phased,
        weight=4.0,
        params={
            "std": 0.3,
            "command_name": "twist",
            "joint_indices": _LEG_JOINTS,
            "hold_end": WINDUP_END,
            "rise_end": THROW_END,
        },
    )
    cfg.rewards["mouth_throw_return_pose_neck"] = RewardTermCfg(
        func=microduck_mdp.ground_pick_return_pose_phased,
        weight=4.0,
        params={
            "std": 0.15,
            "command_name": "twist",
            "joint_indices": _NECK_JOINTS,
            "hold_end": WINDUP_END,
            "rise_end": THROW_END,
        },
    )
    cfg.rewards["mouth_throw_return_upright"] = RewardTermCfg(
        func=microduck_mdp.ground_pick_return_upright_phased,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.4,
            "command_name": "twist",
            "hold_end": WINDUP_END,
            "rise_end": THROW_END,
        },
    )

    # ── Rewards: regularisation ───────────────────────────────────────────────
    cfg.rewards["action_rate_l2"] = RewardTermCfg(
        func=mdp.action_rate_l2, weight=-2.0
    )
    cfg.rewards["neck_action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_action_rate_l2, weight=-1.0
    )
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2, weight=-5e-3
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )
    cfg.rewards["head_impact_penalty"] = RewardTermCfg(
        func=microduck_mdp.body_impact_cost,
        weight=-2.0,
        params={"sensor_name": head_impact_cfg.name, "threshold": 1.0},
    )

    # ── Stability (kept from velocity, tuned for this task) ───────────────────
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 0.2
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["soft_landing"].weight = -1e-5

    # ── Observations (same layout as walking/ground-pick + the mouth joint) ───
    del cfg.observations["actor"].terms["base_lin_vel"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )
    for term in (cfg.observations["actor"].terms["base_ang_vel"],
                 cfg.observations["actor"].terms[gravity_term_name]):
        term.delay_min_lag = 0
        term.delay_max_lag = 3
        term.delay_update_period = 64
        term.noise = Unoise(n_min=-0.03, n_max=0.03) if term is cfg.observations["actor"].terms["base_ang_vel"] \
            else Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)

    # IMU mounting-misalignment DR (obs-level) is already wired by
    # make_velocity_env_cfg (per-env rotation of projected_gravity + base_ang_vel
    # at IMU_ORIENTATION_RANDOMIZATION_ANGLE) — nothing to add here.

    # ARMATURE event (mirror velocity).
    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    # ── Command: cyclic 5-segment phase encoding, one-shot ────────────────────
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs  = 0.0
    # randomize_phase=False → every episode starts at phase 0 (one-shot throw).
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(
        **{**vars(command),
           "class_type": microduck_mdp.GroundPickPhaseCommand,
           "period": GP_PERIOD,
           "randomize_phase": False}
    )

    # ── Terminations ──────────────────────────────────────────────────────────
    # Terminate on NaN physics (extreme contact impulses) before it corrupts obs.
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )
    # Episodic one-shot: the episode ends on the period boundary. No explicit
    # fall termination (matches ground_pick) — the throw is one clean ballistic
    # cycle, and a mid-fall cutoff would truncate the return-to-stand phase.

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["reset_mouth_tip_vel"] = EventTermCfg(
        func=microduck_mdp.reset_mouth_tip_vel_cache,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", site_names=["mouth_tip"])},
    )

    # Pen-held weight mass, sampled per episode (10–40 g).
    cfg.events["sample_mouth_payload"] = EventTermCfg(
        func=microduck_mdp.sample_mouth_payload,
        mode="reset",
        params={"min_kg": 0.01, "max_kg": 0.04},
    )

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.12, 0.13)

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )
    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )
    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    # ── Terrain ───────────────────────────────────────────────────────────────
    if not rough:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
    else:
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = MICRODUCK_ROUGH_TERRAINS_CFG
        if play:
            cfg.scene.terrain.terrain_generator.curriculum = False
            cfg.scene.terrain.terrain_generator.num_cols = 5
            cfg.scene.terrain.terrain_generator.num_rows = 5

    # ── Curriculum ────────────────────────────────────────────────────────────
    if not rough:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": -0.6},
                {"step": 250 * 24,   "weight": -1.2},
                {"step": 500 * 24,   "weight": -2.0},
            ],
        },
    )
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────
MicroduckMouthThrowRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="mouth_throw",
    run_name="mouth_throw",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=30_000,
)

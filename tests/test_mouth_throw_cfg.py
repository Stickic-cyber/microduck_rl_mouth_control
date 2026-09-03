from mjlab_microduck.robot.microduck_constants import MICRODUCK_MOUTH_ROBOT_CFG
from mjlab_microduck.tasks.microduck_mouth_throw_env_cfg import (
    GRAB_END,
    LOWER_END,
    THROW_END,
    WINDUP_END,
    make_microduck_mouth_throw_env_cfg,
)
from mjlab_microduck.tasks.mdp import GroundPickPhaseCommand


def test_mouth_throw_cfg_command_is_one_shot_phase():
    cfg = make_microduck_mouth_throw_env_cfg()
    cmd = cfg.commands["twist"]
    assert cmd.class_type is GroundPickPhaseCommand
    # One-shot: every episode starts at phase 0 (standing) — not randomised.
    assert cmd.randomize_phase is False


def test_mouth_throw_cfg_uses_mouth_robot():
    cfg = make_microduck_mouth_throw_env_cfg()
    # The mouth-throw env must run the 15-joint mouth model (the enabler of the
    # task). Its BAM actuator regex (^(?!passive_).*) picks up the 15th `mouth`
    # actuator → 15-D action / mouth joint in the obs.
    robot = cfg.scene.entities["robot"]
    assert robot.spec_fn is MICRODUCK_MOUTH_ROBOT_CFG.spec_fn


def test_mouth_throw_rewards_wired():
    cfg = make_microduck_mouth_throw_env_cfg()
    r = cfg.rewards
    # Reach: mouth_tip near the ground (gated to the LOWER+GRAB window).
    assert "mouth_ground_proximity" in r
    assert r["mouth_ground_proximity"].weight == 3.0
    assert r["mouth_ground_proximity"].params["grab_end"] == GRAB_END
    # Mouth opens while lowering + throwing; closes while gripping/holding.
    assert "mouth_open" in r and r["mouth_open"].weight == 3.0
    assert "mouth_closed_hold" in r and r["mouth_closed_hold"].weight == 3.0
    assert r["mouth_closed_hold"].params["grab_end"] == GRAB_END
    assert r["mouth_closed_hold"].params["windup_end"] == WINDUP_END
    # Throw: upward mouth_tip velocity during the throw window.
    assert "mouth_throw_upward_velocity" in r
    assert r["mouth_throw_upward_velocity"].weight == 4.0
    assert r["mouth_throw_upward_velocity"].params["windup_end"] == WINDUP_END
    assert r["mouth_throw_upward_velocity"].params["throw_end"] == THROW_END
    # Whip quality potential + no-touch guard.
    assert "mouth_throw_peak" in r
    assert "head_impact_penalty" in r and r["head_impact_penalty"].weight == -2.0
    # Return to stand (legs + neck + upright), gated to the return window.
    assert "mouth_throw_return_pose_legs" in r
    assert "mouth_throw_return_pose_neck" in r
    assert "mouth_throw_return_upright" in r and r["mouth_throw_return_upright"].weight == 3.0
    assert r["mouth_throw_return_upright"].params["hold_end"] == WINDUP_END


def test_mouth_throw_payload_wired():
    cfg = make_microduck_mouth_throw_env_cfg()
    # Pen-held weight hook (weight 0) + per-episode mass DR event.
    assert "mouth_payload_release" in cfg.rewards
    assert cfg.rewards["mouth_payload_release"].weight == 0.0
    assert "sample_mouth_payload" in cfg.events
    assert cfg.events["sample_mouth_payload"].params["min_kg"] == 0.01
    assert cfg.events["sample_mouth_payload"].params["max_kg"] == 0.04
    # The finite-difference tip-velocity cache is re-seeded on reset.
    assert "reset_mouth_tip_vel" in cfg.events


def test_mouth_throw_boundaries_ordered():
    assert 0.0 < LOWER_END < GRAB_END < WINDUP_END < THROW_END < 1.0


def test_mouth_throw_variants_build():
    cfg = make_microduck_mouth_throw_env_cfg(rough=True)
    assert "mouth_throw_upward_velocity" in cfg.rewards
    assert "mouth_ground_proximity" in cfg.rewards


def test_mouth_throw_play_variant_builds():
    cfg = make_microduck_mouth_throw_env_cfg(play=True)
    assert "mouth_throw_upward_velocity" in cfg.rewards


import mujoco  # noqa: E402
import torch  # noqa: E402

from mjlab_microduck.robot.microduck_constants import MICRODUCK_MOUTH_XML  # noqa: E402
from mjlab_microduck.tasks.mdp import (  # noqa: E402
    _mouth_throw_gate_grab_hold,
    _mouth_throw_gate_lower,
    _mouth_throw_gate_return,
    _mouth_throw_gate_windup,
)


def _servo_joint_names(model: mujoco.MjModel):
    """Non-freejoint, non-passive joint names in entity order (= the `^(?!passive_).*`
    joint_pos/joint_vel obs + 15-D action order the mouth model uses)."""
    names = []
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name.endswith("_freejoint") or name.startswith("passive_"):
            continue
        names.append(name)
    return names


def test_mouth_model_joint_layout_and_15d_action():
    """Pin the mouth model's servo joint order — the one assumption every reward
    index list in the env cfg depends on.

    The mouth joint sits BETWEEN the neck (5-8) and the right leg, so the right leg
    is shifted +1 vs the 14-joint layout: left[0-4] neck[5-8] mouth[9] right[10-14].
    This is exactly what makes ground_pick's _LEG_JOINTS (right leg at 9-13) WRONG
    here. Also assert 15 actuated joints -> 15-D action (mouth included).
    """
    model = mujoco.MjModel.from_xml_path(str(MICRODUCK_MOUTH_XML))
    names = _servo_joint_names(model)
    assert names[0:5] == [
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    ]
    assert names[5:9] == ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
    assert names[9] == "mouth"
    assert names[10:15] == [
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    ]
    assert len(names) == 15
    # 15 actuated joints -> 15 actuators (14 + mouth). The BAM target regex
    # `^(?!passive_).*` picks up all of them -> action = 15-D.
    actuated = []
    for i in range(model.nu):
        trnid = model.actuator_trnid[i, 0]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, trnid)
        actuated.append(name)
    assert "mouth" in actuated
    assert len(actuated) == 15


def test_mouth_throw_return_pose_indices_match_model():
    """The env cfg's leg/neck index lists must match the mouth model's layout; a
    stale 14-joint _LEG_JOINTS would select `mouth` and drop `right_ankle`."""
    model = mujoco.MjModel.from_xml_path(str(MICRODUCK_MOUTH_XML))
    servo = _servo_joint_names(model)
    cfg = make_microduck_mouth_throw_env_cfg()
    leg_idx = cfg.rewards["mouth_throw_return_pose_legs"].params["joint_indices"]
    neck_idx = cfg.rewards["mouth_throw_return_pose_neck"].params["joint_indices"]
    # Leg reward must cover exactly the 10 leg actuated joints (no mouth, no neck).
    assert [servo[i] for i in leg_idx] == [
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    ]
    assert [servo[i] for i in neck_idx] == [
        "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    ]
    assert 9 not in leg_idx  # `mouth` must NOT be rewdarded as a leg joint
    assert 14 in leg_idx     # right_ankle must be included


def test_mouth_throw_phase_gates():
    """Cheap pure-function test on the 5-segment gate helpers."""
    phase = torch.tensor([0.0, 0.12, 0.25, 0.45, 0.70, 0.90])
    # LOWER only before GRAB_END.
    assert torch.allclose(
        _mouth_throw_gate_lower(phase, lower_end=0.30),
        torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
    )
    # GRAB..WINDUP.
    assert torch.allclose(
        _mouth_throw_gate_grab_hold(phase, grab_end=0.30, windup_end=0.60),
        torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    )
    # WINDUP..THROW.
    assert torch.allclose(
        _mouth_throw_gate_windup(phase, windup_end=0.60, throw_end=0.85),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    )
    # RETURN (>= throw_end).
    assert torch.allclose(
        _mouth_throw_gate_return(phase, throw_end=0.85),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    )

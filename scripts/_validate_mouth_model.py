"""Validate the derived `*_mouth.xml`: compiles, has a `mouth` joint + actuator, and
driving the mouth swings the mouth_tip while the robot otherwise still holds its pose."""
from __future__ import annotations

import argparse

import mujoco
import numpy as np

PATH = (__import__("pathlib").Path(__file__).resolve().parents[1]
        / "src" / "mjlab_microduck" / "robot" / "microduck"
        / "robot_groundcontact_mouth.xml")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=str(PATH))
    ap.add_argument("--play", action="store_true",
                    help="show interactive viewer stepping the mouth")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    # Joint + actuator inventory.
    joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
              for i in range(model.njnt)]
    actuators = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                 for i in range(model.nu)]
    print("njnt =", model.njnt, "actuated =", model.nu)
    print("joints:", joints)
    print("actuators:", actuators)

    assert "mouth" in joints, "mouth joint missing"
    assert "mouth" in actuators, "mouth actuator missing"
    # head_roll + mouth both present, mouth is a SEPARATE dof from head_roll.
    assert joints.count("head_roll") == 1 and joints.count("mouth") == 1

    mouth_id = joints.index("mouth")
    mouth_act = actuators.index("mouth")
    mouth_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mouth_tip")
    assert mouth_tip_id >= 0, "mouth_tip site missing"

    # The mouth hinge's qpos slot: 7 (freejoint) + one per earlier hinge.
    n_before = sum(1 for j in joints[:mouth_id] if j != "trunk_base_freejoint")
    mouth_qidx = 7 + n_before

    # Isolate the mouth: set its qpos directly, hold the rest at rest, no dynamics.
    mujoco.mj_forward(model, data)
    tip_closed = data.site_xpos[mouth_tip_id].copy()
    data.qpos[mouth_qidx] = 0.4  # open the bill
    mujoco.mj_forward(model, data)
    tip_open = data.site_xpos[mouth_tip_id].copy()
    delta = float(np.linalg.norm(tip_open - tip_closed))
    print(f"mouth qpos idx {mouth_qidx}")
    print(f"mouth closed tip: {tip_closed}")
    print(f"mouth open   tip: {tip_open}")
    print(f"mouth_tip displacement (mouth only, static): {delta:.4f} m")

    assert delta > 0.005, "mouth_tip did NOT follow the mouth joint — split failed"
    print("OK: mouth hinge independently drives the bill/mouth_tip.")

    # Sanity: the rest of the head (head_camera, on jaw_soft) must NOT move with mouth.
    head_cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "head_camera")
    cam_before = data.site_xpos[head_cam].copy()
    data.qpos[mouth_qidx] = -0.4
    mujoco.mj_forward(model, data)
    cam_after = data.site_xpos[head_cam].copy()
    cam_delta = float(np.linalg.norm(cam_after - cam_before))
    print(f"head_camera displacement when mouth moves: {cam_delta:.4f} m")
    assert cam_delta < 1e-6, "head_camera moved with the mouth — split too wide"
    print("OK: only the bill moved; the rest of the head is static.")


if __name__ == "__main__":
    raise SystemExit(main())

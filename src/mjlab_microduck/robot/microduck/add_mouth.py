#!/usr/bin/env python3
"""Inject a 15th actuated ``mouth`` joint into an onshape-to-robot MJCF export.

The sim model's mouth is not independently actuated: the whole beak/face is fused
into the rigid ``jaw_soft`` body carried by the ``head_roll`` hinge, and the walking
model has no mouth joint at all. The real robot, however, has a dedicated mouth servo
(Dynamixel ID 34, ~ -5/+30 deg) that rocks the bill open/closed.

To make the mouth RL-trainable we split the bill meshes (``jaw``, ``soft_mouth_top``,
``jaw_soft`` shell) and the ``mouth_tip`` site out of ``jaw_soft`` into a new child
``<body name="mouth">``, driven by a ``mouth`` hinge + a new ``mouth`` position
actuator. The bill keeps the same geometry at the mouth=0 pose (mesh positions are
re-anchored to the new pivot), so the model is visually unchanged when the mouth is
closed. Contact geoms stay parented under ``jaw_soft``; only the bill + mouth_tip
move.

Follows the same text post-processing convention as ``add_backlash.py`` — it is meant
to run once to produce a derived model committed alongside the originals.

    python3 add_mouth.py robot_groundcontact.xml --out robot_groundcontact_mouth.xml
"""

import argparse
import re
import sys
from pathlib import Path

ATTR_RE = re.compile(r'(\w+)="([^"]*)"')

# Bill meshes that become the new `mouth` body (moved out of jaw_soft).
# Geom onshape exports carry no `name` — the part is selected by `mesh="..."`.
BILL_MESHES = {"jaw", "soft_mouth_top", "jaw_soft"}
BILL_SITES = {"mouth_tip"}

MOUTH_RANGE = (-0.52, 0.52)   # ~30 deg open / closed
MOUTH_MASS = 0.012            # bill is light vs the head (0.19 kg)
# Mouth tip offset from the hinge pivot (for the little-bill inertia we synthesize).
MOUTH_ARM_TIP = [-0.0036, 0.0, -0.0597]


def parse_attrs(line: str) -> dict:
    return dict(ATTR_RE.findall(line))


def shift_pos(line: str, pivot: list[float]) -> str:
    """Return `line` with its `pos` shifted so geometry is identical at mouth=0
    (new parent body sits at `pivot`, identity quat). Onshape geom/site lines are
    self-closing singles; comments pass through unchanged."""
    if "<!--" in line:
        return line
    attrs = parse_attrs(line)
    if attrs.get("pos"):
        x, y, z = (float(v) for v in attrs["pos"].split())
        attrs["pos"] = " ".join(f"{a - p:.17g}" for a, p in zip((x, y, z), pivot))
    tag = re.match(r'^\s*(<\w+)', line).group(1)
    rest = " ".join(f'{k}="{v}"' for k, v in attrs.items())
    return f"      {tag} {rest} />\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("xml", help="MJCF file to read (groundcontact model)")
    ap.add_argument("--out", required=True, help="output MJCF file")
    ap.add_argument("--pivot", default=" ".join(map(str, [-0.0045, 0.0, -0.018])),
                    help="mouth hinge pivot in jaw_soft frame (x y z)")
    args = ap.parse_args()

    pivot = [float(v) for v in args.pivot.split()]
    pivot_s = " ".join(f"{p:.17g}" for p in pivot)

    with open(args.xml) as f:
        raw = f.read()
    lines = raw.splitlines(keepends=True)

    # ---- Locate jaw_soft body open/close ----
    jaw_open = next(i for i, l in enumerate(lines)
                    if re.search(r'<body\b[^>]*name="jaw_soft"', l))
    depth = 0
    jaw_close = None
    for i in range(jaw_open, len(lines)):
        depth += lines[i].count("<body") - lines[i].count("</body>")
        if "</body>" in lines[i] and depth == 0:
            jaw_close = i
            break
    assert jaw_close is not None, "jaw_soft body not closed"

    # ---- Partition jaw_soft children: bill vs keep ----
    body_lines = lines[jaw_open + 1:jaw_close]
    bill, keep = [], []
    i = 0
    n = len(body_lines)
    while i < n:
        l = body_lines[i]
        # A `<!-- Part ... -->` comment that immediately precedes a moved geom/site
        # is carried along so we don't leave an orphaned title in the source head.
        if "<geom " in l:
            mesh = parse_attrs(l).get("mesh")
            if mesh in BILL_MESHES:
                if i > 0 and "Part " in body_lines[i - 1] and "<!--" in body_lines[i - 1]:
                    bill.append(body_lines[i - 1])
                bill.append(l)
            else:
                keep.append(l)
        elif "<site " in l and parse_attrs(l).get("name") in BILL_SITES:
            if i > 0 and "Part " in body_lines[i - 1] and "<!--" in body_lines[i - 1]:
                bill.append(body_lines[i - 1])
            bill.append(l)
        else:
            keep.append(l)
        i += 1

    found = {parse_attrs(g)["mesh"] for g in bill if "<geom " in g}
    missing = BILL_MESHES - found
    if missing:
        print("[add_mouth] ERROR: bill geoms not found:", missing)
        return 1

    shifted = "".join(shift_pos(l, pivot) for l in bill)
    mouth_body = (
        f'      <!-- Mouth injected by add_mouth.py: real duck servo ID 34 -->\n'
        f'      <body name="mouth" pos="{pivot_s}">\n'
        f'        <joint axis="0 1 0" name="mouth" type="hinge" class="chosen_actuator" '
        f'range="{MOUTH_RANGE[0]:g} {MOUTH_RANGE[1]:g}"/>\n'
        # Slender-rod inertia about the bill COM (light; ~6 cm long bill).
        f'        <inertial pos="{MOUTH_ARM_TIP[0]:g} {MOUTH_ARM_TIP[1]:g} '
        f'{MOUTH_ARM_TIP[2]:g}" mass="{MOUTH_MASS}" '
        f'fullinertia="3.6e-6 3.6e-6 1e-7 0 0 0"/>\n'
        + shifted
        + f'      </body>\n'
    )

    new_lines = lines[:jaw_open + 1] + keep + [mouth_body] + lines[jaw_close:]

    # ---- Add the mouth position actuator before the closing </actuator> ----
    out, inserted = [], False
    for l in new_lines:
        if not inserted and "</actuator>" in l:
            out.append(
                f'    <position class="chosen_actuator" name="mouth" joint="mouth" />\n'
            )
            inserted = True
        out.append(l)
    if not inserted:
        print("[add_mouth] ERROR: no </actuator> found")
        return 1

    Path(args.out).write_text("".join(out))
    print(f"[add_mouth] wrote {args.out}: bill {sorted(found)} + mouth_tip moved "
          f"under new 'mouth' hinge; actuator added.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

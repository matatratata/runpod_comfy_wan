from copy import deepcopy
from PIL import Image, ImageDraw
import numpy as np
import torch
import math

COCO18 = {
    "R_ARM": (2, 3, 4),
    "L_ARM": (5, 6, 7),
    "R_LEG": (8, 9, 10),
    "L_LEG": (11, 12, 13),
}
OP25 = {
    "R_ARM": (2, 3, 4),
    "L_ARM": (5, 6, 7),
    "R_LEG": (9, 10, 11),
    "L_LEG": (12, 13, 14),
}

EDGES_COCO18 = [
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
]
OPENPOSE_COLORS = [
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
]
HAND_BONES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


# ---------- utils ----------
def _to_triplets(seq):
    if not seq:
        return []
    if isinstance(seq[0], (list, tuple)) and len(seq[0]) >= 2:
        return [
            [float(a) for a in kp[:3]]
            if isinstance(kp, (list, tuple))
            else [0.0, 0.0, 0.0]
            for kp in seq
        ]
    out = []
    for i in range(0, len(seq), 3):
        x = float(seq[i]) if i < len(seq) else 0.0
        y = float(seq[i + 1]) if i + 1 < len(seq) else 0.0
        c = float(seq[i + 2]) if i + 2 < len(seq) else 0.0
        out.append([x, y, c])
    return out


def _flatten_triplets(tri):
    flat = []
    for x, y, c in tri:
        flat += [float(x), float(y), float(c)]
    return flat


def _scale_segment(a, b, s):
    ax, ay, ac = a
    bx, by, bc = b
    if ac <= 0 or bc <= 0:
        return (bx, by, bc)
    vx, vy = bx - ax, by - ay
    return (ax + vx * s, ay + vy * s, bc)


def _angle(vx, vy):
    return math.atan2(vy, vx)


def _rot(vx, vy, t):
    ct, st = math.cos(t), math.sin(t)
    return (ct * vx - st * vy, st * vx + ct * vy)


def _conf_sum(kps):
    return sum(k[2] for k in (kps or []))


# ---------- POSE_KEYPOINT handlers ----------
def _extract_frames(pk_obj):
    frames = []
    src_list = pk_obj if isinstance(pk_obj, list) else [pk_obj]
    for item in src_list:
        frame = {
            "src": item,
            "people_idx": None,
            "body_key": None,
            "body_is_flat": True,
            "body": [],
        }
        d = item
        people = (
            d.get("people")
            if isinstance(d.get("people"), list) and d["people"]
            else None
        )
        if people:
            p = people[0]
            if "pose_keypoints_2d" in p:
                frame.update(
                    people_idx=0,
                    body_key=("people", "pose_keypoints_2d"),
                    body_is_flat=True,
                    body=_to_triplets(p["pose_keypoints_2d"]),
                )
            elif "keypoints" in p:
                frame.update(
                    people_idx=0,
                    body_key=("people", "keypoints"),
                    body_is_flat=False,
                    body=_to_triplets(p["keypoints"]),
                )
        else:
            if "pose_keypoints_2d" in d:
                frame.update(
                    body_key=("root", "pose_keypoints_2d"),
                    body_is_flat=True,
                    body=_to_triplets(d["pose_keypoints_2d"]),
                )
            elif "keypoints" in d:
                frame.update(
                    body_key=("root", "keypoints"),
                    body_is_flat=False,
                    body=_to_triplets(d["keypoints"]),
                )
        frames.append(frame)
    return frames


def _write_back(frame, body):
    d = frame["src"]
    key = frame["body_key"]
    if not key:
        return
    if key[0] == "people":
        p = d["people"][frame["people_idx"]]
        if key[1] == "pose_keypoints_2d":
            p["pose_keypoints_2d"] = (
                _flatten_triplets(body) if frame["body_is_flat"] else body
            )
        else:
            p["keypoints"] = body if not frame["body_is_flat"] else _to_triplets(body)
    else:
        if key[1] == "pose_keypoints_2d":
            d["pose_keypoints_2d"] = (
                _flatten_triplets(body) if frame["body_is_flat"] else body
            )
        else:
            d["keypoints"] = body if not frame["body_is_flat"] else _to_triplets(body)


def _grab_hands_face(container):
    l = r = f = None
    for k in (
        "hand_left_keypoints_2d",
        "left_hand_keypoints_2d",
        "hands_left_keypoints_2d",
    ):
        if k in container:
            l = _to_triplets(container[k])
    for k in (
        "hand_right_keypoints_2d",
        "right_hand_keypoints_2d",
        "hands_right_keypoints_2d",
    ):
        if k in container:
            r = _to_triplets(container[k])
    for k in ("face_keypoints_2d", "face_kpts_2d", "face_keypoints"):
        if k in container:
            f = _to_triplets(container[k])
    return l, r, f


def _delete_keys(container, what):
    keys_map = {
        "hand_l": (
            "hand_left_keypoints_2d",
            "left_hand_keypoints_2d",
            "hands_left_keypoints_2d",
        ),
        "hand_r": (
            "hand_right_keypoints_2d",
            "right_hand_keypoints_2d",
            "hands_right_keypoints_2d",
        ),
        "face": ("face_keypoints_2d", "face_kpts_2d", "face_keypoints"),
        "body": ("pose_keypoints_2d", "keypoints"),
    }
    for k in keys_map[what]:
        if k in container:
            del container[k]


# ---------- preview ----------
def _edge_color_for_joint(j):
    for idx, (a, b) in enumerate(EDGES_COCO18):
        if j == a or j == b:
            return OPENPOSE_COLORS[min(idx, len(OPENPOSE_COLORS) - 1)]
    return (255, 255, 255)


def _draw_preview(
    body,
    draw_body,
    hands_l,
    hands_r,
    face,
    W,
    H,
    thickness,
    draw_points,
    point_radius,
    joint_point_color_mode,
    face_point_radius,
):
    img = Image.new("RGB", (W, H), (0, 0, 0))
    drw = ImageDraw.Draw(img)
    if draw_body and body:
        for idx, (i, j) in enumerate(EDGES_COCO18):
            if i < len(body) and j < len(body):
                xi, yi, ci = body[i]
                xj, yj, cj = body[j]
                if ci > 0 and cj > 0:
                    col = OPENPOSE_COLORS[min(idx, len(OPENPOSE_COLORS) - 1)]
                    drw.line([(xi, yi), (xj, yj)], fill=col, width=thickness)
        if draw_points:
            for j, (x, y, c) in enumerate(body):
                if c > 0:
                    col = (
                        _edge_color_for_joint(j)
                        if joint_point_color_mode == "limb"
                        else (255, 255, 255)
                    )
                    drw.ellipse(
                        [
                            x - point_radius,
                            y - point_radius,
                            x + point_radius,
                            y + point_radius,
                        ],
                        fill=col,
                    )

    def _hand(kps, col):
        if not kps or len(kps) < 21:
            return
        for a, b in HAND_BONES:
            xa, ya, ca = kps[a]
            xb, yb, cb = kps[b]
            if ca > 0 and cb > 0:
                drw.line([(xa, ya), (xb, yb)], fill=col, width=max(1, thickness - 1))
        if draw_points:
            for x, y, c in kps:
                if c > 0:
                    drw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=col)

    if hands_l:
        _hand(hands_l, (0, 200, 255))
    if hands_r:
        _hand(hands_r, (255, 128, 128))
    if face:
        r = max(1, int(face_point_radius))
        for x, y, c in face:
            if c > 0:
                drw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255))
    return img


class DWPoseScaleLimbsPKPassthru:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_keypoints": ("POSE_KEYPOINT",),
                "width": ("INT", {"default": 640, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 640, "min": 64, "max": 4096}),
                "upper_arm_scale": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.3, "max": 1.5, "step": 0.01},
                ),
                "lower_arm_scale": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.3, "max": 1.5, "step": 0.01},
                ),
                "upper_leg_scale": (
                    "FLOAT",
                    {"default": 0.85, "min": 0.3, "max": 1.5, "step": 0.01},
                ),
                "lower_leg_scale": (
                    "FLOAT",
                    {"default": 0.85, "min": 0.3, "max": 1.5, "step": 0.01},
                ),
                "format": (["COCO18", "OP25"],),
                # existing hand/face controls
                "hand_mode": (["auto", "on", "off"],),
                "face_mode": (["auto", "on", "off"],),
                "hand_attach": (["translate+rotate", "translate", "none"],),
                "min_conf_present": (
                    "FLOAT",
                    {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "delete_keys_when_off": (["yes", "no"],),
                # NEW: body on/off (like DWPose)
                "body_mode": (["auto", "on", "off"],),
                "delete_body_when_off": (["yes", "no"],),
                # preview cosmetics
                "thickness": ("INT", {"default": 4, "min": 1, "max": 30}),
                "draw_points": (["yes", "no"],),
                "point_radius": ("INT", {"default": 3, "min": 1, "max": 20}),
                "joint_point_color": (["white", "limb"],),
                "face_point_radius": ("INT", {"default": 3, "min": 1, "max": 20}),
                "render_preview": (["yes", "no"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT")
    RETURN_NAMES = ("pose_images", "pose_keypoints_out")
    FUNCTION = "run"
    CATEGORY = "Pose/Retarget"

    def run(
        self,
        pose_keypoints,
        width,
        height,
        upper_arm_scale,
        lower_arm_scale,
        upper_leg_scale,
        lower_leg_scale,
        format,
        hand_mode,
        face_mode,
        hand_attach,
        min_conf_present,
        delete_keys_when_off,
        body_mode,
        delete_body_when_off,
        thickness,
        draw_points,
        point_radius,
        joint_point_color,
        face_point_radius,
        render_preview,
    ):
        fmt = COCO18 if format == "COCO18" else OP25
        pk_clone = deepcopy(pose_keypoints)
        frames = _extract_frames(pk_clone)

        previews = []
        for f in frames:
            d = f["src"]
            body = f["body"]

            # keep originals for hand attach
            R_sh, R_el, R_wr = fmt["R_ARM"]
            L_sh, L_el, L_wr = fmt["L_ARM"]
            R_el_old = body[R_el] if len(body) > R_el else [0, 0, 0]
            R_wr_old = body[R_wr] if len(body) > R_wr else [0, 0, 0]
            L_el_old = body[L_el] if len(body) > L_el else [0, 0, 0]
            L_wr_old = body[L_wr] if len(body) > L_wr else [0, 0, 0]

            # scale limbs
            for side in ("R_ARM", "L_ARM"):
                s, e, w = fmt[side]
                if body and max(s, e, w) < len(body):
                    sx, sy, sc = body[s]
                    ex, ey, ec = body[e]
                    wx, wy, wc = body[w]
                    ex, ey, ec = _scale_segment(
                        (sx, sy, sc), (ex, ey, ec), upper_arm_scale
                    )
                    wx, wy, wc = _scale_segment(
                        (ex, ey, ec), (wx, wy, wc), lower_arm_scale
                    )
                    body[e] = [ex, ey, ec]
                    body[w] = [wx, wy, wc]
            for side in ("R_LEG", "L_LEG"):
                h, k, a = fmt[side]
                if body and max(h, k, a) < len(body):
                    hx, hy, hc = body[h]
                    kx, ky, kc = body[k]
                    ax, ay, ac = body[a]
                    kx, ky, kc = _scale_segment(
                        (hx, hy, hc), (kx, ky, kc), upper_leg_scale
                    )
                    ax, ay, ac = _scale_segment(
                        (kx, ky, kc), (ax, ay, ac), lower_leg_scale
                    )
                    body[k] = [kx, ky, kc]
                    body[a] = [ax, ay, ac]

            # new joints
            R_el_new = body[R_el] if len(body) > R_el else [0, 0, 0]
            R_wr_new = body[R_wr] if len(body) > R_wr else [0, 0, 0]
            L_el_new = body[L_el] if len(body) > L_el else [0, 0, 0]
            L_wr_new = body[L_wr] if len(body) > L_wr else [0, 0, 0]

            # write back body now (we may delete keys later for "off")
            _write_back(f, body)

            # hands/face from root or people[0]
            hands_l, hands_r, face = _grab_hands_face(d)
            if hands_l is None or hands_r is None or face is None:
                if isinstance(d.get("people"), list) and d["people"]:
                    p0 = d["people"][0]
                    l2, r2, f2 = _grab_hands_face(p0)
                    hands_l = hands_l or l2
                    hands_r = hands_r or r2
                    face = face or f2

            # resolve modes
            def resolve(mode, kps):
                if mode == "on":
                    return True
                if mode == "off":
                    return False
                return _conf_sum(kps) >= min_conf_present

            use_hand_l = resolve(hand_mode, hands_l)
            use_hand_r = resolve(hand_mode, hands_r)
            use_face = resolve(face_mode, face)
            use_body = resolve(body_mode, body)

            # when OFF, optionally delete keys to mirror DWPose drawers
            if delete_keys_when_off == "yes":
                if not use_hand_l:
                    _delete_keys(d, "hand_l")
                    if isinstance(d.get("people"), list) and d["people"]:
                        _delete_keys(d["people"][0], "hand_l")
                    hands_l = None
                if not use_hand_r:
                    _delete_keys(d, "hand_r")
                    if isinstance(d.get("people"), list) and d["people"]:
                        _delete_keys(d["people"][0], "hand_r")
                    hands_r = None
                if not use_face:
                    _delete_keys(d, "face")
                    if isinstance(d.get("people"), list) and d["people"]:
                        _delete_keys(d["people"][0], "face")
                    face = None
                if not use_body:
                    _delete_keys(d, "body")
                    if isinstance(d.get("people"), list) and d["people"]:
                        _delete_keys(d["people"][0], "body")

            else:
                if not use_hand_l:
                    hands_l = None
                if not use_hand_r:
                    hands_r = None
                if not use_face:
                    face = None
                # keep body keys present even if hidden in preview

            # attach hands to wrists (using moved wrists)
            def attach(hand, wr_old, el_old, wr_new, el_new):
                if not hand or wr_old[2] <= 0 or wr_new[2] <= 0:
                    return hand
                dx, dy = wr_new[0] - wr_old[0], wr_new[1] - wr_old[1]
                moved = [[x + dx, y + dy, c] for x, y, c in hand]  # translate
                if el_old[2] > 0 and el_new[2] > 0:
                    a1 = _angle(wr_old[0] - el_old[0], wr_old[1] - el_old[1])
                    a2 = _angle(wr_new[0] - el_new[0], wr_new[1] - el_new[1])
                    th = a2 - a1
                    out = []
                    for x, y, c in moved:
                        vx, vy = x - wr_new[0], y - wr_new[1]
                        rx, ry = _rot(vx, vy, th)
                        out.append([wr_new[0] + rx, wr_new[1] + ry, c])
                    return out
                return moved

            if use_hand_r and hands_r:
                hands_r = attach(hands_r, R_wr_old, R_el_old, R_wr_new, R_el_new)
                for k in (
                    "hand_right_keypoints_2d",
                    "right_hand_keypoints_2d",
                    "hands_right_keypoints_2d",
                ):
                    if k in d:
                        d[k] = _flatten_triplets(hands_r)
                if isinstance(d.get("people"), list) and d["people"]:
                    for k in (
                        "hand_right_keypoints_2d",
                        "right_hand_keypoints_2d",
                        "hands_right_keypoints_2d",
                    ):
                        if k in d["people"][0]:
                            d["people"][0][k] = _flatten_triplets(hands_r)

            if use_hand_l and hands_l:
                hands_l = attach(hands_l, L_wr_old, L_el_old, L_wr_new, L_el_new)
                for k in (
                    "hand_left_keypoints_2d",
                    "left_hand_keypoints_2d",
                    "hands_left_keypoints_2d",
                ):
                    if k in d:
                        d[k] = _flatten_triplets(hands_l)
                if isinstance(d.get("people"), list) and d["people"]:
                    for k in (
                        "hand_left_keypoints_2d",
                        "left_hand_keypoints_2d",
                        "hands_left_keypoints_2d",
                    ):
                        if k in d["people"][0]:
                            d["people"][0][k] = _flatten_triplets(hands_l)

            # preview
            if render_preview == "yes":
                img = _draw_preview(
                    body,
                    use_body,
                    hands_l if use_hand_l else None,
                    hands_r if use_hand_r else None,
                    face if use_face else None,
                    width,
                    height,
                    thickness,
                    draw_points == "yes",
                    point_radius,
                    joint_point_color,
                    face_point_radius,
                )
                arr = np.array(img, dtype=np.float32) / 255.0
                previews.append(torch.from_numpy(arr))

        if render_preview == "no":
            previews = [torch.zeros((1, 1, 3), dtype=torch.float32)]

        images = torch.stack(previews, dim=0)
        return (images, pk_clone)


NODE_CLASS_MAPPINGS = {"DWPoseScaleLimbsPKPassthru": DWPoseScaleLimbsPKPassthru}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DWPoseScaleLimbsPKPassthru": "Pose: Scale Limbs (Passthrough POSE_KEYPOINT)"
}

import cv2 
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Any, Dict, Optional


# =========================================================
# Stereo camera parameters (from your calibration)
# =========================================================
FOCAL_PX   = 707.0493   # focal length in pixels
BASELINE_M = 0.5373     # baseline in meters

# Central area used inside each bbox (5% of area, same aspect ratio)
AREA_FRACTION = 0.05

# Disparity at which we switch from far→near dominance (px)
D_SPLIT = 10.0
D_BAND  = 2.0   # width of smooth transition band (px)


# =========================================================
# CONFIG: sequences and paths
# =========================================================

sequences = [
    {
        "name": "seq_01",
        "left_base":  Path(
            r"C:\Users\NILMA\Downloads\Project_perception\34759_final_project_rect"
            r"\34759_final_project_rect\seq_01\image_02\data"
        ),
        "right_base": Path(
            r"C:\Users\NILMA\Downloads\Project_perception\34759_final_project_rect"
            r"\34759_final_project_rect\seq_01\image_03\data"
        ),
        "start": 0,
        "end": 144,
        "step": 1,           # set to 1 if you want every frame
        "fmt": "{:06d}.png",
    },
    {
        "name": "seq_02",
        "left_base":  Path(
            r"C:\Users\NILMA\Downloads\Project_perception\34759_final_project_rect"
            r"\34759_final_project_rect\seq_02\image_02\data"
        ),
        "right_base": Path(
            r"C:\Users\NILMA\Downloads\Project_perception\34759_final_project_rect"
            r"\34759_final_project_rect\seq_02\image_03\data"
        ),
        "start": 0,
        "end": 205,
        "step": 1,           # set to 1 if you want every frame
        "fmt": "{:010d}.png",
    },
    # You can add seq_03 here if you want:
    # {
    #     "name": "seq_03",
    #     "left_base":  Path(...),
    #     "right_base": Path(...),
    #     "start": 0,
    #     "end": 170,
    #     "step": 10,
    #     "fmt": "{:010d}.png",
    # },
]

# --- Two specialised SGBM configs for fusion ---
sgbm_configs = [
    {
        "name": "B_near_close",
        "min_disp": 0,
        "num_disp": 160,      # enough range for closer objects
        "block_size": 3,      # finer detail on contours
        "uniquenessRatio": 5,
        "speckleWindowSize": 60,
        "speckleRange": 1,
        "smooth_scale": 0.7,  # slightly lower smoothness → more detail
        "blur_ksize": 3,      # light smoothing
    },
    {
        "name": "B_far_distant",
        "min_disp": 0,
        "num_disp": 128,      # smaller range, focus on small disparities
        "block_size": 7,      # larger window → robust small disparity
        "uniquenessRatio": 3,
        "speckleWindowSize": 120,
        "speckleRange": 2,
        "smooth_scale": 2.0,  # stronger smoothness
        "blur_ksize": 5,      # a bit more smoothing
    },
]


# =========================================================
# Detection and ImageDetections classes
# =========================================================

@dataclass
class Detection:
    """
    Represents a single detected object.
    """
    bbox: Tuple[int, int, int, int]     # (x_min, y_min, x_max, y_max)
    center: Tuple[float, float, float]  # (x_center, y_center, z_center)
    classification: str                 # label or class name


@dataclass
class ImageDetections:
    """
    Represents an image and all its associated detections.
    """
    img: Any                            # NumPy array, path, etc.
    detections: List[Detection]
    seq_name: str                       # which sequence this image belongs to
    frame_idx: int                      # frame index inside that sequence


# =========================================================
# Helpers: SGBM + fusion
# =========================================================

def build_sgbm(cfg):
    w = cfg["block_size"]
    smooth_scale = cfg.get("smooth_scale", 1.0)
    P1 = int(8 * 3 * w * w * smooth_scale)
    P2 = int(32 * 3 * w * w * smooth_scale)

    return cv2.StereoSGBM_create(
        minDisparity=cfg["min_disp"],
        numDisparities=cfg["num_disp"],
        blockSize=w,
        P1=P1,
        P2=P2,
        disp12MaxDiff=1,
        uniquenessRatio=cfg["uniquenessRatio"],
        speckleWindowSize=cfg["speckleWindowSize"],
        speckleRange=cfg["speckleRange"],
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def fuse_near_far(disp_near, disp_far, d_split=D_SPLIT, band=D_BAND):
    """
    Fuse two disparity maps:
      - disp_near: better for large disparities (near objects)
      - disp_far:  better for small disparities (far objects)

    Uses a logistic blending around d_split.
    Returns: fused disparity map (float32)
    """
    disp_near = disp_near.astype(np.float32)
    disp_far  = disp_far.astype(np.float32)

    H, W = disp_near.shape
    fused = np.zeros((H, W), dtype=np.float32)

    valid_near = disp_near > 0
    valid_far  = disp_far  > 0

    # Case 1: only near valid → use near
    only_near = valid_near & ~valid_far
    fused[only_near] = disp_near[only_near]

    # Case 2: only far valid → use far
    only_far = ~valid_near & valid_far
    fused[only_far] = disp_far[only_far]

    # Case 3: both valid → smooth blend based on disparity (using near disparity)
    both = valid_near & valid_far
    if np.any(both):
        d = disp_near[both]
        # logistic transition from far→near around d_split
        w_near = 1.0 / (1.0 + np.exp(-(d - d_split) / band))
        w_far  = 1.0 - w_near
        fused[both] = w_near * disp_near[both] + w_far * disp_far[both]

    fused[fused < 0] = 0.0
    return fused


# =========================================================
# Class: precompute & query depth
# =========================================================

class StereoDepthDatabase:
    """
    - Precomputes fused disparity maps for all stereo pairs defined in `sequences`.
    - Stores them in memory indexed by (sequence_name, frame_idx).
    - Given a bounding box on a frame, returns the depth Z estimated from the
      median disparity in the central 5% of that bbox area.
    """

    def __init__(
        self,
        sequences_cfg,
        sgbm_cfgs,
        focal_px: float,
        baseline_m: float,
        area_fraction: float = AREA_FRACTION,
        d_split: float = D_SPLIT,
        d_band: float = D_BAND,
    ):
        self.sequences_cfg = sequences_cfg
        self.sgbm_cfgs = sgbm_cfgs
        self.focal_px = focal_px
        self.baseline_m = baseline_m
        self.area_fraction = area_fraction
        self.d_split = d_split
        self.d_band = d_band

        # prebuild SGBM matchers (one per config)
        self._matchers = {
            cfg["name"]: build_sgbm(cfg) for cfg in self.sgbm_cfgs
        }
        self._cfg_by_name = {cfg["name"]: cfg for cfg in self.sgbm_cfgs}

        # disparity maps stored as: (seq_name, frame_idx) -> np.ndarray (H,W)
        self.disparities: Dict[Tuple[str, int], np.ndarray] = {}

    # -----------------------------------------------------
    # Precomputation
    # -----------------------------------------------------
    def compute_all_disparities(self):
        """
        Loop over all configured sequences and frames and compute:
          - near disparity
          - far disparity
          - fused disparity (near/far blend)
        Store fused maps in self.disparities[(seq_name, frame_idx)].
        """
        for seq in self.sequences_cfg:
            seq_name   = seq["name"]
            left_base  = seq["left_base"]
            right_base = seq["right_base"]
            fmt        = seq["fmt"]

            for frame_idx in range(seq["start"], seq["end"] + 1, seq["step"]):
                fname = fmt.format(frame_idx)

                left_img_path  = left_base  / fname
                right_img_path = right_base / fname

                imgL = cv2.imread(str(left_img_path),  cv2.IMREAD_GRAYSCALE)
                imgR = cv2.imread(str(right_img_path), cv2.IMREAD_GRAYSCALE)

                if imgL is None or imgR is None:
                    # Image missing → skip
                    continue

                # Contrast enhancement
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                imgL_eq = clahe.apply(imgL)
                imgR_eq = clahe.apply(imgR)

                disp_maps = {}
                for cfg_name, matcher in self._matchers.items():
                    cfg = self._cfg_by_name[cfg_name]
                    raw_disp = matcher.compute(imgL_eq, imgR_eq).astype(np.float32) / 16.0

                    ksize = cfg.get("blur_ksize", 5)
                    if ksize is not None and ksize > 1:
                        raw_disp = cv2.GaussianBlur(raw_disp, (ksize, ksize), 0)

                    disp_maps[cfg_name] = raw_disp

                # near & far
                disp_near = disp_maps["B_near_close"]
                disp_far  = disp_maps["B_far_distant"]

                # fused map
                disp_fused = fuse_near_far(disp_near, disp_far,
                                           d_split=self.d_split, band=self.d_band)

                self.disparities[(seq_name, frame_idx)] = disp_fused

    # -----------------------------------------------------
    # Internal helper: central 5% crop of bbox
    # -----------------------------------------------------
    def _compute_crop_bbox(self, bbox, disp_shape):
        """
        Given a full bbox (x1,y1,x2,y2) and disparity shape (H,W),
        return the central crop bbox whose area is `area_fraction`
        of the original bbox, preserving aspect ratio.
        """
        H, W = disp_shape[:2]
        x1, y1, x2, y2 = bbox

        w_box = x2 - x1
        h_box = y2 - y1
        if w_box <= 1 or h_box <= 1:
            return None  # too small

        scale = np.sqrt(self.area_fraction)  # shrink area but keep aspect ratio

        # center of original bbox
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        half_w = 0.5 * w_box * scale
        half_h = 0.5 * h_box * scale

        x1_c = int(round(cx - half_w))
        x2_c = int(round(cx + half_w))
        y1_c = int(round(cy - half_h))
        y2_c = int(round(cy + half_h))

        # clip to image
        x1_c = int(np.clip(x1_c, 0, W - 1))
        x2_c = int(np.clip(x2_c, 0, W))
        y1_c = int(np.clip(y1_c, 0, H - 1))
        y2_c = int(np.clip(y2_c, 0, H))

        if x2_c <= x1_c or y2_c <= y1_c:
            return None

        return (x1_c, y1_c, x2_c, y2_c)

    # -----------------------------------------------------
    # Public: query depth for a given bbox
    # -----------------------------------------------------
    def get_depth_for_bbox(
        self,
        seq_name: str,
        frame_idx: int,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[dict]:
        """
        Given:
          - sequence name (e.g. "seq_01"),
          - frame index (int),
          - bbox = (x1, y1, x2, y2) in pixel coordinates,

        returns a dict with:
          {
            "bbox": (x1, y1, x2, y2),
            "crop_bbox": (cx1, cy1, cx2, cy2),
            "center_xy": (cx, cy),
            "median_disp": float or NaN,
            "Z_est": float or NaN,
            "frac_invalid": float
          }

        or None if no disparity is available or bbox is invalid.
        """
        key = (seq_name, frame_idx)
        if key not in self.disparities:
            # No precomputed disparity for this frame
            print(f"[DEBUG] No disparity for {key}")
            return None


        disp = self.disparities[key]
        x1, y1, x2, y2 = bbox

        crop_bbox = self._compute_crop_bbox(bbox, disp.shape)
        if crop_bbox is None:
            return None

        cx1, cy1, cx2, cy2 = crop_bbox
        patch = disp[cy1:cy2, cx1:cx2]
        if patch.size == 0:
            return None

        valid = patch > 0
        frac_invalid = float((~valid).sum()) / float(patch.size)

        if valid.sum() == 0:
            median_disp = float("nan")
            Z_est = float("nan")
        else:
            vals = patch[valid]
            median_disp = float(np.median(vals))
            if median_disp <= 0 or np.isnan(median_disp):
                Z_est = float("nan")
            else:
                Z_est = (self.focal_px * self.baseline_m) / median_disp

        # center of full bbox (in image coords)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        return {
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "crop_bbox": crop_bbox,
            "center_xy": (float(cx), float(cy)),
            "median_disp": float(median_disp),
            "Z_est": float(Z_est),
            "frac_invalid": float(frac_invalid),
        }


# =========================================================
# MAIN: compute depth for ALL label bboxes and save to txt
# =========================================================

if __name__ == "__main__":

    # Helper to load labels.txt for a sequence
    def load_labels(label_path: Path):
        labels = []
        with label_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                # Expecting at least: frame, track_id, type, truncation, occlusion,
                # alpha, bbox_left, bbox_top, bbox_right, bbox_bottom, h, w, l, X, Y, Z, rotation_y
                if len(parts) < 17:
                    continue
                labels.append({
                    "frame":      int(parts[0]),
                    "track_id":   int(parts[1]),
                    "type":       parts[2],
                    "truncation": float(parts[3]),
                    "occlusion":  int(parts[4]),
                    "alpha":      float(parts[5]),
                    "bbox_left":  float(parts[6]),
                    "bbox_top":   float(parts[7]),
                    "bbox_right": float(parts[8]),
                    "bbox_bottom":float(parts[9]),
                    "h":          float(parts[10]),
                    "w":          float(parts[11]),
                    "l":          float(parts[12]),
                    "X":          float(parts[13]),
                    "Y":          float(parts[14]),
                    "Z":          float(parts[15]),
                    "rotation_y": float(parts[16]),
                })
        return labels

    # 1) Build the database and compute disparities
    db = StereoDepthDatabase(
        sequences_cfg=sequences,
        sgbm_cfgs=sgbm_configs,
        focal_px=FOCAL_PX,
        baseline_m=BASELINE_M,
        area_fraction=AREA_FRACTION,  # here it's 1.0 (full bbox)
        d_split=D_SPLIT,
        d_band=D_BAND,
    )

    print("Computing fused disparity maps for all sequences/frames...")
    db.compute_all_disparities()
    print("Done computing disparities.")

    # 2) For each sequence, load labels and compute depth for EVERY bbox
    out_path = Path("all_label_depths.txt")
    with out_path.open("w") as f_out:
        for seq in sequences:
            seq_name = seq["name"]
            left_base = seq["left_base"]
            fmt = seq["fmt"]

            # labels.txt is in the seq_xx folder (parent of image_02 / image_03)
            seq_dir = left_base.parents[1]
            labels_path = seq_dir / "labels.txt"
            if not labels_path.is_file():
                print(f"[WARN] No labels.txt found for {seq_name} at {labels_path}")
                continue

            labels = load_labels(labels_path)

            # Organize labels per frame for efficiency
            labels_by_frame: Dict[int, List[dict]] = {}
            for o in labels:
                frame = o["frame"]
                labels_by_frame.setdefault(frame, []).append(o)

            f_out.write(f"############################################################\n")
            f_out.write(f"Sequence {seq_name}\n")
            f_out.write(f"############################################################\n\n")

            # Iterate over all frames of this sequence (as per config)
            for frame_idx in range(seq["start"], seq["end"] + 1, seq["step"]):
                if frame_idx not in labels_by_frame:
                    continue  # no labels for this frame

                fname = fmt.format(frame_idx)
                frame_labels = labels_by_frame[frame_idx]

                f_out.write(f"=== {seq_name} - Frame {fname} (frame_idx={frame_idx}) ===\n")

                for o in frame_labels:
                    x1 = int(round(o["bbox_left"]))
                    y1 = int(round(o["bbox_top"]))
                    x2 = int(round(o["bbox_right"]))
                    y2 = int(round(o["bbox_bottom"]))
                    bbox = (x1, y1, x2, y2)

                    Z_label = o["Z"]  # depth from the label

                    res = db.get_depth_for_bbox(
                        seq_name=seq_name,
                        frame_idx=frame_idx,
                        bbox=bbox,
                    )

                    if res is None:
                        f_out.write(
                            f"  id={o['track_id']:3d}, bbox={bbox}, "
                            f"Z_label={Z_label:.3f}, Z_disp=nan\n"
                        )
                    else:
                        Z_disp = res["Z_est"]
                        f_out.write(
                            f"  id={o['track_id']:3d}, bbox={bbox}, "
                            f"Z_label={Z_label:.3f}, Z_disp={Z_disp:.3f}\n"
                        )


                f_out.write("\n")

    print(f"Done. Depth values for all label bboxes saved to: {out_path}")

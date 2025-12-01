import os
import time
import cv2
import numpy as np
import math
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Any
from pathlib import Path

from kalman_filters import KalmanFilter3D_CA, KalmanFilter3D_CV
import disparity


# =============================
# SINGLE TRACKING LABEL CLASS
# =============================
class TrackingLabel:
    def __init__(self, frame, track_id, obj_type, truncated, occluded, alpha,
                 bbox, dimensions, location, rotation_y, score=None):
        self.frame = int(frame)
        self.track_id = int(track_id)
        self.obj_type = obj_type           # 'Car', 'Pedestrian', 'Cyclist', etc.
        self.truncated = float(truncated)  # 0..1
        self.occluded = int(occluded)      # 0,1,2,3
        self.alpha = float(alpha)          # [-pi, pi]

        self.bbox = np.array(bbox, dtype=float)              # [left, top, right, bottom]
        self.dimensions = np.array(dimensions, dtype=float)  # [h, w, l]
        self.location = np.array(location, dtype=float)      # [x, y, z]
        self.rotation_y = float(rotation_y)                  # [-pi, pi]
        self.score = float(score) if score is not None else None

    def __repr__(self):
        return f"TrackingLabel(frame={self.frame}, track_id={self.track_id}, type={self.obj_type})"


# =============================
# LABEL FILE LOADER CLASS
# =============================
class TrackingLabelLoader:
    def __init__(self, filename):
        self.filename = filename
        self.labels = []

    def load(self):
        with open(self.filename, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 17:  # ignore incomplete lines
                    continue

                frame      = parts[0]
                track_id   = parts[1]
                obj_type   = parts[2]
                truncated  = parts[3]
                occluded   = parts[4]
                alpha      = parts[5]

                bbox       = [parts[6],  parts[7],  parts[8],  parts[9]]
                dimensions = [parts[10], parts[11], parts[12]]
                location   = [parts[13], parts[14], parts[15]]
                rotation_y = parts[16]
                score      = parts[17] if len(parts) > 17 else None

                label = TrackingLabel(frame, track_id, obj_type, truncated, occluded, alpha,
                                      bbox, dimensions, location, rotation_y, score)
                self.labels.append(label)
        return self.labels

    def get_labels_for_frame(self, frame_id):
        """Return all labels for a specific frame."""
        return [label for label in self.labels if label.frame == frame_id]

    def get_labels_for_track(self, track_id):
        """Return all labels for a specific object track."""
        return [label for label in self.labels if label.track_id == track_id]


# =============================
# TRACKED OBJECT CLASS
# =============================
class TrackedObject:
    def __init__(self, track_id, type, color):
        self.track_id = track_id
        self.kf = KalmanFilter3D_CV() if type == "Pedestrian" else KalmanFilter3D_CA()
        self.color = color
        self.last_update = time.time()
        self.initialized = False  # have we ever had a measurement?

    def predict(self, occluded=False):
        """Predict step with occlusion flag."""
        now = time.time()
        dt = now - self.last_update
        if dt <= 0:
            dt = 1e-3
        self.last_update = now
        # kalman_filters.KalmanFilter3D_* predict(dt, occluded=False)
        self.kf.predict(dt, occluded=occluded)

    def update(self, X, Y, Z):
        """Update with a 3D measurement."""
        Z_meas = np.array([[X], [Y], [Z]])
        self.kf.update(Z_meas)
        self.initialized = True

    def get_predicted_position(self):
        return self.kf.x[0, 0], self.kf.x[1, 0], self.kf.x[2, 0]


# =============================
# MULTI-OBJECT TRACKER CLASS
# =============================
class Tracker:
    def __init__(self, colors=None):
        self.tracked_id = None
        # Track ALL types → leave tracked_type = None
        self.tracked_type = None
        self.objects = {}
        self.colors = colors if colors else [(255, 0, 0)]
        self.highest_track_id = 0

    def get_or_create(self, tid, obj_type):
        """Return existing TrackedObject or create a new one."""
        if tid not in self.objects:
            color = self.colors[len(self.objects) % len(self.colors)]
            self.objects[tid] = TrackedObject(tid, obj_type, color)
            if tid > self.highest_track_id:
                self.highest_track_id = tid
        return self.objects[tid]

    def is_tracked(self, tid, type):
        # With tracked_type=None and tracked_id=None, we track everything
        if self.tracked_id is not None and tid == self.tracked_id:
            return True
        if self.tracked_id is None and self.tracked_type is not None and type == self.tracked_type:
            return True
        if self.tracked_id is None and self.tracked_type is None:
            return True
        return False

    def remove_object(self, track_id):
        if track_id in self.objects:
            del self.objects[track_id]


# =============================
# CAMERA
# =============================
class Camera:
    def __init__(self, K_rect):
        self.K = K_rect
        self.fx = K_rect[0, 0]
        self.fy = K_rect[1, 1]
        self.cx = K_rect[0, 2]
        self.cy = K_rect[1, 2]

    def project_point(self, P):
        X, Y, Z = P
        if Z <= 0:
            return None
        u = self.K[0, 0] * X / Z + self.K[0, 2]
        v = self.K[1, 1] * Y / Z + self.K[1, 2]
        return int(u), int(v)

    def backproject_pixel(self, u, v, Z):
        # Z must be in the same scale as K (like meters)
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy
        return np.array([X, Y, Z])


# =============================
# OCCLUSION RECTANGLE
# =============================
def is_in_rectangle(uv, rect):
    """Check if a 2D point uv=(u,v) is inside rectangle rect=(x1,y1,x2,y2)."""
    if uv is None:
        return False
    u, v = uv
    x1, y1, x2, y2 = rect
    return x1 <= u <= x2 and y1 <= v <= y2


def associate_detections_to_tracks(detections, tracks, max_distance=1):
    """
    detections: list of (x,y,z)
    tracks: list of predicted (x,y,z) from the Kalman filters

    returns:
        matches: list of (track_idx, detection_idx)
        unmatched_tracks: list of track indices
        unmatched_detections: list of detection indices
    """
    if len(tracks) == 0:
        return [], [], list(range(len(detections)))

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for t, track_pos in enumerate(tracks):
        for d, det_pos in enumerate(detections):
            dist = np.linalg.norm(track_pos - det_pos)
            cost_matrix[t, d] = dist

    track_idx, det_idx = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_tracks = list(range(len(tracks)))
    unmatched_detections = list(range(len(detections)))

    for t, d in zip(track_idx, det_idx):
        if cost_matrix[t, d] > max_distance:
            continue
        matches.append((t, d))
        unmatched_tracks.remove(t)
        unmatched_detections.remove(d)

    return matches, unmatched_tracks, unmatched_detections


# =============================
# MAIN FUNCTION
# =============================
def main():
    # ---- CONFIG FOR THIS RUN ----
    seq_name = "seq_02"

    image_02_folder = Path(
        r"C:\Users\NILMA\Downloads\Project_perception\34759_final_project_rect"
        r"\34759_final_project_rect\seq_02\image_02\data"
    )
    image_03_folder = Path(
        r"C:\Users\NILMA\Downloads\Project_perception\34759_final_project_rect"
        r"\34759_final_project_rect\seq_02\image_03\data"
    )
    label_file = Path(
        r"C:\Users\NILMA\Downloads\Project_perception\34759_final_project_rect"
        r"\34759_final_project_rect\seq_02\labels.txt"
    )

    # Occlusion rectangle (in left image coordinates)
    occlusion_rect = (660, 100, 910, 350)
    bounding_box = (0, 0, 1224, 370)

    font      = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
    fontColor = (255, 255, 255)
    thickness = 3
    lineType  = 2

    # Camera
    camera = Camera(np.array([
        [7.070493e+02, 0.000000e+00, 6.040814e+02, 0.000000e+00],
        [0.000000e+00, 7.070493e+02, 1.805066e+02, 0.000000e+00],
        [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]
    ]))

    # Load labels
    loader = TrackingLabelLoader(label_file)
    labels = loader.load()

    tracker = Tracker()
    frame_ids = sorted(np.unique([label.frame for label in labels]))

    # ---- BUILD DEPTH DATABASE FROM disparity.py ----
    db = disparity.StereoDepthDatabase(
        sequences_cfg=disparity.sequences,
        sgbm_cfgs=disparity.sgbm_configs,
        focal_px=disparity.FOCAL_PX,
        baseline_m=disparity.BASELINE_M,
        area_fraction=disparity.AREA_FRACTION,
        d_split=disparity.D_SPLIT,
        d_band=disparity.D_BAND,
    )

    print("Computing fused disparity maps...")
    db.compute_all_disparities()
    print("Done computing disparities.")

    # ---- VIDEO WRITER (NEW) ----
    ### NEW: create video writer from first frame
    first_frame_idx = frame_ids[0]
    first_img_path = os.path.join(image_02_folder, f"{int(first_frame_idx):010d}.png")
    first_frame = cv2.imread(first_img_path)
    if first_frame is None:
        raise FileNotFoundError(f"Could not read first frame: {first_img_path}")
    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 10.0
    video_out_path = "seq02_labels_depth_kf.mp4"
    video_out = cv2.VideoWriter(video_out_path, fourcc, fps, (w, h))
    print(f"Writing video to: {video_out_path}")
    # -----------------------------

    # ---- OPEN LOG FILE FOR EVALUATION ----
    log_path = "tracking_depth_with_occlusion_kalman_seq02.txt"
    with open(log_path, "w") as log_f:
        log_f.write(
            "seq frame track_id type bbox "
            "X_true Y_true Z_true "
            "X_meas Y_meas Z_meas "
            "X_est Y_est Z_est "
            "occluded\n"
        )

        # ---- MAIN LOOP OVER FRAMES ----
        for frame_idx in frame_ids:
            if (seq_name, frame_idx) not in db.disparities:
                print(f"[WARN] no disparity precomputed for {seq_name} frame {frame_idx}")
                continue

            frame_labels = loader.get_labels_for_frame(frame_idx)

            # Left image
            img_02_file = os.path.join(image_02_folder, f"{int(frame_idx):010d}.png")
            frame_02 = cv2.imread(img_02_file)
            if frame_02 is None:
                print("Missing frame:", img_02_file)
                continue

            # Right image
            img_03_file = os.path.join(image_03_folder, f"{int(frame_idx):010d}.png")
            frame_03 = cv2.imread(img_03_file)
            if frame_03 is None:
                print("Missing frame:", img_03_file)
                continue

            # Draw occlusion rectangle
            cv2.rectangle(frame_02,
                          (occlusion_rect[0], occlusion_rect[1]),
                          (occlusion_rect[2], occlusion_rect[3]),
                          (0, 255, 255), 2)

            H, W = frame_02.shape[:2]

            for label in frame_labels:
                # Track ALL object types, no filter here

                # Get / create Kalman track
                obj = tracker.get_or_create(label.track_id, label.obj_type)

                # bbox center in image coordinates
                x1_f, y1_f, x2_f, y2_f = label.bbox
                center_u = 0.5 * (x1_f + x2_f)
                center_v = 0.5 * (y1_f + y2_f)
                inside_occ = is_in_rectangle((center_u, center_v), occlusion_rect)

                # ----- STEP 1: measurement from disparity (if visible) -----
                meas_xyz = (float("nan"), float("nan"), float("nan"))
                have_measurement = False

                if not inside_occ:
                    res = db.get_depth_for_bbox(
                        seq_name=seq_name,
                        frame_idx=frame_idx,
                        bbox=label.bbox,
                    )
                    if res is not None and not math.isnan(res["Z_est"]):
                        u = res["center_xy"][0]
                        v = res["center_xy"][1]
                        Z_est_depth = res["Z_est"]
                        P3D = camera.backproject_pixel(u, v, Z_est_depth)
                        meas_xyz = (P3D[0], P3D[1], P3D[2])
                        have_measurement = True
                    else:
                        # no valid depth
                        pass
                # else: inside_occ -> predict only, no disparity

                # ----- STEP 2: Kalman predict + update -----
                if have_measurement:
                    obj.predict(occluded=False)
                    obj.update(*meas_xyz)
                else:
                    obj.predict(occluded=True)

                X_est, Y_est, Z_est = obj.get_predicted_position()

                # ----- STEP 3: logging & drawing -----
                X_true, Y_true, Z_true = label.location

                bbox_int = tuple(label.bbox.astype(int))
                log_f.write(
                    f"{seq_name} {frame_idx:3d} {label.track_id:3d} {label.obj_type:10s} "
                    f"{bbox_int} "
                    f"{X_true:7.3f} {Y_true:7.3f} {Z_true:7.3f} "
                    f"{meas_xyz[0]:7.3f} {meas_xyz[1]:7.3f} {meas_xyz[2]:7.3f} "
                    f"{X_est:7.3f} {Y_est:7.3f} {Z_est:7.3f} "
                    f"{int(inside_occ)}\n"
                )

                # Console debug
                print(f"[{seq_name}] frame={frame_idx:3d} id={label.track_id:3d} type={label.obj_type}")
                print(f"  bbox = {label.bbox}")
                print(f"  True XYZ       = ({X_true:.2f}, {Y_true:.2f}, {Z_true:.2f})")
                print(f"  Measured XYZ   = ({meas_xyz[0]:.2f}, {meas_xyz[1]:.2f}, {meas_xyz[2]:.2f})")
                print(f"  Kalman XYZ     = ({X_est:.2f}, {Y_est:.2f}, {Z_est:.2f})")
                print(f"  inside_occ     = {inside_occ}\n")

                # Real (ground-truth) circle
                real_uv = camera.project_point(label.location)
                if real_uv is not None and tracker.is_tracked(label.track_id, label.obj_type):
                    color = (0, 255, 0) if not inside_occ else (0, 128, 0)
                    cv2.circle(frame_02, real_uv, 4, color, -1)
                    cv2.putText(frame_02, str(label.track_id), real_uv,
                                font, fontScale, fontColor, thickness, lineType)

                # Estimated 2D position from Kalman
                est_uv = camera.project_point((X_est, Y_est, Z_est))
                if est_uv is not None and tracker.is_tracked(label.track_id, label.obj_type):
                    color = (0, 0, 255) if not inside_occ else (0, 0, 128)
                    cv2.circle(frame_02, est_uv, 4, color, -1)

                # ----- NEW: draw bbox + x,y,z text at bottom of bbox -----
                x1, y1, x2, y2 = bbox_int
                if tracker.is_tracked(label.track_id, label.obj_type):
                    # bbox color: green if visible, yellow if occluded
                    box_color = (0, 255, 0) if not inside_occ else (0, 255, 255)
                    cv2.rectangle(frame_02, (x1, y1), (x2, y2), box_color, 2)

                    # choose values:
                    if inside_occ or not have_measurement:
                        # occluded or no depth → use Kalman estimate
                        x_val, y_val, z_val = X_est, Y_est, Z_est
                    else:
                        # visible & has depth → x,y from labels, z from depth map
                        x_val, y_val = X_true, Y_true
                        z_val = Z_est_depth

                    text_xyz = f"x={x_val:.2f}, y={y_val:.2f}, z={z_val:.2f}"
                    text_org = (x1, min(H - 5, y2 - 5))
                    cv2.putText(frame_02, text_xyz, text_org,
                                font, 0.5, box_color, 1, lineType)
                # -----------------------------------------

            cv2.putText(frame_02, str(frame_idx), (1000, 20),
                        font, fontScale, fontColor, thickness, lineType)

            # write frame to video
            video_out.write(frame_02)

            # (optional) show window
            cv2.imshow("Tracking with occlusion-aware Kalman", frame_02)
            if cv2.waitKey(1) == 27:
                break
            time.sleep(0.02)

    video_out.release()
    cv2.destroyAllWindows()
    print(f"\nDepth + Kalman log (with occlusion) saved to: {log_path}")
    print(f"Video saved to: {video_out_path}")


if __name__ == "__main__":
    main()

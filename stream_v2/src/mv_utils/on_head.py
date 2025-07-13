
import cv2
import numpy as np

def face_padd(image, bbox, keypoints, padding=0.5, square=False):
    h_img, w_img = image.shape[:2]
    x1, y1, x2, y2 = bbox

    # Width and height of the original box
    w = x2 - x1
    h = y2 - y1

    # === Make square if requested ===
    if square:
        side = max(w, h)
        cx = x1 + w // 2
        cy = y1 + h // 2
        x1 = cx - side // 2
        y1 = cy - side // 2
        x2 = x1 + side
        y2 = y1 + side

    # Recalculate width/height
    w = x2 - x1
    h = y2 - y1

    # Apply padding
    pad_w = int(w * padding)
    pad_h = int(h * padding)

    new_x1 = x1 - pad_w
    new_y1 = y1 - pad_h
    new_x2 = x2 + pad_w
    new_y2 = y2 + pad_h

    # Compute padding needed
    left_pad = max(0, -new_x1)
    top_pad = max(0, -new_y1)
    right_pad = max(0, new_x2 - w_img)
    bottom_pad = max(0, new_y2 - h_img)

    crop_w = new_x2 - new_x1
    crop_h = new_y2 - new_y1

    channels = image.shape[2] if image.ndim == 3 else 1
    face_crop = np.zeros((crop_h, crop_w, channels), dtype=image.dtype)

    src_x1 = max(0, new_x1)
    src_y1 = max(0, new_y1)
    src_x2 = min(w_img, new_x2)
    src_y2 = min(h_img, new_y2)

    dst_x1 = src_x1 - new_x1
    dst_y1 = src_y1 - new_y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    face_crop[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]

    # Adjust bbox and keypoints
    adjusted_bbox = [x1 - new_x1, y1 - new_y1, x2 - new_x1, y2 - new_y1]
    adjusted_keypoints = [{"x": kp["x"] - new_x1, "y": kp["y"] - new_y1} for kp in keypoints]

    return face_crop, adjusted_bbox, adjusted_keypoints


    


def face_warp(image: np.ndarray, keypoints: list, output_size=(112, 112), scale_factor=1.0) -> np.ndarray:
    """
    Align a face using only 5 facial keypoints.

    Parameters:
        image (np.ndarray): Input image.
        keypoints (list): List of 5 dicts [{'x':..., 'y':...}, ...], in order:
                          [left_eye, right_eye, nose, left_mouth, right_mouth].
        output_size (tuple): Desired output image size (width, height). Default is (112, 112).
        scale_factor (float): Zoom factor for alignment template (1.0 = default size, >1 = zoom in).

    Returns:
        np.ndarray: Aligned face image of size `output_size`.
    """
    if len(keypoints) != 5:
        raise ValueError("Expected 5 keypoints: [left_eye, right_eye, nose, left_mouth, right_mouth]")

    # Convert keypoints to np.float32 array
    src_kps = np.array([[kp['x'], kp['y']] for kp in keypoints], dtype=np.float32)

    # Canonical 5-point template for 96x112 reference
    default_crop_size = (96, 112)
    reference_points = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]
    ], dtype=np.float32)

    # Scale reference points to match output size
    scale = np.array(output_size) / np.array(default_crop_size, dtype=np.float32)
    dst_kps = reference_points * scale

    # Apply zoom (scale_factor) centered at face center
    center = np.mean(dst_kps, axis=0)
    dst_kps = (dst_kps - center) * scale_factor + center

    # Estimate affine transformation
    M, _ = cv2.estimateAffinePartial2D(src_kps, dst_kps, method=cv2.LMEDS)
    if M is None:
        raise RuntimeError("Failed to estimate affine transformation.")

    # Warp the image
    aligned = cv2.warpAffine(image, M, output_size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return aligned

def plot_landmark(img, bbox, keypoints):
    img = img.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw keypoints
    for kp in keypoints:
        cv2.circle(img, (kp["x"], kp["y"]), 3, (0, 0, 255), -1)
    return img



def face_box_extract(keypoints, bbox, score_threshold=0.5, padding=0.2):
    """
    Crop the head (face + ears + top of shoulders) from pose keypoints.

    Returns:
        (x_min, y_top, x_max, y_bottom, avg_score) if face detected, else None
    """

    KP = {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
        "left_shoulder": 5,
        "right_shoulder": 6
    }

    # Step 1: Check if face is present
    nose_score = keypoints[KP["nose"]]["score"]
    leye_score = keypoints[KP["left_eye"]]["score"]
    reye_score = keypoints[KP["right_eye"]]["score"]

    if not (nose_score > score_threshold and leye_score > score_threshold and reye_score > score_threshold):
        return None, None

    avg_score = (nose_score + leye_score + reye_score) / 3

    # Step 2: Person bounding box
    x1_box, y1_box, x2_box, y2_box = bbox

    # Step 3: Horizontal bounds
    x_left = (
        keypoints[KP["left_ear"]]["x"]
        if keypoints[KP["left_ear"]]["score"] > score_threshold
        else keypoints[KP["left_eye"]]["x"]
    )
    x_right = (
        keypoints[KP["right_ear"]]["x"]
        if keypoints[KP["right_ear"]]["score"] > score_threshold
        else keypoints[KP["right_eye"]]["x"]
    )

    x_min = min(x_left, x_right)
    x_max = max(x_left, x_right)
    x_min = int(x_min - padding * (x_max - x_min))
    x_max = int(x_max + padding * (x_max - x_min))

    # Step 4: Vertical bounds
    y_top = y1_box
    shoulders_y = [keypoints[KP["left_shoulder"]]["y"], keypoints[KP["right_shoulder"]]["y"]]
    y_bottom = int(max(shoulders_y))

    # Clip to person bounding box
    x_min = max(x_min, x1_box)
    x_max = min(x_max, x2_box)
    y_top = max(y_top, 0)
    y_bottom = min(y_bottom, y2_box)

    return (x_min, y_top, x_max, y_bottom), avg_score


def summarize_keypoints(keypoints, min_conf=0.0):
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    summary = ', '.join(
        f"{name}={kp['score']:.2f}"
        for name, kp in zip(KEYPOINT_NAMES, keypoints)
        if kp["score"] >= min_conf
    )

    return f"keypoints: {summary}"

import cv2
import numpy as np
from preprocess.dfa_inference import dfa_triton, dfa_warp
from typing import Union

def pad_bounding_box(x1: int, y1: int, x2: int, y2: int, image_shape: Union[np.ndarray, tuple[int, int]], padding=0.1):

    """
    Pads a bounding box while ensuring it remains within the image boundaries.

    Args:
        x1, y1, x2, y2 (int): Original bounding box coordinates.
        image_shape (tuple): Shape of the image (height, width, channels).
        padding (int): Amount of padding to apply.

    Returns:
        tuple: Padded bounding box (x1_new, y1_new, x2_new, y2_new).
    """
    face_width = x2 - x1
    face_height = y2 - y1
    padding_x = int(padding * face_width)
    padding_y = int(padding * face_height)
    height, width = image_shape[:2]  # Extract height and width
    height = int(height)
    width = int(width)
    # Apply padding
    x1_new = max(0, x1 - padding_x)
    y1_new = max(0, y1 - padding_y)
    x2_new = min(width, x2 + padding_x)
    y2_new = min(height, y2 + padding_y)
    return x1_new, y1_new, x2_new, y2_new


def pad_bounding_box_symmetrically(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    image_shape: Union[tuple[int, int], np.ndarray],
    padding: float = 0.1
):
    """
    Expands the bounding box symmetrically around its center by a 'padding' fraction,
    ensuring the final box stays completely within the image.

    The 'padding' value is a fraction of the face size in each dimension.
    E.g. padding=0.1 => 10% extra width on each side & 10% extra height on each side.

    Args:
        x1, y1, x2, y2 (int): Original bounding box coordinates.
        image_shape (tuple or ndarray): Shape of the image [height, width, (channels)].
        padding (float): Fraction of the bounding box size to expand on each side.

    Returns:
        (x1_new, y1_new, x2_new, y2_new): A new bounding box, guaranteed to fit within the image.
    """
    # Ensure padding is non-negative
    padding = max(padding, 0.0)

    # Image dimensions
    height, width = image_shape[:2]

    # Current bounding-box width/height
    face_width = x2 - x1
    face_height = y2 - y1

    # Center of the original bounding box
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    # Desired half-width/half-height with padding.
    # E.g., if face_width=100 and padding=0.1,
    #   we want half_w = 0.5*100 + 0.1*100 = 60.
    half_w_desired = (0.5 + padding) * face_width
    half_h_desired = (0.5 + padding) * face_height

    # Now clamp these half-dimensions so the box does not go out of the image.
    # We need:
    #   center_x - half_w >= 0   ==>   half_w <= center_x
    #   center_x + half_w <= width   ==>   half_w <= (width - center_x)
    # And similarly for half_h wrt center_y and height.
    half_w_max = min(center_x, (width - center_x))
    half_h_max = min(center_y, (height - center_y))

    # The final half-w/half-h must not exceed these maxima
    half_w = min(half_w_desired, half_w_max)
    half_h = min(half_h_desired, half_h_max)

    # Construct the new bounding box
    x1_new = center_x - half_w
    x2_new = center_x + half_w
    y1_new = center_y - half_h
    y2_new = center_y + half_h

    # Convert to integers (e.g. round or floor/ceil).
    # We'll just round them here for simplicity.
    x1_new = int(round(x1_new))
    x2_new = int(round(x2_new))
    y1_new = int(round(y1_new))
    y2_new = int(round(y2_new))

    return x1_new, y1_new, x2_new, y2_new


def squre_bounding_box(x1, y1, x2, y2, original_shape):
    width = x2 - x1
    height = y2 - y1
    cx = x1 + width // 2
    cy = y1 + height // 2
    side = max(width, height)
    new_x1 = max(0, cx - side // 2)
    new_y1 = max(0, cy - side // 2)
    new_x2 = min(original_shape[1], new_x1 + side)
    new_y2 = min(original_shape[0], new_y1 + side)

    return new_x1, new_y1, new_x2, new_y2


def preprocess(img, size=(160, 160)):
    """Preprocessing step before feeding the network.
    Arguments:
        img: a uint numpy array of shape [h, w, c] BGR format.
    Returns:
        a float numpy array of shape [1, c, h, w] BGR format.
    """
    h, w = img.shape[:2]
    target_w, target_h = size

    # Compute the scaling factor such that the image
    # fits inside target_size while preserving aspect ratio.
    scale = min(target_w / float(w), target_h / float(h))

    # Compute new resized dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize the image (this does not distort aspect ratio)
    resized = cv2.resize(img, (new_w, new_h))

    # Create an output array of the full target size, filled with zeros (black background)
    out = np.zeros((target_h, target_w, 3), dtype=resized.dtype)

    # Center the resized image within the output array
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    out[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    # img_show = cv2.resize(img, size)
    out = out.astype(np.float32)
    out = out.transpose((2, 0, 1))  # HWC to CHW
    out = np.expand_dims(out, 0)  # change to batch form
    out = (out / 255.0 - 0.5) / 0.5  # normalize image
    return out


def do_align_dfa(bbox: tuple[int, int, int, int, float],
                 image: np.ndarray, original_shape: tuple[int, int],
                 threshold=0.5) -> np.ndarray:
    """make the input image align, so it is a good image for input of vectorization

    Args:
        bbox (tuple[int, int, int, int, float]): x1, y1, x2, y2, conf based on pixel for face.
        image (np.ndarray): ,original image (not cropted face. the main raw input image)
        image_shape (tuple[int, int]): shape of input image, [height, width]
        note:

    Returns:
        np.ndarray: alignmenet image
    """
    x1, y1, x2, y2, _ = bbox
    x1, y1, x2, y2 = squre_bounding_box(x1, y1, x2, y2, original_shape)
    x1, y1, x2, y2 = pad_bounding_box(x1, y1, x2, y2, original_shape, padding=0.5)
    face_crop = image[y1:y2, x1:x2, :]
    face_crop = preprocess(face_crop, size=(160, 160))
    # will change to triton inside of mtcnn_step3_onnx
    bbox, landmark = dfa_triton(face_crop, threshold=threshold)
    if landmark is None or bbox is None:
        return None

    try:
        warped_face = dfa_warp(face_crop, landmark=landmark)
    except Exception as e:
        return None

    return warped_face


def do_align_dfa_oncrop(
        # bbox:tuple[int, int, int, int, float],
        face_crop: np.ndarray,
        #    original_shape:tuple[int, int],
        threshold=0.5) -> np.ndarray:
    """make the input image align, so it is a good image for input of vectorization

    Args:
        bbox (tuple[int, int, int, int, float]): x1, y1, x2, y2, conf based on pixel for face.
        image (np.ndarray): ,original image (not cropted face. the main raw input image)
        image_shape (tuple[int, int]): shape of input image, [height, width]
        note:

    Returns:
        np.ndarray: alignmenet image
    """
    # x1, y1, x2, y2, _ = bbox
    # x1, y1, x2, y2 = squre_bounding_box(x1, y1, x2, y2, original_shape)
    # x1, y1, x2, y2 = pad_bounding_box(x1, y1, x2, y2, original_shape, padding=0.5)
    # face_crop = image[y1:y2, x1:x2, :]
    face_crop = preprocess(face_crop, size=(160, 160))
    # will change to triton inside of mtcnn_step3_onnx
    bbox, landmark = dfa_triton(face_crop, threshold=threshold)
    if landmark is None or bbox is None:
        return None

    try:
        warped_face = dfa_warp(face_crop, landmark=landmark)
    except Exception as e:
        return None

    return warped_face


def do_align_dfa_onbbxo(
        bbox: tuple[int, int, int, int],
        image: np.ndarray,
        original_shape: tuple[int, int],
        threshold=0.5,
) -> np.ndarray:
    """make the input image align, so it is a good image for input of vectorization

    Args:
        bbox (tuple[int, int, int, int, float]): x1, y1, x2, y2, conf based on pixel for face.
        image (np.ndarray): ,original image (not cropted face. the main raw input image)
        image_shape (tuple[int, int]): shape of input image, [height, width]
        note:

    Returns:
        np.ndarray: alignmenet image
    """
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = squre_bounding_box(x1, y1, x2, y2, original_shape)
    x1, y1, x2, y2 = pad_bounding_box_symmetrically(x1, y1, x2, y2, original_shape, padding=0.5)
    face_crop = image[y1:y2, x1:x2, :]
    face_crop = preprocess(face_crop, size=(160, 160))
    # will change to triton inside of mtcnn_step3_onnx
    bbox, landmark = dfa_triton(face_crop, threshold=threshold)
    if landmark is None or bbox is None:
        return None

    try:
        warped_face = dfa_warp(face_crop, landmark=landmark)
    except Exception as e:
        return None

    return warped_face
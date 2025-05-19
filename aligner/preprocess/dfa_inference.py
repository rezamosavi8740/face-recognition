import numpy as np
import preprocess.allign_helper as aligner_helper
import torch
import torch.nn.functional as F
import tritonclient.http as model_client

TRITON_SERVER_URL = "localhost:{port_}".format(port_=8000)
triton_client = model_client.InferenceServerClient(url=TRITON_SERVER_URL)
TRITON_MODEL_NAME = "dfa_landmark_detection"  # Change this to match your deployed model name


def dfa_triton(image, threshold: float):
    # bounding_boxes = np.array([bounding_box])

    # Convert image to float32 and reshape for Triton
    # image2 = image.astype(np.float32)
    inputs = model_client.InferInput("input", image.shape, "FP32")
    # inputs.set_data_from_numpy(image, binary_data=True)#worked in http mode well
    inputs.set_data_from_numpy(image)
    # Send request to Triton
    results = triton_client.infer(model_name=TRITON_MODEL_NAME, inputs=[inputs])
    # Get outputs from Triton
    bbox = results.as_numpy('bbox')  # shape [n_boxes, 10]
    score = results.as_numpy('score')  # shape [n_boxes, 4]
    landmarks = results.as_numpy('landmarks')  # shape [n_boxes, 2]
    if score < threshold:
        print(f"Failed to get dfa landmarks with prob: {score[0]:.2f}")
        return None, None

    return bbox, landmarks


def dfa_warp(input_data, landmark, input_size=160, output_size=112):
    reference_ldmk = aligner_helper.reference_landmark()

    cv2_tfms = aligner_helper.get_cv2_affine_from_landmark(torch.from_numpy(landmark.reshape(-1, 10)), reference_ldmk,
                                                           input_size, input_size)
    thetas = aligner_helper.cv2_param_to_torch_theta(cv2_tfms, input_size, input_size, output_size, output_size)
    output_size = torch.Size((len(thetas), 3, output_size, output_size))
    grid = F.affine_grid(thetas, output_size, align_corners=True)
    aligned_x = F.grid_sample(torch.from_numpy(input_data) + 1, grid,
                              align_corners=True) - 1  # +1, -1 for making padding pixel 0
    aligned_ldmks = aligner_helper.adjust_ldmks(torch.from_numpy(landmark), thetas)

    aligned_x_np = aligned_x.numpy()  # CHW = 3, 112, 112
    # normalized_array = (aligned_x_np + 1) / 2  # Shift and scale values from [-1, 1] to [0, 1]
    # # Step 2: Scale to the range [0, 255] and convert to uint8
    # uint8_image = (normalized_array * 255).astype(np.uint8)
    return aligned_x_np
import tritonclient.http as model_client
import numpy as np


TRITON_SERVER_URL = "localhost:{port_}".format(port_=8000)
triton_client = model_client.InferenceServerClient(url=TRITON_SERVER_URL)
TRITON_MODEL_NAME =  "face_recognition"  # Change this to match your deployed model name


def extract_features_triton(batch_tensor):
    """
    Run batch inference on the model using Triton.
    
    Args:
        batch_tensor (np.ndarray): Input batch tensor with shape (batch_size, 3, 112, 112)
    
    Returns:
        list: List of normalized feature vectors for each image
    """
    # Create input for Triton
    input_data = model_client.InferInput("input", batch_tensor.shape, "FP32")
    input_data.set_data_from_numpy(batch_tensor)

    # Send request to Triton
    results = triton_client.infer(model_name=TRITON_MODEL_NAME, inputs=[input_data])

    # Extract outputs
    features = results.as_numpy("output")  # Shape: (batch_size, 512)

    # Normalize feature vectors
    normalized = []
    for vec in features:
        norm = np.linalg.norm(vec)
        normalized.append((vec / norm).tolist() if norm else vec.tolist())

    return normalized
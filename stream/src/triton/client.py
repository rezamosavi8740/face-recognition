
from src.config import CONFIG
from typing import Callable, Dict, Any
import asyncio
import numpy as np

# Common utility
from tritonclient.utils import np_to_triton_dtype

# GRPC
from tritonclient.grpc.aio import InferenceServerClient as GrpcClient
from tritonclient.grpc import InferInput as GrpcInferInput, InferRequestedOutput as GrpcInferOutput

# HTTP
from tritonclient.http.aio import InferenceServerClient as HttpClient
from tritonclient.http import InferInput as HttpInferInput, InferRequestedOutput as HttpInferOutput




class TritonModel:
    def __init__(
        self,
        name: str,
        input_name: str,
        output_names: list[str],
        pre_process: Callable[[Any], np.ndarray],
        post_process: Callable[[dict[str, np.ndarray]], Any],
    ):
        self.name = name
        self.input_name = input_name
        self.output_names = output_names
        self.pre_process = pre_process
        self.post_process = post_process

    def build_io(self, np_input: np.ndarray, protocol: str):
        if protocol == "grpc":
            from tritonclient.grpc import InferInput as GrpcInferInput, InferRequestedOutput as GrpcInferOutput

            infer_input = GrpcInferInput(
                self.input_name,
                np_input.shape,
                np_to_triton_dtype(np_input.dtype)
            )
            infer_input.set_data_from_numpy(np_input)

            infer_outputs = [GrpcInferOutput(name) for name in self.output_names]

        elif protocol == "http":
            from tritonclient.http import InferInput as HttpInferInput, InferRequestedOutput as HttpInferOutput

            infer_input = HttpInferInput(
                self.input_name,
                np_input.shape,
                np_to_triton_dtype(np_input.dtype)
            )
            infer_input.set_data_from_numpy(np_input)

            infer_outputs = [HttpInferOutput(name) for name in self.output_names]

        else:
            raise ValueError(f"Unsupported protocol: {protocol}")

        return [infer_input], infer_outputs


class TritonModelManager:
    _instance = None

    def __init__(self):
        self.client = None
        self.models: Dict[str, TritonModel] = {}
        self.protocol = CONFIG.triton.use.lower()
        self.url = CONFIG.triton.url

    @classmethod
    async def get_instance(cls):
        if cls._instance is None:
            self = cls()
            await self._init_client()
            cls._instance = self
        return cls._instance

    async def _init_client(self):
        if self.protocol == "grpc":
            self.client = GrpcClient(url=self.url)
        elif self.protocol == "http":
            self.client = HttpClient(url=self.url)
        else:
            raise ValueError(f"Unsupported protocol: {self.protocol}")

        if not await self.client.is_server_ready():
            raise RuntimeError(f"Triton server not ready at {self.url}")

    def register_model(self, model: TritonModel):
        self.models[model.name] = model

    async def infer(self, model_name: str, raw_input: Any):
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not registered.")

        model = self.models[model_name]
        input_tensor = model.pre_process(raw_input)
        inputs, outputs = model.build_io(input_tensor, protocol=self.protocol)

        response = await self.client.infer(model_name=model.name, inputs=inputs, outputs=outputs)

        # Collect all outputs into a dict
        output_data = {
            name: response.as_numpy(name) for name in model.output_names
        }
        

        return model.post_process(output_data)

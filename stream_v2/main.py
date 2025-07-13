import asyncio
from src.flow import Bina
# from prometheus_client import start_http_server
# import os 

# start_http_server(int(os.getenv("PROMETHEUS_PORT", 9026)))

# docker run  --gpus device=0 --restart always --name triton_bina_v2 -v $(pwd):/models  -p 8090:8000 -p 8091:8001 -p 8092:8002 nvcr.io/nvidia/tritonserver:25.01-py3 tritonserver --model-repository=/models
# trtexec --onnx=model.onnx  --saveEngine=model.plan  --minShapes=images:1x3x480x800 --optShapes=images:8x3x480x800 --maxShapes=images:16x3x480x800 --fp16 --device=0
# trtexec --onnx=model.onnx  --saveEngine=model.plan  --minShapes=images:1x3x288x480 --optShapes=images:8x3x288x480 --maxShapes=images:16x3x288x480 --fp16 --device=0

# trtexec --onnx=model.onnx  --saveEngine=model.plan --minShapes=input:1x3x160x160  --optShapes=input:1x3x160x160 --maxShapes=input:16x3x160x160 --device=0
# trtexec --onnx=model.onnx  --saveEngine=model.plan --minShapes=input:1x3x112x112  --optShapes=input:1x3x112x112 --maxShapes=input:16x3x112x112 --device=0
# trtexec --onnx=model.onnx  --saveEngine=model.plan --minShapes=input:1x3x112x112  --optShapes=input:1x3x112x112 --maxShapes=input:16x3x112x112 --device=0
# trtexec --onnx=model.onnx  --saveEngine=model.plan  --minShapes=pixel_values:1x3x224x224 --optShapes=pixel_values:1x3x224x224 --maxShapes=pixel_values:16x3x224x224 --fp16 --device=0



#   



async def main():
    try:
        bina = await Bina.create()
        await bina.inference()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    asyncio.run(main())

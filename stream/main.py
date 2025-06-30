import asyncio
from src.flow import Bina
from prometheus_client import start_http_server
import os 
# ---- Add yappi here ----
import yappi
# import tracemalloc
# tracemalloc.start()

# Start Prometheus server
# start_http_server(int(os.getenv("PROMETHEUS_PORT", 9026)))

# docker run --gpus all --restart always --name triton_bina_v2 -v $(pwd):/models  -p 8090:8000 -p 8091:8001 -p 8092:8002 nvcr.io/nvidia/tritonserver:25.01-py3 tritonserver --model-repository=/models
# trtexec --onnx=model.onnx  --saveEngine=model.plan  --minShapes=images:1x3x480x800 --optShapes=images:8x3x480x800 --maxShapes=images:16x3x480x800 --fp16 --device=1
# docker run --gpus all --rm --name trt_build -v$(pwd):/models -it nvcr.io/nvidia/tensorrt:25.01-py3 bash

# async def main():
#     try:
#         bina = await Bina.create()
#         await bina.inference()
#     except KeyboardInterrupt:
#         await bina.shutdown()


# if __name__ == "__main__":
#     asyncio.run(main())






def start_profiler():
    yappi.set_clock_type("cpu")  # or "wall" for wall time
    yappi.start()

def stop_profiler():
    yappi.stop()
    stats = yappi.get_func_stats()
    stats.sort("tsub", "desc")
    
    # Save human-readable report
    with open("yappi_cpu_report.txt", "w") as f:
        stats.print_all(out=f)

    # Save in pstat format for visualization tools
    stats.save("yappi_stats.prof", type="pstat")

# -------------------------

start_http_server(int(os.getenv("PROMETHEUS_PORT", 9026)))

async def main():
    try:
        # start_profiler()  # Start profiler before running
        bina = await Bina.create(use_load="aiortc")
        await bina.inference()
    except KeyboardInterrupt:
        await bina.shutdown()
    finally:
        pass
        # stop_profiler()  # Always stop profiler even on exceptions

if __name__ == "__main__":
    asyncio.run(main())

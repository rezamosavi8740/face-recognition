import asyncio
from prometheus_client import start_http_server
import os 
# Start Prometheus server
start_http_server(int(os.getenv("PROMETHEUS_PORT", 9000)))
if os.getenv("LOADER__QUEUE", "false") == "false" :
    from src.flow import Bina
else:
    from src.flow2 import Bina

# docker run --gpus all --rm --name bina_2 -v $(pwd):/models  -p 8090:8000 -p 8091:8001 -p 8092:8002 nvcr.io/nvidia/tritonserver:25.01-py3 tritonserver --model-repository=/models

async def main():
    try:
        bina = await Bina.create(str(os.getenv("LOADER__USE", "aiortc")))
        await bina.inference()
    except KeyboardInterrupt:
        await bina.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
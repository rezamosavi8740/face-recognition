import asyncio
from prometheus_client import start_http_server
import os 
from src.flow import Bina
# from src.flow2 import Bina


start_http_server(int(os.getenv("PROMETHEUS_PORT", 9000)))

async def main():
    try:
        bina = await Bina.create()
        await bina.inference()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    asyncio.run(main())
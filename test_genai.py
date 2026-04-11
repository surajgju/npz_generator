import inspect
from google import genai
import asyncio

async def main():
    try:
        from google.genai.live import AsyncSession
        print("AsyncSession.send signature:")
        print(inspect.signature(AsyncSession.send))
        print("AsyncSession.send_realtime_input signature:")
        print(inspect.signature(AsyncSession.send_realtime_input))
        print("LiveSession.receive signature:")
        print(inspect.signature(AsyncSession.receive))
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(main())

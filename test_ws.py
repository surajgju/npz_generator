import asyncio
import websockets

async def test():
    try:
        async with websockets.connect("ws://localhost:8000/ws/anim", origin="http://localhost:5173") as ws:
            print("Connected!")
            await ws.send('{"type": "anim_subscribe", "protocol_version": 2}')
            res = await ws.recv()
            print("Received:", res)
            await asyncio.sleep(2)
    except Exception as e:
        print("Error:", repr(e))

asyncio.run(test())


import asyncio
import websockets
import json

async def test_binance_ws():
    url = "wss://stream.binance.com:9443/ws/btcusdt@ticker"
    print(f"Connecting to {url}...")
    try:
        async with websockets.connect(url) as websocket:
            print("Connected! Waiting for message...")
            message = await asyncio.wait_for(websocket.recv(), timeout=10)
            data = json.loads(message)
            print(f"Received ticker for {data['s']}: Last Price {data['c']}")
            return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_binance_ws())
    if success:
        print("Websocket test PASSED")
    else:
        print("Websocket test FAILED")
        exit(1)
from ml_engine import *
import cv2
import websockets
import asyncio
import threading
import time
import requests

class SocketEngine:
    def __init__(self, mlengine, port):
        self.me = mlengine
        self.port = port
        self.connected = set()

        asyncio.run(self.run())
        print('socket')

    async def send(self):
        while True:
            if self.me.get_is_new_frame() == True:
                frame = self.me.get_frame()
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
                _, imgencode = cv2.imencode('.jpg', frame, encode_param)
                imgencode = np.array(imgencode)
                
                # websockets.broadcast(self.connected, 'imgencode.tobytes()')
                websockets.broadcast(self.connected, imgencode.tobytes())
                await asyncio.sleep(0)
            else:
                await asyncio.sleep(0)

 


    async def accept(self, websocket, path):
        self.connected.add(websocket)
        # res2 = requests.post('http://163.180.117.39:8092/api/test', json={'test': list(self.connected)})
        try:
            await websocket.wait_closed()

        finally:
            self.connected.remove(websocket)


    async def run(self):
        # loop = asyncio.get_running_loop()
        # stop = loop.create_future()

        async with websockets.serve(self.accept, '0.0.0.0', self.port):
            await self.send()
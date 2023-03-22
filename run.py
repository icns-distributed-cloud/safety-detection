import logging
import argparse
import time

from ml_engine import *
from socket_engine import *


def init_logger():
    _logger = logging.getLogger('Main')
    _logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename='test.log')
    file_handler.setLevel(logging.INFO)
    
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setLevel(logging.INFO)

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    _logger.addHandler(file_handler)
    _logger.addHandler(stream_handler)
    return _logger



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fire Smoke Detection on IPCamera')
    parser.add_argument('-i', '--ipcam', type=str, help='IPCamera URL')
    parser.add_argument('-s', '--socketport', type=str, help='Socket URL')
    parser.add_argument('-c', '--cctvid', type=str, help='CCTV Id')
    parser.add_argument('-r', '--resturl', type=str, help='abnormal rest url')
    parser.add_argument('-w', '--websocketurl', type=str, help='abnormal websocket url')
    parser.add_argument('-p', '--posid', type=str, help='cctv position id')
    args = parser.parse_args()




    logger = init_logger()
   

    '''
    me = MLEngine(
        # 'rtsp://icns:iloveicns@icns01.iptimecam.com:21124/stream_ch00_1',
        'data/video17.avi',
        30501,
        17)
    '''

    print(args)

    me = MLEngine(
        args.ipcam,
        args.socketport,
        args.cctvid,
        args.resturl,
        args.websocketurl,
        args.posid
    )

    print('MLEngine done')

    time.sleep(3)

    se = SocketEngine(
        me, 
        args.socketport)

    print('SocketEngine done')
    # asyncio.run(se.run())

    # me.run()


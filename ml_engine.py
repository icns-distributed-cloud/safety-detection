from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *


import cv2
import threading
import requests

from video_capture import *

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import logging


class MLEngine:
    def __init__(self, ipcam, socketport, cctvid, resturl, websocketurl, posid):
        self.init_logger()
        self.resturl = resturl
        self.websocketurl = websocketurl

        self.cap = BufferlessVideoCapture(ipcam)
        # self.cap = cv2.VideoCapture(ipcam)
        print('VideoCapture done')

        self.socketport = socketport
        self.cctvid = cctvid
        self.posid = posid

        self.Tensor = torch.cuda.FloatTensor

        self.model = Darknet('config/yolov3-tiny.cfg', img_size=416).to(torch.device('cuda'))
        
        # self.model.load_state_dict(torch.load('checkpoints/2/tiny1_301.pth'))
        self.model.load_state_dict(torch.load('checkpoints/fire-smoke-650.pth'))
        
        
        self.model.eval()

        # self.classes = load_classes('config/helmet.names')
        self.classes = load_classes('config/fire-smoke.names')

        self.color = [(0, 0, 255 ), (0, 255, 0)]
        self.a = []

        self.draw_frame = 0
        self.is_new_frame = False
        self.predict_done = True

        self.fire_count = 0
        self.smoke_count = 0
        self.isAbnormal = False

        self.thr = threading.Thread(target=self.run, args=(ipcam, ))
        self.thr.daemon = True
        self.thr.start()


    def get_is_new_frame(self):
        return self.is_new_frame


    def get_frame(self):
        self.is_new_frame = False
        return self.draw_frame

    def init_logger(self):
        logger = logging.getLogger('Main.MLEngine')
        logger.setLevel(logging.INFO)
        self.logger = logger


    def run(self, ipcam):
        while self.cap.isOpened():
            # if self.predict_done == True:
            self.predict_done = False
            ret, frame = self.cap.read()
            if ret == False:
                self.cap = cv2.VideoCapture(ipcam)
                continue
            

            draw_frame = frame.copy()
            PILimg = np.array(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            # PILimg = np.array(Image.fromarray(frame))
            # cv2.imshow('d', PILimg)
            imgTensor = transforms.ToTensor()(PILimg)
            imgTensor, _ = pad_to_square(imgTensor, 0)
            imgTensor = resize(imgTensor, 416)
            imgTensor = imgTensor.unsqueeze(0)
            imgTensor = Variable(imgTensor.type(self.Tensor))

            with torch.no_grad():
                detections = self.model(imgTensor)
                detections = non_max_suppression(detections, 0.8, 0.4)

            self.a.clear()
            if detections is not None:
                self.a.extend(detections)
            
            if len(self.a):
                for detections in self.a:
                    if detections is not None:
                        detections = rescale_boxes(detections, 416, PILimg.shape[:2])

                        for x1, y1, x2, y2, conf, _, cls_pred in detections:
                            if self.classes[int(cls_pred)] == 'fire':
                                self.fire_count += 1
                            elif self.classes[int(cls_pred)] == 'smoke':
                                self.smoke_count += 1

                            if self.fire_count > 10 or self.smoke_count > 10:
                                # 아래 2줄이 진짜
                                draw_frame = cv2.rectangle(draw_frame, (int(x1), int(y1)), (int(x2), int(y2)), self.color[int(cls_pred)], 2)
                                cv2.putText(draw_frame, self.classes[int(cls_pred)], (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 2, self.color[int(cls_pred)], 6)
                            
                                if self.isAbnormal == False:
                                    self.isAbnormal = True
                                    try:
                                        res = requests.post(f'http://{self.websocketurl}/api/SocketDataReceive', data={"posid": f"{self.posid}"})
                                        res = requests.post(f'http://{self.resturl}/api/accident/smoke/1')
                                    except:
                                        pass


                            # box_h = y2 - y1

                            # color = [int(c) for c in self.colors[int(cls_pred)]]

                            # draw_frame = cv2.rectangle(draw_frame, (int(x1), int(y1+box_h)), (int(x2), int(y1)), self.color, 2)


                            
                            



                            # cv2.putText(draw_frame, str("%.2f" % float(conf)), (int(x2), int(y2 - box_h)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            #             self.color[int(cls_pred)], 2)

                    else:
                        self.fire_count -= 1
                        self.smoke_count -= 1

                        if self.fire_count < 10 and self.smoke_count < 10 and self.isAbnormal == True:
                            self.isAbnormal = False

            self.draw_frame = draw_frame
            # cv2.imshow('sd', draw_frame)
            # cv2.waitKey(1)
            self.is_new_frame = True
            self.predict_done = True
        # self.cap.release()

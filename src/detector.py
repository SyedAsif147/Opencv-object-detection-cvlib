# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:07:29 2020

@author: Syed
"""

import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
import os

# filename = 'video.avi'
# frames_per_second = 24.0
# res = '720p'
    
# VIDEO_TYPE = {
#     'avi': cv2.VideoWriter_fourcc(*'XVID'),
#     #'mp4': cv2.VideoWriter_fourcc(*'H264'),
#     'mp4': cv2.VideoWriter_fourcc(*'XVID'),
# }

# def change_res(cap, width, height):
#     cap.set(3, width)
#     cap.set(4, height)

# STD_DIMENSIONS =  {
#     "480p": (640, 480),
#     "720p": (1280, 720),
#     "1080p": (1920, 1080),
#     "4k": (3840, 2160),
# }

# def get_video_type(filename):
#     filename, ext = os.path.splitext(filename)
#     if ext in VIDEO_TYPE:
#       return  VIDEO_TYPE[ext]
#     return VIDEO_TYPE['avi']

# def get_dims(cap, res='1080p'):
#     width, height = STD_DIMENSIONS["480p"]
#     if res in STD_DIMENSIONS:
#         width,height = STD_DIMENSIONS[res]
#     ## change the current caputre device
#     ## to the resulting resolution
#     change_res(cap, width, height)
#     return width, height

im = cv2.imread('Data_images/apple-256261_640-618x395.jpg')

bbox, label, conf = cv.detect_common_objects(im)

output_image = draw_bbox(im, bbox, label, conf)
print(bbox, label, conf)
plt.imshow(output_image)
plt.savefig('object_detect.png')
plt.show()

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
# video_type = get_video_type(filename)
# dims = get_dims(cap, res)
# out = cv2.VideoWriter(filename, video_type, 1, dims)
# while True:
#     ret, frame = cap.read()
#     bbox, label, conf = cv.detect_common_objects(frame)
#     draw_bbox(frame, bbox, label, conf)
#     print(conf)
#     if len(conf)>0:
#         out.write(frame)
#     cv2.imshow('Detector', frame)
#     if cv2.waitKey(20) & 0xFF == ord('q'):
#         break    
    
# cap.release()
# cv2.destroyAllWindows()

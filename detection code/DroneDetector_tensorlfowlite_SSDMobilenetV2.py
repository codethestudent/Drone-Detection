#!/usr/bin/env python
# coding: utf-8

# In[18]:


import os
import argparse
import cv2
import numpy as np
import sys
import time
import importlib.util
from tensorflow.lite.python.interpreter import Interpreter

MODEL_NAME = 'custom_model_lite'
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'
use_TPU = False

# Program settings
min_conf_threshold = 0.50
resW, resH = 1280, 720 # Resolution to run camera at
imW, imH = resW, resH


# 디렉토리 경로 설정
CWD_PATH = "C:\\Users\\user\\Downloads\\custom_model_lite2"
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# 레이블 맵
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

#interpreter 클래스를 사용하여 모델 파일을 읽고 메모리에 할당, 실행에 필요한 텐서들의 메모리를 할당,
#초기화 작업 수행
interpreter = Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()

#TensorFlow Lite 모델의 입력과 출력에 대한 세부 정보를 가져오고, 입력 이미지의 높이와 너비
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

#부동소수점 타입인지 확인, 맞다면 이미지 데이터를 0-1사이의 실수 값으로 정규화해야함 (True)
#정수 타입이라면 정규화 필요 ㄴㄴ(False)
floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

# 모델의 첫 번째 출력 텐서의 이름
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

cap = cv2.VideoCapture(0)
ret = cap.set(3, resW)#가로 해상도
ret = cap.set(4, resH)#세로 해상도

bbox = None
cnt =0
fps_list = []
fps = 0
prev_time = 0
boxarea = 0

while True:

    hasFrame, frame1 = cap.read()
    #바운딩 박스 없거나 cnt=10일때마다
    if bbox == None or cnt%10==0:
        cnt+=1
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0) #이미지 차원을 늘림, 

        #이미지 정규화
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # 모델을 실행시키는 실질적인 코드, interpreter.set_tensor() 메서드를 사용하여 입력 데이터를 설정,
        #invoke()메서드를 호출해 출력 결과를 반환
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        # Bounding box coordinates of detected objects
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects
        print(scores)
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))

                # Draw bounding box
                cv2.rectangle(frame1, (xmin,ymin), (xmax,ymax), (255, 0, 0), 2)
                
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) 
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, labelSize[1] + 10) #레이블이 객체를 가리지 않도록설정
                cv2.rectangle(frame1, 
                              (xmin, label_ymin-labelSize[1]-10), 
                              (xmin+labelSize[0], label_ymin+baseLine-10), 
                              (255, 255, 255), 
                              cv2.FILLED)
                # Draw label text
                cv2.putText(frame1, 
                            label, 
                            (xmin, label_ymin-7), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, 
                            (0, 0, 0), 
                            2)
                tracker = cv2.TrackerMIL_create()
                bbox = (int(xmin), int(ymin), int(xmax-xmin), int(ymax-ymin)) 
                w = bbox[2]
                h = bbox[3]
                boxarea = width*height
                # Set initial bounding box for object tracking using detected object coordinates
                tracker.init(frame1, tuple(bbox))
                
    else:
        # Update the tracker with the new frame
        cnt+=1
        success, bbox = tracker.update(frame1)
        # Draw the bounding box around the tracked object
        if success:
            x, y, w, h = [int(i) for i in bbox]
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
            boxarea = w*h
        else:
            bbox = None # Reset the bounding box if tracking fails
            cv2.putText(frame1, "Tracking Failed", (200, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    if boxarea >= 4890:
        dist = np.log(boxarea/278343.948)/-0.95182
    else:
        dist = np.log(boxarea/8011.83)/-0.2001
        
    cv2.putText(frame1, f'distance : {dist:.1f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2) 
    cv2.putText(frame1, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('Object detector', frame1)
    
    cur_time = time.time()
    if prev_time == 0:
        prev_time = cur_time
    else: 
        fps = 1 / (cur_time - prev_time)
        prev_time = cur_time

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()


# In[8]:


scores


# In[ ]:





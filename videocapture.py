import os
import json

import cv2
import torch
import pandas as pd

from external_library.sort.sort import Sort

def captureFrames(cfg):
  print('\nstart video capturing')

  frame_width = 1000
  frame_height = 1000

  vc = cv2.VideoCapture(0)

  if not vc.isOpened():
    raise Exception("Could not open video capture")
  else:
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    
  
  #OBJECT DETECTION
  model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
  model.float()
  model.eval()
  
  #MODEL
  mot_tracker = Sort()

  while True:
    success, frame = vc.read()

    # check whether a frame is captured or not
    if not success:
      print("failed to capture frames")
      break

    # wait key ESC(no. 27) to break the while loop
    key = cv2.waitKey(1)
    if key == 27:
      print("Stopped capturing video")
      vc.release()
      cv2.destroyWindow("mac")
      break
    
    
    #OBJECT DETECTION
    results = model(frame)    
    df = results.pandas().xyxy[0]
    detections = df[df['name']=='person'].drop(columns='name').to_numpy()
    track_ids = mot_tracker.update(detections)
    # print(track_ids)

    for i in range(len(track_ids.tolist())):
      coords = track_ids.tolist()[i]
      xmin, ymin, xmax, ymax = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
      name_idx = int(coords[4])
      name = 'ID: {}'.format(str(name_idx))

      cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0 ,0), 2)
      cv2.putText(frame, name, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
      
      cv2.imshow("mac", frame)
  
if __name__ == "__main__":
  path_config = "./configs/config.json"
  with open(path_config) as f:
    cfg = json.load(f)
  
  captureFrames(cfg)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 18:11:35 2022

@author: thierryk
"""

import cv2
from PIL import Image
import time
import numpy as np

# define a video capture object
vid = cv2.VideoCapture(0)

capturing = True
frame_rate = 5
prev = 0
count = 0
name = "antoine"

while(capturing):    
    
    # Capture the video frame by frame
    time_elapsed = time.time() - prev
    ret, frame_instant = vid.read()

    if time_elapsed > 1./frame_rate:
        prev = time.time()
        
        frame_PIL = Image.fromarray(np.uint8(frame_instant)).convert('RGB')
        frame_PIL.save(f'../../dataset/new_img_MBA_2023/frame_{name}_{str(count)}.jpeg')
        count += 1
        # print(count)
    
    if count >= 60:    
        capturing = False

    
    if count % 10 == 0:
        print(count)


# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

end = time.time()

    
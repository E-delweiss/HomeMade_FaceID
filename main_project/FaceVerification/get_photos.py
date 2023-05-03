import os, sys
import cv2
import time
import numpy as np
import pickle

import torch
import torchvision

from PIL import Image


import subprocess


# define a video capture object
vid = cv2.VideoCapture(0)
capturing = True

prev = 0
count = 0
FRAME_RATE = 10
DIST_LIM = 0.05



while(capturing):
    # Capture the video frame by frame
    time_elapsed = time.time() - prev
    ret, frame_instant = vid.read()
    
    if time_elapsed > 1./FRAME_RATE:
        prev = time.time()

    count += 1
        
        
    
        
                
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

end = time.time()
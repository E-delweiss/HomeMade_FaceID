import os, sys
import cv2
import time
import numpy as np
import pickle

import torch
import torchvision

from PIL import Image

import fct_faceID as fcID
import imgPrep_faceID as fcPrep

import subprocess


# define a video capture object
vid = cv2.VideoCapture(0)
capturing = True

prev = 0
count = 0
FRAME_RATE = 10
DIST_LIM = 0.05


faceVerification_model = fcID.create_faceVerification_model('SiameseNet_calibrated2_12032022.pth')
# with open("anchor_base.pkl",'rb') as anchor:
    # anchor_embedding = pickle.load(anchor)

anchor_PIL = Image.open('anchor_base.jpg').convert('RGB')
anchor_t = fcPrep.encoder_temp(anchor_PIL, doNormalize=True, isPIL=True)
with torch.no_grad():
    anchor_embedding = faceVerification_model(anchor_t.unsqueeze(0))

# os.system("sh close_session.sh")

while(capturing):
    # Capture the video frame by frame
    time_elapsed = time.time() - prev
    ret, frame_instant = vid.read()
    
    if time_elapsed > 1./FRAME_RATE:
        prev = time.time()
    
        frame_preprocess = fcID.preprocess(frame_instant)
        
        #pred = faceDetection_model(frame_preprocess.unsqueeze(0))
        #output = bool(torch.max(pred, dim=1)[1])
        
        output = True
        if output == True:
            if count == 0:
                frame_1 = frame_instant
                frame_1_PIL = Image.fromarray(np.uint8(frame_1)).convert('RGB')
                print("Got the first image...")
                
            elif count == 1 :
                frame_2 = frame_instant
                frame_2_PIL = Image.fromarray(np.uint8(frame_2)).convert('RGB')
                print("Got the second image...")
                print("----------------------------------")
                print("Is the subject steady ?")
                is_steady = fcID.is_steady(frame_1_PIL, frame_2_PIL, dist_lim=DIST_LIM)
                
                if is_steady[0] is False:
                    print('\n')
                    print("Un probleme est survenu, je recommence")
                    print('\n')
                    count = -1
                    
                elif is_steady[0] is True:
                    print(f"La distance entre les deux visages est bien inférieure à {DIST_LIM*100}% de la diagonale de l'image")
                    print("Je renvois donc True")
                    print("La verification faciale peut commencer")
                    img_align = fcPrep.face_crop_align(frame_2_PIL, is_steady[1], margin=20)
                    if img_align is False:
                        break
                    img_t = fcPrep.transformer(img_align, doNormalize=True)
                    with torch.no_grad():
                        img_embedding = faceVerification_model(img_t.unsqueeze(0))
                        
                    d = np.linalg.norm(anchor_embedding.detach().numpy() - img_embedding.detach().numpy())
                    
                    if d > 0.5:
                        print("ERROR : NO AUTHORIZED")
                        print(d)
                        break
                    else :
                        # os.system("sh faceID.sh")
                        print("ok good")
                        print(d)
                        break
            
            
            count += 1
        
        
    
        
                
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

end = time.time()
        

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 21:09:42 2022

@author: thierry
"""

import numpy as np
import pprint 

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from PIL import Image

import torch
import torchvision

import fct_faceID as fcID

from facenet_pytorch import MTCNN, InceptionResnetV1


def face_crop_align(img_PIL:Image, bounding_box:dict, margin:int)-> Image:
    img_PIL_cropped = img_PIL.crop((
        bounding_box['x_left'] - margin, 
        bounding_box['y_top'] - margin,
        bounding_box['x_right'] + margin,
        bounding_box['y_bottom'] + margin
        ))
    
    output_dict = fcID.face_localization(img_PIL_cropped, landmarks=True)
    if output_dict is False:
        return False
    alpha, center = compute_align_angle(output_dict)
    img_align = img_PIL_cropped.rotate(alpha, center=center)
    img_align.save('temp.jpeg')
    return img_align


def compute_align_angle(output_dict):
    """
    

    Parameters
    ----------
    output_dict : TYPE
        DESCRIPTION.

    Returns
    -------
    alpha_deg : TYPE
        DESCRIPTION.
    center : TYPE
        DESCRIPTION.

    """
    oppose = np.abs(output_dict['landmarks']['eye_right'][1] - output_dict['landmarks']['eye_left'][1])
    adjacent = np.abs(output_dict['landmarks']['eye_right'][0] - output_dict['landmarks']['eye_left'][0])
    alpha_rad = np.arctan(oppose/adjacent)
    alpha_deg = np.rad2deg(alpha_rad)
    
    if output_dict['landmarks']['eye_right'][1] < output_dict['landmarks']['eye_left'][1]:
        # right eye higher than left eye
        alpha_deg = -alpha_deg
        center = (output_dict['landmarks']['eye_left'][0], output_dict['landmarks']['eye_left'][1])
    else : 
        # right eye lower than left eye
        center = (output_dict['landmarks']['eye_right'][0], output_dict['landmarks']['eye_right'][1])
        
    return alpha_deg, center



def transformer(input_img:Image, doNormalize:bool=False)->torch.Tensor:
    """Transform an input image

    -------------------
    Parameters:
        input: PIL.image
            Input to transform before feeding it to the model.
        doNormalize: bool, optionnal
            Determines if the function must normalize the image or not. Default is False.
    -------------------
    Returns:
        img_t: torch.Tensor of shape (3,160,160)
    """
    img_t = torchvision.transforms.ToTensor()(input_img)
    img_t = torchvision.transforms.Resize((160,160))(img_t)

    if doNormalize:
        mean, std = img_t.mean(), img_t.std()
        img_t = (img_t - mean) / std

    return img_t





def plot(output_dict, img, plot_box=True, plot_landmarks=True):
    """
    

    Parameters
    ----------
    output_dict : TYPE
        DESCRIPTION.
    img : TYPE
        DESCRIPTION.
    plot_box : TYPE, optional
        DESCRIPTION. The default is True.
    plot_landmarks : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    box_coord : TYPE
        DESCRIPTION.

    """
    width = np.abs(output_dict['box'][2] - output_dict['box'][0])
    height = np.abs(output_dict['box'][1] - output_dict['box'][3])
    
    
    plt.imshow(img)
    if plot_box :
        plt.gca().add_patch(Rectangle((output_dict['box'][0],output_dict['box'][1]),
                                      width, height,
                                      edgecolor='red',
                                      facecolor='none',
                                      lw=2))
    
    if plot_landmarks :
        color = ['red', 'blue', 'orange', 'green', 'purple']
        for it, key in enumerate(output_dict['landmarks']):
            plt.plot(output_dict['landmarks'][key][0], output_dict['landmarks'][key][1], marker='x', markersize=7, color=color[it])
            







def encoder_temp(input, isCrop:bool=False, doNormalize:bool=False, isPIL:bool=False)->torch.Tensor:
    if not isCrop:
        detector = MTCNN()
        if isPIL is False:
            PIL = Image.open(input).convert('RGB')
        else:
            PIL = input

        x1, y1, x2, y2 = detector.detect(PIL)[0][0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        img_t = torchvision.transforms.ToTensor()(PIL)
        img_t = img_t[:,y1:y2, x1:x2]
        img_t = torchvision.transforms.Resize((160,160))(img_t)
    else : 
        img_t = input

    if doNormalize:
        mean, std = img_t.mean(), img_t.std()
        img_t = (img_t - mean) / std

    return img_t
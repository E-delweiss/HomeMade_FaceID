import numpy as np

from PIL import Image
import glob, os

import torch
import torch.nn as nn
import torchvision

from facenet_pytorch import MTCNN, InceptionResnetV1



def face_localization(img_PIL:Image, landmarks:bool=True)->dict:
    ### Face bounding box
    mtcnn = MTCNN()
    output = mtcnn.detect(img_PIL, landmarks=landmarks)
    

    
    if output[0] is not None:
        output_dict = {'box':output[0][0], 
                       'confidence':output[1]}
    
        output_dict = {'x_left' : output_dict['box'][0],
                 'x_right' : output_dict['box'][2],
                 'y_bottom': output_dict['box'][3],
                 'y_top' : output_dict['box'][1]}
        return output_dict
    
    else:
        print("Aucune bounding box !")
        print("output = ", output)

    return None





if __name__ == "__main__":
    img_paths = glob.glob("../../dataset/new_img_MBA_2023/val_set/pasmoi/*")
    margin = 2

    for img in img_paths:
        if "crop" in img.split("/")[-1]:
            pass
        else:
            img_PIL = Image.open(img).convert('RGB')
            bounding_box = face_localization(img_PIL, False)

            if bounding_box:
                img_PIL_cropped = img_PIL.crop((
                    bounding_box['x_left'] - margin, 
                    bounding_box['y_top'] - margin,
                    bounding_box['x_right'] + margin,
                    bounding_box['y_bottom'] + margin
                ))

                # crop_img_name = f"../../dataset/new_img_MBA_2023/new_img_MBA_2023/val_set/pasmoi/{img}".split("/")[-1]
                os.remove(img)
                img_PIL_cropped.save(img)
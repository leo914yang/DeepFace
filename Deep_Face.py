import numpy as np
import pandas as pd
from deepface import DeepFace
import cv2
import Image_processing as ip


def verify_face(img_path, i1=0, i2=1, m1=2):
    img1 = cv2.imread(img_path[i1])
    img2 = cv2.imread(img_path[i2])
    print(DeepFace.verify(img1, img2))

    model_list = ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib']
    result  = DeepFace.verify(img1, img2, model_name=model_list[m1])
    print(result['verified'])


def find_face(img_path, dirname, path):
    for i in dirname:
        df = DeepFace.find(img_path=img_path, db_path=path + '/' + i, enforce_detection=False)
        
        print(df)


if __name__ == '__main__':
    path = 'C:/git_workspace/DeepFace/image'
    img_path = ip.image_path(path)
    #verify_face(img_path)

    img_path1, dirname = ip.image_path_AllFolder(path)
    
    for i in img_path1:
        find_face(i, dirname, path)
    
    





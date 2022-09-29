import argparse
import os
import time
import glob

from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torchvision
from PIL import Image
import torch.nn as nn

from metriclearning.model import TripletNet

from sklearn.neighbors import KNeighborsClassifier



def inference(bbox_img,model_path):

     img_compose = torchvision.transforms.Compose([
         torchvision.transforms.Resize((64,64)),
         torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
     ])

     composed_im = img_compose(bbox_img)

     composed_img = torch.stack([composed_im for _ in range(64)])

    #  print(np.shape(composed_im))
    #  print(np.shape(composed_img))

     device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
     model = TripletNet().to(device)
     model.eval()
     model.load_state_dict(torch.load(model_path))

     with torch.no_grad():

         composed_img = composed_img.to(device)
         feature = model(composed_img).detach().cpu().numpy()

         feature = feature.reshape(feature.shape[0], feature.shape[1])

     return np.array(feature)



def calc_metric(numpy_data,numpy_labels,bbox_img,model_path):

     model_path = model_path

     feature = inference(bbox_img,model_path)[0]

     knn = KNeighborsClassifier(n_neighbors=10)

     knn.fit(numpy_data,numpy_labels)

     ML_class = knn.predict([feature])[0]

     return ML_class


if __name__=="__main__":

     bbox_img = Image.open("data/MetricLearning5/test/0/000262.PNG")

     model_path = "work_dirs/ML_finn/model_ML_final.pth"

     npz = np.load("work_dirs/ML_finn/features.npz")

     numpy_data,numpy_labels = npz["arr_0"],npz["arr_1"]

     ML_class = calc_metric(numpy_data,numpy_labels,bbox_img,model_path)

     print(ML_class)
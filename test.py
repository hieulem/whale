
from __future__ import print_function, division

from PIL import Image
from PIL import ImageFilter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
test_transforms = data_transforms['test']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = torchvision.models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(device)
model_ft.load_state_dict(torch.load('./trained_model/fold_3/epoch_24.pth'))
model_ft.eval() #set model to eval mode
data_dir = './All_aerial_training/all/train_bak/'
data_dir = './fulldata/fold_3/val/'
image_datasets = datasets.ImageFolder(data_dir, data_transforms['test'])
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=48,
                                             shuffle=False, num_workers=4)
print(image_datasets)
def test_im(model_ft,im):
    A_img = Image.open(im)
    A_img = A_img.resize((224, 224),Image.NEAREST)
    A_img = test_transforms(A_img)
    A_img = torch.unsqueeze(A_img,0)
    pred = model_ft(A_img)
    if pred[0,0] > pred[0,1]:
        print("Whale")
    else:
        print("not whale")

def test_dir(model_ft,dataloader):
    tp=0
    fp=0
    tn= 0
    fn =0
    for im, labs in dataloader:
        outputs = model_ft(im)
        _,preds = torch.max(outputs,1)
        tp = tp+ torch.sum(preds[labs==0] == 0)
        fn = fn+ torch.sum(preds[labs==0] == 1)
        fp = fp +torch.sum(preds[labs==1] == 0)
        tn = tn + torch.sum(preds[labs==1] ==1)
    
    print(tp,fp,fn,tn)
    prec = float(tp)/float(tp+fp)
    recall =  float(tp)/ float(tp+fn)
    print("prec: %f, recall: %f"%(prec,recall))



test_dir(model_ft,dataloaders)

#im = 'whale.png'
#test_im(model_ft,im)
#im = 'water.png'
#test_im(model_ft,im)

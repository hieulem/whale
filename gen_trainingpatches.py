import numpy as np
from PIL import Image 
import time
import torch
import os.path
import argparse
from scipy import misc
from m_util import *
parse = argparse.ArgumentParser()
parse.add_argument('--dataset')
opt = parse.parse_args()
opt.root = '/gpfs/projects/LynchGroup/Penguin_workstation/Train_all/fullsize/'
opt.root = './Water_Training/'
opt.im_fold = opt.root
opt.step = 500 #128 for testing, 64 for training
opt.size = 30 #256 for testing, 386 for training

opt.results = './tiled_water_'+str(opt.step)+'/'
sdmkdir(opt.results)
opt.input_nc =3
imlist=[]
imnamelist=[]

for root,_,fnames in sorted(os.walk(opt.root)):
    for fname in fnames:
        if fname.endswith('.PNG'):
            path = os.path.join(root,fname)
            imlist.append((path,fname))
            imnamelist.append(fname)
            
for im_path,imname in  imlist:
    png = misc.imread(im_path,mode='RGB')
    w,h,z = png.shape
    savepatch_train(png,w,h,opt.step,opt.size,opt.results+'/'+imname[:-4]+'#')

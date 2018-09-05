from shutil import copyfile
import argparse
import os.path
from m_util import *
opt = argparse.ArgumentParser().parse_args()
#opt.im_fold = '/nfs/bigbox/hieule/penguin_data/CROPPED/p500'
#opt.im_fold='/gpfs/projects/LynchGroup/Penguin_workstation/Train_all/p1000/'
class_n='Aerial_whales_31cm'
class_n='Water_Training_31cm'
opt.im_fold='/gpfs/projects/LynchGroup/Penguin_workstation/Penguin_Code/Whales/All_aerial_training/all/train/' + class_n 
opt.split = '/gpfs/projects/LynchGroup/Penguin_workstation/Penguin_Code/Whales/fulldata/split/'+class_n
opt.new_fold = '/gpfs/projects/LynchGroup/Penguin_workstation/Penguin_Code/Whales/fulldata/fold_'

sdmkdir(opt.split)
def gen_folds(NIM,n):
    idx = np.random.permutation(NIM)
    return [idx[i::n] for i in range(n)]
def choose_n_k(n,k):
    idx = np.random.permutation(n)
    return idx[0:k],idx[k:]

imlist = []
test_ratio = 0.25

for root,_,fnames in sorted(os.walk(opt.im_fold)):
    for fname in fnames:
        if fname.endswith('.png'):
            imlist.append(fname)
            
nim = len(imlist)
print("all: ",nim)
print(imlist)
list_to_file(opt.split+'/all.txt',imlist)
nim_test = int(float(nim)*test_ratio)
print(nim_test)
nim_train = nim - nim_test
n_folds=4
for split_idx in range(1):
    test_list= []
    train_list = []

    folds = gen_folds(nim,n_folds)
    print(folds)
    for f in range(n_folds):
        test_list = [imlist[i] for i in folds[f]] 
        train_list = [i for i in imlist if  i not in test_list]
        sdmkdir(opt.new_fold+str(f)+'/train/'+class_n)
        sdmkdir(opt.new_fold+str(f)+'/val/'+class_n)
        for file in train_list:
            copyfile(opt.im_fold+'/'+file,opt.new_fold+str(f)+'/train/'+class_n+'/'+file)
        for file in test_list:
            copyfile(opt.im_fold+'/'+file,opt.new_fold+str(f)+'/val/'+class_n+'/'+file)
        list_to_file(opt.split+"/val_"+str(f)+'.txt',test_list)
        list_to_file(opt.split+"/train_"+str(f)+'.txt',train_list)
    #for fold_i in range(1,n_folds):
    #    trainidx= []
    #    for i in range(1,fold_i+1):
    #        for item in folds[i]:
    #            trainidx.append(item)
    #    tlist = [imlist[i] for i in trainidx]
    #    print("train fold %d len:", fold_i,len(trainidx))
    #    file_n = opt.split+"/train_"+str(split_idx)+'_'+str(fold_i)+'.txt'
    #    list_to_file(file_n,tlist)

            

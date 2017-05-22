import scipy
import scipy.misc
import os
from PIL import Image
import numpy as np
import scipy as sp
import skimage 
import random

class Dataset:
    def __init__(self,path,input_shape=(1024,1024),gt_shape=(1024,1024)):
        self.input_shape=input_shape
        self.gt_shape=gt_shape
        self.path=path
        if not os.path.exists(path):
            print("path is wrong!")
        else:
            for file in os.listdir(path):
                if file.endswith('.jpeg'):
                    self.files.append(file)
                    self.max_idx+=1
    def read_image(self,path,size,option):   # this function reads image as float64
        if option==64:
            image =(scipy.misc.imread(path)).astype(float)
            ret=scipy.misc.imresize(image,size).astype(float)
        elif option==8:
            image = Image.open(path)
            ret = np.asarray( image.resize(size,Image.ANTIALIAS))
        return ret
    def change_format(self,image):
       return ((image*255)/np.max(image)).astype('uint8')
    def get_batch_inputs(self,path,idx):
        path = self.path+self.files[idx]
        i_image = self.read_image(path,self.input_shape,64)
        poissonNoise = np.random.poisson(50,i_image.shape).astype(float) # noise model
        i_image=i_image+poissonNoise # add noise to image 
        g_image = self.read_image(path,self.gt_shape,64)
        return i_image,g_image

    def next_batch(self,batch_size):
        in_image=[]
        gt_image=[]
        cur_idx=self.cur_idx
        for i in range(batch_size):
            idx = cur_idx+i
            if not idx <self.max_idx:
                idx=0
            path = self.path+self.files[idx]
            i_image,g_image = self.get_batch_inputs(path,idx)
   # for debug
   #        formatted = self.change_format(i_image)
   #        ret = Image.fromarray(formatted)
   #        ret.show()
   #        formatted = self.change_format(g_image)
   #        ret = Image.fromarray(formatted)
   #        ret.show()
            in_image.append(i_image)
            gt_image.append(g_image)
        in_image = np.array(in_image)
        gt_image=np.array(gt_image)
        self.cur_idx=(cur_idx+batch_size)%self.max_idx # update for next batching
        return in_image,gt_image        
    def random_batch(self,batch_size):
        in_image=[]
        gt_image=[]
        cur_idx = random.random(0,max_idx)
        for i in range(batch_size):
            if not cur_idx < self.max_idx:
                cur_idx=0
            path =self.path+self.files[cur_idx]
            i_image,g_image=self.get_batch_inputs(path,cur_idx)
            in_image.append(i_image)
            gt_image.append(g_image)
            cur_idx+=1
        in_image = np.array(in_image)
        gt_image=np.array(gt_image)
        self.cur_idx=(cur_idx+batch_size)%self.max_idx # update for next batching 
        return in_image,gt_image        


    input_shape=None
    gt_shape=None
    cur_idx=0
    max_idx=0
    files=[] 


def main():
    a=Dataset("images/",(512,512),(1024,1024))
    b,c=a.next_batch(2)
    print(b.shape)
    print(c.shape)
    
main()

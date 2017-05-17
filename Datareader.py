import scipy
import scipy.misc
import os
#import PIL
from PIL import Image
import numpy as np
import scipy as sp
import skimage 
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

    def next_batch(self,batch_size):
        in_image=[]
        gt_image=[]
        cur_idx=self.cur_idx
        for i in range(cur_idx,cur_idx+batch_size):
            if not i <self.max_idx:
                break
            path = self.path+self.files[i]
            input_image = self.read_image(path,self.input_shape,64)
            poissonNoise = np.random.poisson(50,input_image.shape).astype(float) # noise model
            input_image=input_image+poissonNoise # add noise to image 
            g_image = self.read_image(path,self.gt_shape,64)
            formatted = self.change_format(input_image)
            ret = Image.fromarray(formatted)
            ret.show()
            formatted = self.change_format(g_image)
            ret = Image.fromarray(formatted)
            ret.show()
            in_image.append(input_image)
            gt_image.append(g_image)

        in_image = np.array(in_image)
        gt_image=np.array(gt_image)
        self.cur_idx=cur_idx+batch_size
        if not self.cur_idx < self.max_idx:
            self.cur_idx=0
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

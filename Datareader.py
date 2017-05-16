import scipy
import os
#import PIL
from PIL import Image
import numpy as np
import scipy as sp
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
    def next_batch(self,batch_size):
        in_image=[]
        gt_image=[]
        cur_idx=self.cur_idx
        for i in range(cur_idx,cur_idx+batch_size):
            if not i <self.max_idx:
                break
            path = self.path+self.files[i]
            image = Image.open(path)
            input_image = np.asarray( image.resize(self.input_shape,Image.ANTIALIAS))
            resized_image = np.asarray(image.resize(self.gt_shape,Image.ANTIALIAS))
            in_image.append(input_image)
            gt_image.append(resized_image)

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
    b,c=a.next_batch(3)
   # b,c=a.next_batch(5)
    print(b.shape)
    print(c.shape)
    
main()

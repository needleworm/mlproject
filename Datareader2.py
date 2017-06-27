import scipy
import scipy.misc
import os
from PIL import Image
import numpy as np
from Degradation import degrade
import random

class Dataset:

    def __init__(self, path, input_shape=(1024,1024), gt_shape=(1024,1024)):

        self.input_shape = input_shape
        self.gt_shape = gt_shape
        self.path = path
        self.gt_path = path+'GT/'
        self.degrade_path = path+'degrade/'
        self.cur_idx = 0
        self.max_idx = 0
        self.degrade_files = []
        self.gt_files = []
        #print(self.path)
        if not os.path.exists(self.degrade_path):
            print("degrade path is wrong!")
        else:
            for filename in os.listdir(self.degrade_path):

                if filename.endswith('.jpeg') or filename.endswith('.jpg'):
                    self.degrade_files.append(filename)
                    self.max_idx += 1
                    
        if not os.path.exists(self.gt_path):
            print("gt path is wrong!")
        else:
            for filename in os.listdir(self.gt_path):
                if filename.endswith('.jpeg') or filename.endswith('.jpg'):
                    self.gt_files.append(filename)


    def read_image(self, path, size):   # this function reads image as float64
        image = Image.open(path)
        width, height = image.size
        new_width, new_height = size
        image = image.resize((new_width,new_height), Image.LANCZOS).convert('RGB')

        #ret = np.asarray(image, dtype=np.uint8)
        '''
        if width >= new_width and height >= new_height:
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = (width + new_width) // 2
            bottom = (height + new_height) // 2
            ret = np.array(image.crop((left, top, right, bottom)), dtype=np.uint8)
        elif width >= new_width:
   
            if height < new_height:
                left = (width - new_width) // 2
                right = (width + new_width) // 2
                top = (height - new_height) // 2
                bottom = height + (new_height-height) // 2
                image = image.crop((left, top, right, bottom))
                padding = np.absolute(height-new_height)// 2
                ret = np.array(image, dtype=np.uint8)
                ret[:,0:padding,:] = ret[:,(padding-1):padding, :]
                ret[:,(new_height-padding):new_height,:] = ret[:,(heigth+padding-1):(heigth+padding), :]  
            else:
                left = (width - new_width) // 2
                right = (width + new_width) // 2
                top = 0
                bottom = new_height
                image = image.crop((left, top, right, bottom))
                ret = np.array(image, dtype=np.uint8)
        elif height >= new_height:

            if weight < new_weight:
                left = (width - new_width) // 2
                right = width + (new_width-width) // 2
                top = (height - new_height) // 2            
                bottom = (height + new_height) // 2
                image = image.crop((left, top, right, bottom))
                padding = np.absolute(width-new_width)// 2
                ret = np.array(image, dtype=np.uint8)
                ret[:,:,0:padding] = ret[:,:, padding-1:padding]
                ret[:,:,(new_width-padding):new_width] = ret[:,:,(width+padding-1):width+padding]  
            else:
                left = 0
                right = new_width
                top = (height - new_height) // 2            
                bottom = (height + new_height) // 2
                image = image.crop((left, top, right, bottom))
                ret = np.asarray(image, dtype=np.uint8)            
        else:
                left = (width - new_width) // 2
                right = width + (new_width-width) // 2
                top = (height - new_height) // 2
                bottom = height + (new_height-height) // 2
                image = image.crop((left, top, right, bottom))
                padding = np.absolute(width-new_width)// 2
                ret = np.asarray(image, dtype=np.uint8)
                ret[:,:,0:padding] = ret[:,:, padding-1:padding]
                ret[:,:,(new_width-padding):new_width] = ret[:,:,(width+padding-1):(width+padding)]
                ret[:,0:padding,:] = ret[:,(padding-1):padding, :]
                ret[:,(new_height-padding):new_height,:] = ret[:,(heigth+padding-1):(heigth+padding), :]
            #padding
        '''    
        return np.array(image, dtype=np.uint8)

    def get_batch_inputs(self, idx):

        i_image = self.read_image(self.degrade_path+self.degrade_files[idx], self.input_shape)
        g_image = self.read_image(self.gt_path+self.gt_files[idx], self.gt_shape)
        return i_image.astype(np.float32), g_image.astype(np.float32)

        
    def next_batch(self, batch_size):
        in_image=[]
        gt_image=[]
        cur_idx=self.cur_idx
        for i in range(batch_size):
            i_image, g_image = self.get_batch_inputs(cur_idx)
            in_image.append(i_image)
            gt_image.append(g_image)
            cur_idx = (cur_idx+1)%self.max_idx
        in_image= np.array(in_image)
        gt_image=np.array(gt_image)
        self.cur_idx=cur_idx # update for next batching
        return in_image,gt_image

    def random_batch(self,batch_size):
        in_image=[]
        gt_image=[]
        cur_idx = random.randint(0, self.max_idx-1)
        for i in range(batch_size):
            i_image, g_image = self.get_batch_inputs(cur_idx)
            in_image.append(i_image)
            gt_image.append(g_image)
            cur_idx = (cur_idx+1)%self.max_idx
        in_image = np.array(in_image)
        gt_image = np.array(gt_image)
        self.cur_idx = cur_idx # update for next batching
        return in_image, gt_image




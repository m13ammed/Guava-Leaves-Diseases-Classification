
import os
import copy
from keras.preprocessing.image import load_img
from imgaug import augmenters as iaa
from tensorflow.keras.utils import Sequence
from scipy import ndimage as ndi 
from PIL import ImageOps
from itertools import cycle, islice
import cv2 as cv
import numpy as np


class Dataset(Sequence):
      
  def __init__(self,path,to_fit=True,batch_size=16,AE=True, is_val=False,input_shape=252, Eff=True):
    
    self.is_val=is_val
    self.idxList=[]    
    self.input_shape=input_shape
    self.images,self.Class= self.ImagesLoadFromPath(path,self.input_shape)
    self.to_fit=to_fit
    self.batch_size=batch_size
    self.AE=AE
    self.numImages= len(self.images)
    if self.is_val:
      self.idxList=[i for i in range(0,self.numImages)]
    
    self.Eff = Eff 
    self.maxAray=[]
  def __getitem__(self, index):


    start=index*self.batch_size
    ending= index*self.batch_size+self.batch_size
    tempList=self.idxList[start:ending]
    image=[self.images[i] for i in tempList]
    Class=[self.Class[i] for i in tempList]


    if self.is_val:
      image_seg=  self.Segmentation(image)
      imageDenom= np.array(image_seg)
      image=self.Norming(imageDenom)
    else:
      if self.AE:
        image_seg= image
      else:
        image_seg=  self.Segmentation(image)
        
      imageDenom=  self.DataAugemntation(image_seg)
      image=self.Norming(imageDenom)
      
    if self.to_fit:
      if self.AE:
        return image, {'Dec':image,}
      else:
        #print(np.asarray(Class).shape)
        return image, {'FC':np.asarray(Class)}
    else:
      return image
    
  def DataAugemntation(self, images): #input should be a list of numpy arrays (list of images)
    Auge= iaa.RandAugment(n=(1,5),m=(3,15))
    Auge= iaa.RandAugment(n=(1,5),m=(10))
    out=Auge(images=images)
    return np.array(out)

  def Segmentation(self, someImages):  
    img = someImages
    for i in range(len(img)):  
      img[i] = cv.cvtColor(img[i], cv.COLOR_BGR2RGB)
    High = (125, 200, 195)
    Low = (17, 15, 50)
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2,2))
    result = []
    for i in range(0,len(img)):
      hsv_leaf = cv.cvtColor(img[i], cv.COLOR_RGB2HSV)
      mask = cv.inRange(hsv_leaf, Low, High)
      fill_up = np.array(ndi.binary_fill_holes(mask),np.uint8)*255
      fill_up = cv.dilate(fill_up, element, iterations = 1)
      fill_up = cv.erode(fill_up, element, iterations = 1)
      #image_list = [j for j in cv.bitwise_and(img[i], img[i], mask=fill_up)]
      IImage=cv.bitwise_and(img[i], img[i], mask=fill_up)
      result.append(IImage)
      #result [i] = cv.bitwise_and(img[i],img[i], mask = fill_up)
      #

    return result

  def on_epoch_end(self):
    
    np.random.seed(20)
    np.random.shuffle(self.idxList)
      #Shuffle list magic goes here?
    print("shuffle done!")

  def ImagesLoadFromPath(self,path,desired_size=500):
    ImagesArray=[]
    Classes= []
    NumberOfClasses= len(os.listdir(path))
    dummyVector=[0 for i in range(NumberOfClasses)]
    folderss=os.listdir(path)
    folderss.sort()
    if not self.is_val:
      min_img= 984
    else: 
      min_img= 0
    counterIdx=0
    for i,folder in enumerate(folderss):
      
      arrayidx=[]
      dummyVectorUp=copy.deepcopy(dummyVector)
      dummyVectorUp[i]=1
      dummyVectorUp=np.asarray(dummyVectorUp)
      folderImage= os.path.join(path,folder)
      print(folder)
      for imageName in os.listdir(folderImage):
        imagePath= os.path.join(folderImage,imageName)
        im=load_img(imagePath)
        if im.size == (desired_size,desired_size):
          ImagesArray.append(np.asarray(im))
        else: 
          new_im=self.resize_with_padding(im,desired_size)
          old_size = im.size
          ImagesArray.append(np.asarray(new_im))
        Classes.append(dummyVectorUp)

        arrayidx.append(counterIdx)
        counterIdx+=1
      if len(arrayidx)<min_img:
        output = list(islice(cycle(arrayidx), min_img))
      else:
        output=arrayidx
      self.idxList.extend(output)
    return np.array(ImagesArray), np.array(Classes)

  def __len__(self):

    if self.numImages % self.batch_size:
      return int(len(self.idxList) / self.batch_size) 
    else:
      return int(len(self.idxList)  / self.batch_size)

  def resize_with_padding(self,img, expected_size):
      img.thumbnail((expected_size, expected_size))
      # print(img.size)
      delta_width = expected_size - img.size[0]
      delta_height = expected_size - img.size[1]
      pad_width = delta_width // 2
      pad_height = delta_height // 2
      padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
      return ImageOps.expand(img, padding)

  def Norming(self, img): 
    #normalized = img.astype('float64')/255.0
    if self.Eff:
    	normalized = img.astype('float64')
    else:
      normalized = img.astype('float64')/255.0
    #for i in range(len(img)):
     #   normalized[i,:,:,:] = img[i,:,:,:].astype('float64')/255.0
    #self.maxAray.append(maxx)
    return normalized

# -*- coding: utf-8 -*-

from read_data import read_file
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
import random
from read_data import *

#建立一个用于存储和格式化读取训练数据的类
class DataSet(object):
    def __init__(self,path1,path2):
        self.num_classes = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.y_train = None
        self.y_test = None
        self.extract_data(path1,path2)
        #在这个类初始化的过程中读取path下的训练数据

    def extract_data(self,path1,path2):
        #根据指定路径读取出图片、标签和类别数
        imgs,labels,counter = process_img(path1,path2)

        print("label out")
        print(labels)

        #将数据集打乱随机分组    
        
        
        X_train,X_test,self.y_train,self.y_test = train_test_split(imgs,labels,test_size=0.1,random_state=random.randint(0, 100))
        print("y_train")
        print(self.y_train)
        print(len(X_train))
        print(X_train[1])
        print("X_test")
        print(len(X_test))
        print(self.y_test)
        print("num of classes")
        print(counter)

        #重新格式化和标准化
        # 本案例是基于thano的，如果基于tensorflow的backend需要进行修改
        # X_train = X_train.reshape(X_train.shape[0], 256, 256, 3)
        # X_test = X_test.reshape(X_test.shape[0], 256, 256,3)
        
        
        # X_train = X_train.astype('float32')/255
        # X_test = X_test.astype('float32')/255
        print(X_train[1])

        #将labels转成 binary class matrices
        Y_train = np_utils.to_categorical(self.y_train, num_classes=counter)
        Y_test = np_utils.to_categorical(self.y_test, num_classes=counter)
        print(Y_train)
        #将格式化后的数据赋值给类的属性上
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.num_classes = counter

    def check(self):
        print('num of dim:', self.X_test.ndim)
        print('shape:', self.X_test.shape)
        print('size:', self.X_test.size)

        print('num of dim:', self.X_train.ndim)
        print('shape:', self.X_train.shape)
        print('size:', self.X_train.size)
        
if __name__ == '__main__':
    datast = DataSet(r"D:/python_file/keras-resnet-master/Data/test/AD",
                                               r"D:/python_file/keras-resnet-master/Data/test/NC")
    path='./dataset'
        
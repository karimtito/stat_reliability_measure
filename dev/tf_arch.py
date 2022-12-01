import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Dense,Flatten
from stat_reliability_measure.dev.tf_utils import dic_in_shape_tf



class CNN_custom_tf(tf.keras.Model):
    def __init__(self, num_classes=10,dataset='mnist'):
        super().__init__()
        self.model_shape=dic_in_shape_tf[dataset]
        self.conv1=Conv2D(filters=32,kernel_size=3,padding="same", input_shape=self.model_shape,
        activation=tf.nn.relu)
     
        self.conv2=Conv2D(filters=32,kernel_size=3, padding="same",strides=2, 
        activation=tf.nn.relu)
    
        self.conv3=Conv2D(filters=64,kernel_size=3,padding="same", 
        activation=tf.nn.relu)
        self.conv4=Conv2D(filters=64,kernel_size=3,padding="same",strides=2,
        activation=tf.nn.relu)
        self.flatten = Flatten()
        self.linear1=Dense(units=100,activation= tf.nn.relu)
        self.linear2=Dense(units=num_classes)


   
    def call(self, x):
        out=self.conv1(x)
        out=self.conv2(out)
        out=self.conv3(out)
        out=self.conv4(out)
        out= self.flatten(out)
        out=self.linear1(out)
        out= self.linear2(out)
        return out


tf_supp_archs={'mnist':{'cnn_custom':CNN_custom_tf}}
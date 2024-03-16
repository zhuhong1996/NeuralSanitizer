# -*- coding: utf-8 -*-

import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
np.random.seed(0)


clean_images = np.zeros(shape=[10,600,28,28,1])
for i in range(10):
    for j in range(600):
        filename = str(j).zfill(3)+'.jpg'
        image = cv2.imread('./test_image/'+str(i)+'/'+filename,0)
        image = image[:,:,np.newaxis]
        image = image/255.0
        image = image - 0.5
        clean_images[i,j] = image
        


reverse_image_indexs = np.zeros(shape=[10],dtype=np.int32)
def get_reverse_trigger_data(modelname,index,source,batch_size):
    
    global reverse_image_indexs
    
    path = './critical_features/'+modelname
    mask_image = np.float32(cv2.imread(path+'/mask_'+str(index)+'.png',0))
    mask_image = mask_image/255.0
    mask_image = mask_image[:,:,np.newaxis]
    trigger_image = np.float32(cv2.imread(path+'/trigger_'+str(index)+'.png',0))
    trigger_image = trigger_image/255.0
    trigger_image = trigger_image[:,:,np.newaxis]
    trigger_image = trigger_image - 0.5
     
    images = np.zeros(shape=[batch_size,28,28,1])
    
    np.random.shuffle(source)
    source_length = len(source)
    index = 0
    
    for i in range(batch_size):
        
        label = source[index]
        index = (index+1)%source_length
        
        image_index = reverse_image_indexs[label]
        reverse_image_indexs[label] = (image_index+1)%600
        image = clean_images[label,image_index].copy()
        image = image*(1-mask_image)+trigger_image*mask_image
        images[i] = image
        
    return images 


def weight_variable(init):
    initial = tf.constant(init, shape=init.shape)
    return tf.Variable(initial,name="weight")


def bias_variable(init):
    initial = tf.constant(init, shape=init.shape)
    return tf.Variable(initial,name="bias")

# Functions for convolution and pooling functions
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pooling_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def define_model(path,x,keep_prob):

    reader = tf.train.NewCheckpointReader(path)

    cw1 = reader.get_tensor("conv1/weight")
    cb1 = reader.get_tensor("conv1/bias")
    cw2 = reader.get_tensor("conv2/weight")
    cb2 = reader.get_tensor("conv2/bias")
    cw3 = reader.get_tensor("conv3/weight")
    cb3 = reader.get_tensor("conv3/bias")
    cw4 = reader.get_tensor("conv4/weight")
    cb4 = reader.get_tensor("conv4/bias")    
    fw1 = reader.get_tensor("full/weight")
    fb1 = reader.get_tensor("full/bias")
    fw2 = reader.get_tensor("class/weight")
    fb2 = reader.get_tensor("class/bias")
       
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable(cw1)
        b_conv1 = bias_variable(cb1)
        x_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable(cw2)
        b_conv2 = bias_variable(cb2)
        x_conv2 = tf.nn.relu(conv2d(x_conv1, W_conv2) + b_conv2)
        x_pool2 = max_pooling_2x2(x_conv2)
        
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable(cw3)
        b_conv3 = bias_variable(cb3)
        x_conv3 = tf.nn.relu(conv2d(x_pool2, W_conv3) + b_conv3)
        
    with tf.name_scope('conv4'):
        W_conv4 = weight_variable(cw4)
        b_conv4 = bias_variable(cb4)
        x_conv4 = tf.nn.relu(conv2d(x_conv3, W_conv4) + b_conv4)
        x_pool4 = max_pooling_2x2(x_conv4)
        x_flat = tf.reshape(x_pool4, [-1, 7*7*64])
        
    # Dense fully connected layer
    with tf.name_scope('full'):
        W_fc1 = weight_variable(fw1) 
        b_fc1 = bias_variable(fb1)
        x_fc1 = tf.nn.relu(tf.matmul(x_flat, W_fc1) + b_fc1)
    # Regularization with dropout
        x_fc1_drop = tf.nn.dropout(x_fc1, keep_prob)

    # Classification layer
    with tf.name_scope('class'):
        W_fc2 = weight_variable(fw2)
        b_fc2 = bias_variable(fb2)
        y_conv = tf.matmul(x_fc1_drop, W_fc2) + b_fc2
    # Probabilities - output from model (not the same as logits)
        y = tf.nn.softmax(y_conv)
    
    return y_conv,y
        

def transfer_once(isfilt,path,target_modelname):
    
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    
    # Create placeholders nodes for images and label inputs
    x = tf.placeholder(tf.float32, shape=[None, 28,28,1], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    
    y_conv,y = define_model("./model/"+target_modelname+'/model',x,keep_prob)
  
    # Setup to test accuracy of model
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.add_to_collection("accuracy", accuracy)
    
    # Initilize all global variables
    sess.run(tf.global_variables_initializer())
    
    isfilt2 = isfilt.copy()
    
    for i in range(10):
        
        if isfilt2[i]:
            continue
        
        source = []
        for j in range(10):
            if j!=i:
                source.append(j)
        
        acc = 0
        labels = np.zeros(shape=[50,10])
        labels[:,i]=1 
        for j in range(4):
            images = get_reverse_trigger_data(path,i,source,50)
            acc2 = sess.run(accuracy,{x:images, y_:labels, keep_prob:1.0})
            acc += acc2
        acc = acc/4.0
        
        if acc>=0.2:
            isfilt2[i]=1
        
    sess.close()
    
    return isfilt2

def detect(modelname):
    
    isfilt = np.zeros(shape=[10])
    asrs = np.load("./potential_triggers/"+modelname+'/asrs.npy')
    for i in range(10):
        if asrs[i]<0.6:
            isfilt[i]=1
    
    isfilt = transfer_once(isfilt,modelname,modelname+'/pnnir_1') 
    isfilt = transfer_once(isfilt,modelname,modelname+'/pnnir_2') 
    isfilt = transfer_once(isfilt,modelname,modelname+'/pnnir_3') 
    isfilt = transfer_once(isfilt,modelname,modelname+'/pnnir_12') 
    isfilt = transfer_once(isfilt,modelname,modelname+'/pnnir_23') 
    isfilt = transfer_once(isfilt,modelname,modelname+'/pnnir_13')
    
    backdoor_list = []
    clean_list = []
    for i in range(10):
        if isfilt[i]:
            clean_list.append(i)
        else:
            backdoor_list.append(i)
            
    dirs = "./detection_results/"
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    
    file = open(dirs+modelname+".txt", "w+")
    
    print(modelname)
    file.write(modelname+'\n')
    
    print('infected label list: ')
    file.write('infected label list: \n')
    for i in range(len(backdoor_list)):
        print(backdoor_list[i], end=' ')
        file.write(str(backdoor_list[i])+' ')
    print('')
    file.write('\n')
    
    print('clean label list: ')
    file.write('clean label list: \n')
    for i in range(len(clean_list)):
        print(clean_list[i], end=' ')
        file.write(str(clean_list[i])+' ')
    print('')
    file.write('\n')
    file.close()
    
    
    
if __name__ ==  '__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    modelname = 'patch1'
    detect(modelname)
    
    modelname = 'patch2'
    detect(modelname)
    
    modelname = 'patch3'
    detect(modelname)
    
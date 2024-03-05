# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import random
import tensorflow as tf
np.random.seed(0)
random.seed(0)

def get_namelist():
    with open('names.txt', 'r') as f:
        namelist = f.read().splitlines()
    return namelist


def get_reverse_trigger_data(modelname,namelist,index,source,batch_size):
    
    path = './potential_triggers_feature_space/'+modelname
    trigger = np.load(path+'/'+str(index)+'.npy')
    
    averageImage3 = np.zeros(shape=[224,224,3])
    averageImage3[:,:,2] = 129.1863
    averageImage3[:,:,1] = 104.7624
    averageImage3[:,:,0] = 93.5940

    images = np.zeros(shape=[batch_size,224,224,3])
    
    np.random.shuffle(source)
    source_length = len(source)
    index = 0
    
    for i in range(batch_size):
        
        label = source[index]
        index = (index+1)%source_length
        
        filename = str(random.randint(0,79)).zfill(3)+'.jpg'
        image = cv2.imread(path+namelist[label]+'/'+filename)
        image = cv2.resize(image,(224,224))
        image = np.reshape(image,(-1,3))
        image = np.matmul(image,trigger)
        image = np.reshape(image,(224,224,3))
        image = np.float32(image) - averageImage3
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

def define_model(model_path,x,keep_prob):
     
    reader = tf.train.NewCheckpointReader(model_path)

    w_c1_1 = reader.get_tensor("conv1_1/weight")
    b_c1_1 = reader.get_tensor("conv1_1/bias")  
    w_c1_2 = reader.get_tensor("conv1_2/weight")
    b_c1_2 = reader.get_tensor("conv1_2/bias") 
    
    w_c2_1 = reader.get_tensor("conv2_1/weight")
    b_c2_1 = reader.get_tensor("conv2_1/bias") 
    w_c2_2 = reader.get_tensor("conv2_2/weight")
    b_c2_2 = reader.get_tensor("conv2_2/bias") 
    
    w_c3_1 = reader.get_tensor("conv3_1/weight")
    b_c3_1 = reader.get_tensor("conv3_1/bias") 
    w_c3_2 = reader.get_tensor("conv3_2/weight")
    b_c3_2 = reader.get_tensor("conv3_2/bias") 
    w_c3_3 = reader.get_tensor("conv3_3/weight")
    b_c3_3 = reader.get_tensor("conv3_3/bias") 
    
    w_c4_1 = reader.get_tensor("conv4_1/weight")
    b_c4_1 = reader.get_tensor("conv4_1/bias") 
    w_c4_2 = reader.get_tensor("conv4_2/weight")
    b_c4_2 = reader.get_tensor("conv4_2/bias") 
    w_c4_3 = reader.get_tensor("conv4_3/weight")
    b_c4_3 = reader.get_tensor("conv4_3/bias") 
    
    w_c5_1 = reader.get_tensor("conv5_1/weight")
    b_c5_1 = reader.get_tensor("conv5_1/bias") 
    w_c5_2 = reader.get_tensor("conv5_2/weight")
    b_c5_2 = reader.get_tensor("conv5_2/bias") 
    w_c5_3 = reader.get_tensor("conv5_3/weight")
    b_c5_3 = reader.get_tensor("conv5_3/bias") 
    
    w_f6 = reader.get_tensor("fc6/weight")
    b_f6 = reader.get_tensor("fc6/bias")
    w_f7 = reader.get_tensor("fc7/weight")
    b_f7 = reader.get_tensor("fc7/bias")
    w_f8 = reader.get_tensor("fc8/weight")
    b_f8 = reader.get_tensor("fc8/bias")
    
    with tf.name_scope('conv1_1'):
        W_conv1_1 = weight_variable(w_c1_1)
        b_conv1_1 = bias_variable(b_c1_1)
        x_conv1_1 = tf.nn.relu(conv2d(x, W_conv1_1) + b_conv1_1)
    
    with tf.name_scope('conv1_2'):
        W_conv1_2 = weight_variable(w_c1_2)
        b_conv1_2 = bias_variable(b_c1_2)
        x_conv1_2 = tf.nn.relu(conv2d(x_conv1_1, W_conv1_2) + b_conv1_2)
        
    with tf.name_scope('pool1'):       
        x_pool1 = max_pooling_2x2(x_conv1_2)
        
    with tf.name_scope('conv2_1'):
        W_conv2_1 = weight_variable(w_c2_1)
        b_conv2_1 = bias_variable(b_c2_1)
        x_conv2_1 = tf.nn.relu(conv2d(x_pool1, W_conv2_1) + b_conv2_1)
        
    with tf.name_scope('conv2_2'):
        W_conv2_2 = weight_variable(w_c2_2)
        b_conv2_2 = bias_variable(b_c2_2)
        x_conv2_2 = tf.nn.relu(conv2d(x_conv2_1, W_conv2_2) + b_conv2_2)

    with tf.name_scope('pool2'):       
        x_pool2 = max_pooling_2x2(x_conv2_2)
        
    with tf.name_scope('conv3_1'):
        W_conv3_1 = weight_variable(w_c3_1)
        b_conv3_1 = bias_variable(b_c3_1)
        x_conv3_1 = tf.nn.relu(conv2d(x_pool2, W_conv3_1) + b_conv3_1)
        
    with tf.name_scope('conv3_2'):
        W_conv3_2 = weight_variable(w_c3_2)
        b_conv3_2 = bias_variable(b_c3_2)
        x_conv3_2 = tf.nn.relu(conv2d(x_conv3_1, W_conv3_2) + b_conv3_2)
        
    with tf.name_scope('conv3_3'):
        W_conv3_3 = weight_variable(w_c3_3)
        b_conv3_3 = bias_variable(b_c3_3)
        x_conv3_3 = tf.nn.relu(conv2d(x_conv3_2, W_conv3_3) + b_conv3_3)

    with tf.name_scope('pool3'):       
        x_pool3 = max_pooling_2x2(x_conv3_3)
        
    with tf.name_scope('conv4_1'):
        W_conv4_1 = weight_variable(w_c4_1)
        b_conv4_1 = bias_variable(b_c4_1)
        x_conv4_1 = tf.nn.relu(conv2d(x_pool3, W_conv4_1) + b_conv4_1)
        
    with tf.name_scope('conv4_2'):
        W_conv4_2 = weight_variable(w_c4_2)
        b_conv4_2 = bias_variable(b_c4_2)
        x_conv4_2 = tf.nn.relu(conv2d(x_conv4_1, W_conv4_2) + b_conv4_2)
        
    with tf.name_scope('conv4_3'):
        W_conv4_3 = weight_variable(w_c4_3)
        b_conv4_3 = bias_variable(b_c4_3)
        x_conv4_3 = tf.nn.relu(conv2d(x_conv4_2, W_conv4_3) + b_conv4_3)

    with tf.name_scope('pool4'):       
        x_pool4 = max_pooling_2x2(x_conv4_3)
        
    with tf.name_scope('conv5_1'):
        W_conv5_1 = weight_variable(w_c5_1)
        b_conv5_1 = bias_variable(b_c5_1)
        x_conv5_1 = tf.nn.relu(conv2d(x_pool4, W_conv5_1) + b_conv5_1)
        
    with tf.name_scope('conv5_2'):
        W_conv5_2 = weight_variable(w_c5_2)
        b_conv5_2 = bias_variable(b_c5_2)
        x_conv5_2 = tf.nn.relu(conv2d(x_conv5_1, W_conv5_2) + b_conv5_2)
        
    with tf.name_scope('conv5_3'):
        W_conv5_3 = weight_variable(w_c5_3)
        b_conv5_3 = bias_variable(b_c5_3)
        x_conv5_3 = tf.nn.relu(conv2d(x_conv5_2, W_conv5_3) + b_conv5_3)

    with tf.name_scope('pool5'):       
        x_pool5 = max_pooling_2x2(x_conv5_3)

    with tf.name_scope('x_flat'):          
        x_flat = tf.reshape(x_pool5, [-1, 25088])
        
    # Dense fully connected layer
    with tf.name_scope('fc6'):
        W_fc6 = weight_variable(w_f6) 
        b_fc6 = bias_variable(b_f6)
        x_fc6 = tf.nn.relu(tf.matmul(x_flat, W_fc6) + b_fc6)
    # Regularization with dropout
        x_fc6_drop = tf.nn.dropout(x_fc6, keep_prob)
        
    with tf.name_scope('fc7'):
        W_fc7 = weight_variable(w_f7) 
        b_fc7 = bias_variable(b_f7)
        x_fc7 = tf.nn.relu(tf.matmul(x_fc6_drop, W_fc7) + b_fc7)
    # Regularization with dropout
        x_fc7_drop = tf.nn.dropout(x_fc7, keep_prob)

    # Classification layer
    with tf.name_scope('fc8'):
        W_fc8 = weight_variable(w_f8)
        b_fc8 = bias_variable(b_f8)
        y_conv = tf.matmul(x_fc7_drop, W_fc8) + b_fc8
        y = tf.nn.softmax(y_conv)
    
    return y_conv,y


def transfer_once(isfilt,path,target_modelname,source):
    
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    
    # Create placeholders nodes for images and label inputs
    x = tf.placeholder(tf.float32, shape=[None, 224,224,3], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, 2622], name="y_")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    
    y_conv,y = define_model("./model/"+target_modelname+'/model',x,keep_prob)
  
    # Setup to test accuracy of model
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.add_to_collection("accuracy", accuracy)
    
    # Initilize all global variables
    sess.run(tf.global_variables_initializer())
    
    isfilt2 = isfilt.copy()
    
    namelist = get_namelist()
    
    for i in range(2622):
        
        if isfilt2[i]:
            continue
        
        source = []
        for j in range(2622):
            if j!=i:
                source.append(j)
        
        acc = 0
        labels = np.zeros(shape=[50,2622])
        labels[:,i]=1 
        for j in range(4):
            images = get_reverse_trigger_data(path,namelist,i,source,50)
            acc2 = sess.run(accuracy,{x:images, y_:labels, keep_prob:1.0})
            acc += acc2
        acc = acc/4.0
        
        if acc>=0.2:
            isfilt2[i]=1
        
    sess.close()
    
    return isfilt2


def detect(modelname):
    
    isfilt = np.zeros(shape=[2622])
    asrs = np.load("./potential_triggers_feature_space/"+modelname+'/asrs.npy')
    for i in range(2622):
        if asrs[i]<0.6:
            isfilt[i]=1
    
    isfilt = transfer_once(isfilt,modelname,modelname+'/pnnir_1') 
    isfilt = transfer_once(isfilt,modelname,modelname+'/pnnir_2') 
    isfilt = transfer_once(isfilt,modelname,modelname+'/pnnir_3') 
    isfilt = transfer_once(isfilt,modelname,modelname+'/pnnir_12') 
    isfilt = transfer_once(isfilt,modelname,modelname+'/pnnir_23') 
    isfilt = transfer_once(isfilt,modelname,modelname+'/pnnir_123')
    
    backdoor_list = []
    clean_list = []
    for i in range(2622):
        if isfilt[i]:
            clean_list.append(i)
        else:
            backdoor_list.append(i)
            
    dirs = "./detection_results_feature_space/"
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
    
    
    modelname = 'feature1'
    detect(modelname)
    
    modelname = 'feature2'
    detect(modelname)
    
    modelname = 'feature3'
    detect(modelname)
    

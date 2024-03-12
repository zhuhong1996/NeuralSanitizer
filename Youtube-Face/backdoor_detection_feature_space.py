# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import tensorflow as tf
np.random.seed(0)

def get_namelist():
    with open('names.txt','r') as f:
        namelist = f.read().splitlines()
    return namelist

reverse_image_indexs = np.zeros(shape=[1595],dtype=np.int32)
def get_reverse_trigger_data(modelname,namelist,index,source,batch_size):
    
    global reverse_image_indexs
    
    path = './potential_triggers_feature_space/'+modelname
    trigger = np.load(path+'/'+str(index)+'.npy')

    images = np.zeros(shape=[batch_size,224,224,3])
    
    np.random.shuffle(source)
    source_length = len(source)
    index = 0
    
    for i in range(batch_size):
        
        label = source[index]
        index = (index+1)%source_length
        
        image_index = reverse_image_indexs[label]
        reverse_image_indexs[label] = (image_index+1)%80
        filename = str(image_index).zfill(3)+'.jpg'
        image = cv2.imread('./test_image/'+namelist[label]+'/'+filename)
        image = image[:,:,::-1]
        image = image/255.0
        image = image - 0.5
        images[i] = image
        
    images = np.reshape(images,(-1,3))
    images = np.matmul(images,trigger)
    images = np.reshape(images,(-1,224,224,3))
        
    return images 


def weight_variable(init):
    initial = tf.constant(init, shape=init.shape)
    return tf.Variable(initial,name="weight")


def bias_variable(init):
    initial = tf.constant(init, shape=init.shape)
    return tf.Variable(initial,name="bias")

# Functions for convolution and pooling functions
def conv2d(x,W,stride):
    return tf.nn.conv2d(x, W, strides=[1,stride,stride,1], padding='SAME')

def max_pooling_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def define_model(path,x,keep_prob):

    reader = tf.train.NewCheckpointReader(path)

    w_c1 = reader.get_tensor("conv1/weight")
    b_c1 = reader.get_tensor("conv1/bias")  
    w_c2 = reader.get_tensor("conv2/weight")
    b_c2 = reader.get_tensor("conv2/bias")   
    w_c3 = reader.get_tensor("conv3/weight")
    b_c3 = reader.get_tensor("conv3/bias")   
     
    w_a1 = reader.get_tensor("add/weight")
    b_a1 = reader.get_tensor("add/bias")
    w_a2 = reader.get_tensor("add/weight_1")
    b_a2 = reader.get_tensor("add/bias_1")
    w_a3 = reader.get_tensor("add/weight_2")
    b_a3 = reader.get_tensor("add/bias_2")
    
    w_f = reader.get_tensor("class/weight")
    b_f = reader.get_tensor("class/bias")
    
    x_input  = x
    
    # Conv layer 1 - 32x3x3
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable(w_c1)
        b_conv1 = bias_variable(b_c1)
        x_conv1 = tf.nn.relu(conv2d(x_input,W_conv1,2) + b_conv1)
          
    with tf.name_scope('pool1'):       
        x_pool1 = max_pooling_2x2(x_conv1)
        
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable(w_c2)
        b_conv2 = bias_variable(b_c2)
        x_conv2 = tf.nn.relu(conv2d(x_pool1,W_conv2,2) + b_conv2)

    with tf.name_scope('pool2'):       
        x_pool2 = max_pooling_2x2(x_conv2)
        
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable(w_c3)
        b_conv3 = bias_variable(b_c3)
        x_conv3 = tf.nn.relu(conv2d(x_pool2, W_conv3,2) + b_conv3)

    with tf.name_scope('pool3'):       
        x_pool3 = max_pooling_2x2(x_conv3)
        
    # Dense fully connected layer
    with tf.name_scope('add'):
        x_flat_fc1 = tf.reshape(x_pool3, [-1, 960])
        W_fc1 = weight_variable(w_a1) 
        b_fc1 = bias_variable(b_a1)
        x_fc1 = tf.matmul(x_flat_fc1, W_fc1) + b_fc1
    # Regularization with dropout
        x_fc1_drop = tf.nn.dropout(x_fc1, keep_prob)   
        
        W_conv4 = weight_variable(w_a2)
        b_conv4 = bias_variable(b_a2)
        x_conv4 = tf.nn.relu(conv2d(x_pool3, W_conv4,1) + b_conv4)

        x_flat_fc2 = tf.reshape(x_conv4, [-1, 1280])
        W_fc2 = weight_variable(w_a3) # max pooling reduced image to 8x8
        b_fc2 = bias_variable(b_a3)
        x_fc2 = tf.matmul(x_flat_fc2, W_fc2) + b_fc2
    # Regularization with dropout
        x_fc2_drop = tf.nn.dropout(x_fc2, keep_prob)  

    # Classification layer
    with tf.name_scope('class'):
        x_add = tf.nn.relu(x_fc1_drop + x_fc2_drop)
        W_fc3 = weight_variable(w_f)
        b_fc3 = bias_variable(b_f)
        y_conv = tf.matmul(x_add, W_fc3) + b_fc3
        y = tf.nn.softmax(y_conv)
    
    return y_conv,y


def transfer_once(isfilt,path,target_modelname):
    
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    
    # Create placeholders nodes for images and label inputs
    x = tf.placeholder(tf.float32, shape=[None, 224,224,3], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, 1595], name="y_")
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
    
    for i in range(1595):
        
        if isfilt2[i]:
            continue
        
        source = []
        for j in range(1595):
            if j!=i:
                source.append(j)
        
        acc = 0
        labels = np.zeros(shape=[50,1595])
        labels[:,i]=1 
        for j in range(4):
            images = get_reverse_trigger_data(path,namelist,i,source,50)
            acc2 = sess.run(accuracy,{x:images, y_:labels, keep_prob:1.0})
            acc += acc2
        acc = acc/4.0
        
        if acc>=0.6:
            isfilt2[i]=1
        
    sess.close()
    
    return isfilt2


def detect(modelname):
    
    isfilt = np.zeros(shape=[1595])
    asrs = np.load("./potential_triggers_feature_space/"+modelname+'/asrs.npy')
    for i in range(1595):
        if asrs[i]<0.6:
            isfilt[i]=1
    
    isfilt = transfer_once(isfilt,modelname,modelname+'/pnnir_1') 
    isfilt = transfer_once(isfilt,modelname,modelname+'/pnnir_2') 
    isfilt = transfer_once(isfilt,modelname,modelname+'/pnnir_3') 
    isfilt = transfer_once(isfilt,modelname,modelname+'/pnnir_12') 
    isfilt = transfer_once(isfilt,modelname,modelname+'/pnnir_13') 
    isfilt = transfer_once(isfilt,modelname,modelname+'/pnnir_23')
    
    backdoor_list = []
    clean_list = []
    for i in range(1595):
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
    
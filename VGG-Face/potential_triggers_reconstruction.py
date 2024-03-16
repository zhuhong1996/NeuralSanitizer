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

def get_next_clean_batch_source(namelist,batch_size,source):
    
    path = './test_image/'
    
    averageImage3 = np.zeros(shape=[224,224,3])
    averageImage3[:,:,2] = 129.1863
    averageImage3[:,:,1] = 104.7624
    averageImage3[:,:,0] = 93.5940
    
    np.random.shuffle(source)
    source_index=0
    source_length = len(source)

    label_num = 2622
    images = np.zeros(shape=[batch_size,224,224,3])
    labels = np.zeros(shape=[batch_size,label_num])
    
    for i in range(batch_size):
        label = source[source_index]
        source_index = (source_index+1)%source_length
        labels[i][label]=1
        filename = str(random.randint(0,79)).zfill(3)+'.jpg'
        image = cv2.imread(path+namelist[label]+'/'+filename)
        image = cv2.resize(image,(224,224))
        image = np.float32(image) - averageImage3
        images[i] = image
        
    return images,labels

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


def define_model(path,x,keep_prob,mask_normalized,trigger):
    
    reader = tf.train.NewCheckpointReader(path)

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
    
    x_input =  x*(1-mask_normalized)+trigger*mask_normalized
    
    with tf.name_scope('conv1_1'):
        W_conv1_1 = weight_variable(w_c1_1)
        b_conv1_1 = bias_variable(b_c1_1)
        x_conv1_1 = tf.nn.relu(conv2d(x_input, W_conv1_1) + b_conv1_1)
    
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

    
def reverse(modeltrigger):
    
    modelclean = modeltrigger+'/pnnir_123'

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    
    # Create placeholders nodes for images and label inputs
    x = tf.placeholder(tf.float32, shape=[None, 224,224,3], name="x")
    y_t = tf.placeholder(tf.float32, shape=[None, 2622], name="y_t")
    y_c = tf.placeholder(tf.float32, shape=[None, 2622], name="y_c")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    w1 = tf.placeholder(tf.float32, name="w1")
    w2 = tf.placeholder(tf.float32, name="w2")
    w3 = tf.placeholder(tf.float32, name="w3")
    mask_ph = tf.placeholder(tf.float32, shape=[1, 224,224,1], name="mask_ph")
    trigger_ph = tf.placeholder(tf.float32, shape=[1, 224,224,3], name="trigger_ph")
    threshold = tf.placeholder(tf.float32, name="threshold")
    
    averageImage2 = np.zeros(shape=[1,224,224,3])
    averageImage2[:,:,:,2] = 129.1863
    averageImage2[:,:,:,1] = 104.7624
    averageImage2[:,:,:,0] = 93.5940
    averageImage3 = tf.constant(averageImage2,dtype=tf.float32)
    
    init3 = np.random.uniform(127.5,127.5,size=[1,224,224,1])
    initial3 = tf.constant(init3, dtype=tf.float32, shape=init3.shape) 
    mask =  tf.Variable(initial3,name="mask")
    
    mask_normalized = 0.5*(tf.tanh(0.03*(mask-127.5))+1)
    
    init4 = np.random.uniform(0,255,size=[1,224,224,3])-averageImage2
    initial4 = tf.constant(init4, dtype=tf.float32, shape=init4.shape)
    trigger =  tf.Variable(initial4,name="trigger")

    clip_mask=tf.assign(mask,tf.clip_by_value(mask, 0, 255))
    clip_trigger=tf.assign(trigger,tf.clip_by_value(trigger+averageImage3, 0, 255)-averageImage3)
    
    init_mask=tf.assign(mask,mask_ph)
    init_trigger=tf.assign(trigger,trigger_ph)
    
    y_conv_trigger,y_trigger = define_model('./model/'+modeltrigger+'/model',x,keep_prob,mask_normalized,trigger)
    y_conv_clean,y_clean = define_model('./model/'+modelclean+'/model',x,keep_prob,mask_normalized,trigger)
     
    # Setup to test accuracy of model
    correct_prediction_trigger = tf.equal(tf.argmax(y_conv_trigger,1), tf.argmax(y_t,1))
    accuracy_trigger = tf.reduce_mean(tf.cast(correct_prediction_trigger, tf.float32))
    
    namelist = get_namelist()
    
    
    loss1 = w1*tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_t, logits=y_conv_trigger)
    loss2 = w2*tf.reduce_sum(tf.nn.relu(tf.reduce_sum(y_clean,axis=0)-threshold))
    loss3 = w3*(tf.reduce_sum(tf.abs(mask_normalized[:,0:223,0:223,:]-mask_normalized[:,1:224,1:224,:]))\
                +tf.reduce_sum(tf.abs(mask_normalized[:,0:223,1:224,:]-mask_normalized[:,1:224,0:223,:]))\
                +tf.reduce_sum(tf.abs(mask_normalized[:,0:223,:,:]-mask_normalized[:,1:224,:,:]))\
                +tf.reduce_sum(tf.abs(mask_normalized[:,:,1:224,:]-mask_normalized[:,:,0:223,:])))
    
    loss = loss1 + loss2 + loss3
    cross_entropy = tf.reduce_mean(loss)
    train_step = tf.train.AdamOptimizer(5).minimize(cross_entropy,var_list=[trigger,mask])
    
    # Initilize all global variables
    sess.run(tf.global_variables_initializer())
    
    dirs = "./potential_triggers/"+modeltrigger
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    
    asrs = np.zeros(shape=[2622])
    for i in range(2622):
        
        source = []
        for j in range(2622):
            if j!=i:
                source.append(j)
        
        target = i
        
        batch_size = 10
        label_trigger = np.zeros(shape=[batch_size,2622])
        label_trigger[:,target]=1
        
        init3 = np.random.uniform(0,0,size=[1,224,224,1])
        init4 = np.random.uniform(0,255,size=[1,224,224,3])-averageImage2
        sess.run(init_mask,{mask_ph:init3})
        sess.run(init_trigger,{trigger_ph:init4})
        
        w1_value = 1
        w2_value = 1
        w3_value = 0.0001
       
        for j in range(100):
            
            batch_images, batch_labels = get_next_clean_batch_source(namelist,batch_size,source)
                 
            sess.run([train_step,clip_mask,clip_trigger],feed_dict={x:batch_images, y_t:label_trigger, keep_prob:1.0, w1:w1_value,w2:w2_value,w3:w3_value,threshold:1})
            
        mask_image = sess.run(mask_normalized)
        trigger_image = sess.run(trigger)
            
        print(target)

        asr = 0
        y_label = np.zeros(shape=[50,2622])
        y_label[:,i]=1
        for j in range(4):      
            batch_images, batch_labels = get_next_clean_batch_source(namelist,50,source)
            asr += sess.run(accuracy_trigger,{x:batch_images, y_t:y_label, keep_prob:1.0})
        asr/=4.0
        print("trigger attack success rate: ",asr)
        asrs[i]=asr
        
        
        trigger_image = trigger_image + averageImage2
        trigger_image = np.reshape(trigger_image,[224,224,3])  
        trigger_image = np.clip(trigger_image,0,255)
        mask_image = np.reshape(mask_image,[224,224])
        mask_image = mask_image*255
        mask_image = np.clip(mask_image,0,255)
        trigger_path = dirs+"/trigger_"+str(target)+".png"
        mask_path = dirs+"/mask_"+str(target)+".png"
        if not os.path.exists(trigger_path):
            cv2.imwrite(trigger_path,trigger_image)    
        if not os.path.exists(mask_path):
            cv2.imwrite(mask_path,mask_image)
    
    np.save(dirs+'/asrs.npy',asrs)
    
    sess.close()
        
    

if __name__ ==  '__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    
    modeltrigger = 'patch1'
    reverse(modeltrigger)
    modeltrigger = 'patch2'
    reverse(modeltrigger)
    modeltrigger = 'patch3'
    reverse(modeltrigger)

    
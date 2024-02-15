# -*- coding: utf-8 -*-
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
np.random.seed(0)

clean_images = np.zeros(shape=[43,225,32,32,3])
for i in range(43):
    for j in range(225):
        filename = str(j).zfill(4)+'.jpg'
        image = cv2.imread('./test_image/'+str(i).zfill(2)+'/'+filename)
        image = image[:,:,::-1]
        image = image/255.0
        image = image - 0.5
        clean_images[i,j] = image


clean_image_indexs = np.zeros(shape=[43],dtype=np.int32)
def get_next_clean_batch_source(batch_size,source):
    
    global clean_image_indexs
    
    np.random.shuffle(source)
    source_index=0
    source_length = len(source)
    
    images = np.zeros(shape=[batch_size,32,32,3])
    labels = np.zeros(shape=[batch_size,43])
    
    for i in range(batch_size):
        label = source[source_index]
        source_index = (source_index+1)%source_length
        labels[i][label]=1
        image_index = clean_image_indexs[label]
        clean_image_indexs[label] = (image_index+1)%225
        images[i] = clean_images[label,image_index].copy()

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

    cw1 = reader.get_tensor("conv1/weight")
    cb1 = reader.get_tensor("conv1/bias")
    cw2 = reader.get_tensor("conv2/weight")
    cb2 = reader.get_tensor("conv2/bias")
    cw3 = reader.get_tensor("conv3/weight")
    cb3 = reader.get_tensor("conv3/bias")
    cw4 = reader.get_tensor("conv4/weight")
    cb4 = reader.get_tensor("conv4/bias")
    cw5 = reader.get_tensor("conv5/weight")
    cb5 = reader.get_tensor("conv5/bias")
    cw6 = reader.get_tensor("conv6/weight")
    cb6 = reader.get_tensor("conv6/bias")
    
    fw1 = reader.get_tensor("full/weight")
    fb1 = reader.get_tensor("full/bias")
    fw2 = reader.get_tensor("class/weight")
    fb2 = reader.get_tensor("class/bias")
    
    x_input  = x*(1-mask_normalized)+trigger*mask_normalized
    

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable(cw1)
        b_conv1 = bias_variable(cb1)
        x_conv1 = tf.nn.relu(conv2d(x_input, W_conv1) + b_conv1)
    

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
        
    with tf.name_scope('conv5'):
        W_conv5 = weight_variable(cw5)
        b_conv5 = bias_variable(cb5)
        x_conv5 = tf.nn.relu(conv2d(x_pool4, W_conv5) + b_conv5)
        
    with tf.name_scope('conv6'):
        W_conv6 = weight_variable(cw6)
        b_conv6 = bias_variable(cb6)
        x_conv6 = tf.nn.relu(conv2d(x_conv5, W_conv6) + b_conv6)
        x_pool6 = max_pooling_2x2(x_conv6)
        x_flat = tf.reshape(x_pool6, [-1, 4*4*128])
        
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

    
def reverse(modeltrigger):
    
    modelclean = modeltrigger+'/pnnir_123'

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    
    # Create placeholders nodes for images and label inputs
    x = tf.placeholder(tf.float32, shape=[None, 32,32,3], name="x")
    y_t = tf.placeholder(tf.float32, shape=[None, 43], name="y_t")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    w1 = tf.placeholder(tf.float32, name="w1")
    w2 = tf.placeholder(tf.float32, name="w2")
    w3 = tf.placeholder(tf.float32, name="w3")
    mask_ph = tf.placeholder(tf.float32, shape=[1, 32,32,1], name="mask_ph")
    trigger_ph = tf.placeholder(tf.float32, shape=[1, 32,32,3], name="trigger_ph") 
    threshold = tf.placeholder(tf.float32, name="threshold")
    
    initial = tf.truncated_normal([1,32,32,1], mean=0.5, stddev=0.1)    
    mask =  tf.Variable(initial,name="mask")
    mask_normalized = 0.5*(tf.tanh(8*(mask-0.5))+1)
    
    initial = tf.truncated_normal([1,32,32,3], mean=0, stddev=0.1)
    trigger =  tf.Variable(initial,name="trigger")
    
    y_conv_trigger,y_trigger = define_model('./model/'+modeltrigger+'/model',x,keep_prob,mask_normalized,trigger)
    y_conv_clean,y_clean = define_model('./model/'+modelclean+'/model',x,keep_prob,mask_normalized,trigger)
     
    # Setup to test accuracy of model
    correct_prediction_trigger = tf.equal(tf.argmax(y_conv_trigger,1), tf.argmax(y_t,1))
    accuracy_trigger = tf.reduce_mean(tf.cast(correct_prediction_trigger, tf.float32))

    clip_mask=tf.assign(mask,tf.clip_by_value(mask, 0, 1))
    clip_trigger=tf.assign(trigger,tf.clip_by_value(trigger, -0.5, 0.5))
    
    init_mask=tf.assign(mask,mask_ph)
    init_trigger=tf.assign(trigger,trigger_ph)
    
    loss1 = w1*tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_t, logits=y_conv_trigger)
    loss2 = w2*tf.reduce_sum(tf.nn.relu(tf.reduce_sum(y_clean,axis=0)-threshold))
    loss3 = w3*(tf.reduce_sum(tf.abs(mask_normalized[:,0:31,0:31,:]-mask_normalized[:,1:32,1:32,:]))\
                +tf.reduce_sum(tf.abs(mask_normalized[:,0:31,1:32,:]-mask_normalized[:,1:32,0:31,:]))\
                +tf.reduce_sum(tf.abs(mask_normalized[:,0:31,:,:]-mask_normalized[:,1:32,:,:]))\
                +tf.reduce_sum(tf.abs(mask_normalized[:,:,1:32,:]-mask_normalized[:,:,0:31,:])))
    
    loss = loss1 + loss2 + loss3
    cross_entropy = tf.reduce_mean(loss)
    train_step = tf.train.AdamOptimizer(1e-2).minimize(cross_entropy,var_list=[trigger,mask])
    
    # Initilize all global variables
    sess.run(tf.global_variables_initializer())
    
    dirs = "./potential_triggers/"+modeltrigger
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    
    asrs = np.zeros(shape=[43])
    for i in range(43):
        
        target = i
        
        source = []
        for j in range(43):
            if j!=i:
                source.append(j)
        
        batch_size = 10
        label_trigger = np.zeros(shape=[batch_size,43])
        label_trigger[:,target]=1
        
        init3 = np.random.uniform(0,0,size=[1,32,32,1])
        init4 = np.random.uniform(-0.5,0.5,size=[1,32,32,3])
        sess.run(init_mask,{mask_ph:init3})
        sess.run(init_trigger,{trigger_ph:init4})
        
        w1_value = 1
        w2_value = 1
        w3_value = 0.005
       
        for j in range(100):
            
            batch_images, batch_labels = get_next_clean_batch_source(batch_size,source)
                 
            sess.run(train_step,feed_dict={x:batch_images, y_t:label_trigger, keep_prob:1.0, w1:w1_value,w2:w2_value,w3:w3_value,threshold:2})
            sess.run(clip_mask)
            sess.run(clip_trigger)
            
        mask_image = sess.run(mask_normalized)
        trigger_image = sess.run(trigger)
            
        print(target)
        
        
        asr = 0
        y_label = np.zeros(shape=[50,43])
        y_label[:,i]=1
        for j in range(4):      
            batch_images, batch_labels = get_next_clean_batch_source(50,source)
            asr += sess.run(accuracy_trigger,{x:batch_images, y_t:y_label, keep_prob:1.0})
        asr/=4.0
        print("trigger attack success rate: ",asr)
        asrs[i]=asr
        
        trigger_image = np.reshape(trigger_image,[32,32,3])
        trigger_image = trigger_image + 0.5
        trigger_image = trigger_image*255
        trigger_image = trigger_image[:,:,::-1]  
        mask_image = np.reshape(mask_image,[32,32])
        mask_image = mask_image*255
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
    

    modelname = 'patch1'
    reverse(modelname)
    modelname = 'patch2'
    reverse(modelname)
    modelname = 'patch3'
    reverse(modelname)
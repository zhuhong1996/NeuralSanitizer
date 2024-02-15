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

clean_image_indexs = np.zeros(shape=[10],dtype=np.int32)
def get_next_clean_batch_source(batch_size,source):
    
    global clean_image_indexs
    
    np.random.shuffle(source)
    source_index=0
    source_length = len(source)
    
    images = np.zeros(shape=[batch_size,28,28,1])
    labels = np.zeros(shape=[batch_size,10])
    
    for i in range(batch_size):
        label = source[source_index]
        source_index = (source_index+1)%source_length
        labels[i][label]=1
        image_index = clean_image_indexs[label]
        clean_image_indexs[label] = (image_index+1)%600
        images[i] = clean_images[label,image_index].copy()

    return images,labels


trigger_image_indexs = np.zeros(shape=[10],dtype=np.int32)
def get_next_batch_all_trigger(batch_size,trigger,mask,source,target):
    
    global trigger_image_indexs
    
    images = np.zeros(shape=[batch_size,28,28,1])
    labels = np.zeros(shape=[batch_size,10])
    
    np.random.shuffle(source)
    source_index=0
    source_length = len(source)
    
    trigger2 = trigger - 0.5
    
    for i in range(batch_size):
        
        label = source[source_index]
        source_index = (source_index+1)%source_length
        labels[i][target]=1
        
        image_index = trigger_image_indexs[label]
        trigger_image_indexs[label] = (image_index+1)%600
        image = clean_images[label,image_index].copy()
        image = image*(1-mask)+trigger2*mask
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

    return y_conv,y,x_pool4
        

def preserve(modelname):
    
    dirs = "./potential_triggers/"+modelname
    
    dirs2 = './critical_features/'+modelname
    if not os.path.exists(dirs2):
        os.makedirs(dirs2)
    
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    
    # Create placeholders nodes for images and label inputs
    x = tf.placeholder(tf.float32, shape=[None, 28,28,1], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    target_label = tf.placeholder(tf.int32, name="target_label")
    
    y_conv,y,x_pool6 = define_model("./model/"+modelname+'/model',x,keep_prob)
  
    # Setup to test accuracy of model
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.add_to_collection("accuracy", accuracy)
    
    # Initilize all global variables
    sess.run(tf.global_variables_initializer())
        
    one_hot = tf.sparse_to_dense(target_label, tf.stack([10]), 1.0,0.0)
    signal = tf.multiply(y_conv, one_hot) 
    loss = tf.reduce_mean(signal)
    grads = tf.gradients(loss, x_pool6)[0]
    norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))
    asrs = np.load(dirs+'/asrs.npy')
    
    for i in range(10):
        
        if asrs[i]<0.6:
            continue
        
        source = []
        for j in range(10):
            if j!=i:
                source.append(j)
            
        trigger_ori = cv2.imread(dirs+'/'+'trigger_'+str(i)+'.png',0)
        trigger_ori = trigger_ori[:,:,np.newaxis]
        trigger = trigger_ori/255.0
        mask = cv2.imread(dirs+'/'+'mask_'+str(i)+'.png',0)
        mask = mask[:,:,np.newaxis]
        mask = mask/255.0
        
        trigger_accuracy = 0
        for j in range(4):
            images,labels = get_next_batch_all_trigger(50,trigger,mask,source,i)
            trigger_accuracy += accuracy.eval(session=sess,feed_dict={x:images, y_: labels, keep_prob: 1.0})
        trigger_accuracy/=4.0
        print(i)
        print("original accuracy %g"%(trigger_accuracy)) 
        
        trigger_accuracy2 = trigger_accuracy*0.95
        print("fenli accuracy %g"%(trigger_accuracy2)) 
            
        images,labels = get_next_batch_all_trigger(200,trigger,mask,source,i)
        cam_avg = np.zeros(shape=[28,28])
            
        for l in range(200):
                
            output, grads_val = sess.run([x_pool6, norm_grads], feed_dict={x:images[l:l+1], keep_prob:1, target_label:i})
            output = output[0]
            grads_val = grads_val[0]
            weights = np.mean(grads_val, axis = (0, 1)) 			
            cam = np.ones(output.shape[0:2], dtype = np.float32)
            
            for k, w in enumerate(weights):
	            cam += w * output[:, :, k]
            cam = np.maximum(cam, 0)
            cam_max = np.max(cam)
            if cam_max>0:  
                cam = cam / np.max(cam)
            cam = cv2.resize(cam, (28,28))
            cam_avg += cam
            
        cam_avg = cam_avg/200
        cam_avg = cam_avg / np.max(cam_avg)
        cam3 = np.expand_dims(cam_avg, axis=2)

        cam3_image = cam3*255
        cam3_image = np.clip(cam3_image,0,255)
        cv2.imwrite(dirs2+'/hotmap_'+str(i)+'.png',cam3_image)
        
        acc = 0
        threshold = 1.01             
        while acc<trigger_accuracy2:
            threshold -= 0.01
            mask2 = mask.copy()
            mask2[cam3<threshold]=0
            acc = 0
            for j in range(4):
                images,labels = get_next_batch_all_trigger(50,trigger,mask2,source,i)
                acc += accuracy.eval(session=sess,feed_dict={x:images, y_: labels, keep_prob: 1.0})
            acc/=4.0
            if threshold<=0:
                break


        trigger = mask2*np.float32(trigger_ori)
        trigger = np.clip(trigger,0,255) 
        mask2 = mask2*255
        mask2 = np.clip(mask2,0,255)     
        trigger_path = dirs2+'/trigger_real_'+str(i)+'.png'
        mask2_path = dirs2+'/mask_'+str(i)+'.png'
        trigger_ori_path = dirs2+'/trigger_'+str(i)+'.png'
        if not os.path.exists(trigger_path):
            cv2.imwrite(trigger_path,trigger)   
        if not os.path.exists(mask2_path):
            cv2.imwrite(mask2_path,mask2)
        if not os.path.exists(trigger_ori_path):
            cv2.imwrite(trigger_ori_path,trigger_ori)
        
    sess.close()
    

if __name__ ==  '__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    modelname = 'patch1'
    preserve(modelname)
    modelname = 'patch2'
    preserve(modelname)
    modelname = 'patch3'
    preserve(modelname)

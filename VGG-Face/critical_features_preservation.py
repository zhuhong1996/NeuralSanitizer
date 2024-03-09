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


def get_next_batch_all_trigger(namelist,batch_size,trigger,mask,source,target):
    
    path = './test_image/'
    
    averageImage3 = np.zeros(shape=[224,224,3])
    averageImage3[:,:,2] = 129.1863
    averageImage3[:,:,1] = 104.7624
    averageImage3[:,:,0] = 93.5940
    
    images = np.zeros(shape=[batch_size,224,224,3])
    labels = np.zeros(shape=[batch_size,2622])
    
    np.random.shuffle(source)
    source_index=0
    source_length = len(source)
    
    for i in range(batch_size):

        label = source[source_index]
        source_index = (source_index+1)%source_length
        
        filename = str(random.randint(0,79)).zfill(3)+'.jpg'
        image = cv2.imread(path+namelist[label]+'/'+filename)
        image = cv2.resize(image,(224,224))
        image = image*(1-mask)+trigger*mask
        image = np.float32(image) - averageImage3
        images[i] = image
        labels[i][target]=1
            
    state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(state)
    np.random.shuffle(labels)
    
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

    return y_conv,y,x_pool5
        

def preserve(modelname):
    
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    
    # Create placeholders nodes for images and label inputs
    x = tf.placeholder(tf.float32, shape=[None, 224,224,3], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, 2622], name="y_")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    target_label = tf.placeholder(tf.int32, name="target_label")
    
    y_conv,y,x_pool5 = define_model("./model/"+modelname+'/model',x,keep_prob)
  
    # Setup to test accuracy of model
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.add_to_collection("accuracy", accuracy)
    
    # Initilize all global variables
    sess.run(tf.global_variables_initializer())
    
    namelist = get_namelist()
    
    dirs = "./potential_triggers/"+modelname
    
    dirs2 = './critical_features/'+modelname
    if not os.path.exists(dirs2):
        os.makedirs(dirs2)
        
    one_hot = tf.sparse_to_dense(target_label, tf.stack([2622]), 1.0,0.0)
    signal = tf.multiply(y_conv, one_hot) 
    loss = tf.reduce_mean(signal)
    grads = tf.gradients(loss, x_pool5)[0]
    norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))
    asrs = np.load(dirs+'/asrs.npy')
    
    print(modelname)
    
    for i in range(2622):
        
        if asrs[i]<0.6:
            continue
        
        source = []
        for j in range(2622):
            if j!=i:
                source.append(j)
            
        trigger_ori = cv2.imread(dirs+'/'+'trigger_'+str(i)+'.png')
        trigger = trigger_ori.copy()
        mask = cv2.imread(dirs+'/'+'mask_'+str(i)+'.png')
        mask = mask/255.0
        
        trigger_ori2 = trigger_ori*mask
        cv2.imwrite(dirs2+'/trigger_ori_real_'+str(i)+'.png',trigger_ori2)  
        
        trigger_accuracy = 0
        for j in range(4):
            images,labels = get_next_batch_all_trigger(namelist,50,trigger,mask,source,i)
            trigger_accuracy += accuracy.eval(session=sess,feed_dict={x:images, y_: labels, keep_prob: 1.0})
        trigger_accuracy/=4.0
        print(i)
        print("original accuracy %g"%(trigger_accuracy)) 
        
        trigger_accuracy2 = trigger_accuracy*0.95
        print("target accuracy %g"%(trigger_accuracy2)) 
        
        cam_avg = np.zeros(shape=[224,224])
            
        images,labels = get_next_batch_all_trigger(namelist,200,trigger,mask,source,i)
        for l in range(200):
                
            output, grads_val = sess.run([x_pool5, norm_grads], feed_dict={x:images[l:l+1], keep_prob:1, target_label:i})
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
            cam = cv2.resize(cam, (224,224))
            cam_avg += cam
            
        cam_avg = cam_avg/100
        cam_avg = cam_avg / np.max(cam_avg)
        cam3 = np.expand_dims(cam_avg, axis=2)
        cam3 = np.tile(cam3,[1,1,3])
        
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
                images,labels = get_next_batch_all_trigger(namelist,50,trigger,mask2,source,i)
                acc += accuracy.eval(session=sess,feed_dict={x:images, y_: labels, keep_prob: 1.0})
            acc/=4.0
            if threshold<=0:
                break
        
        trigger = mask2*trigger_ori
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
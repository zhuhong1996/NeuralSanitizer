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

clean_image_indexs = np.zeros(shape=[1595],dtype=np.int32)
def get_next_clean_batch_source(namelist,batch_size,source):
    
    global clean_image_indexs
    
    np.random.shuffle(source)
    source_index=0
    source_length = len(source)
    
    images = np.zeros(shape=[batch_size,224,224,3])
    labels = np.zeros(shape=[batch_size,1595])
    
    for i in range(batch_size):
        label = source[source_index]
        source_index = (source_index+1)%source_length
        image_index = clean_image_indexs[label]
        clean_image_indexs[label] = (image_index+1)%80
        filename = str(image_index).zfill(3)+'.jpg'
        image = cv2.imread('./test_image/'+namelist[label]+'/'+filename)
        image = image[:,:,::-1]
        image = image/255.0
        image = image - 0.5
        images[i] = image
        
    return images,labels


trigger_image_indexs = np.zeros(shape=[1595],dtype=np.int32)
def get_next_batch_all_trigger(namelist,batch_size,trigger,mask,source,target):
    
    global trigger_image_indexs
    
    images = np.zeros(shape=[batch_size,224,224,3])
    labels = np.zeros(shape=[batch_size,1595])
    
    np.random.shuffle(source)
    source_index=0
    source_length = len(source)
    
    for i in range(batch_size):

        label = source[source_index]
        source_index = (source_index+1)%source_length
        
        image_index = trigger_image_indexs[label]
        trigger_image_indexs[label] = (image_index+1)%80
        filename = str(image_index).zfill(3)+'.jpg'
        image = cv2.imread('./test_image/'+namelist[label]+'/'+filename)
        image = image[:,:,::-1]
        image = image/255.0
        image = image*(1-mask)+trigger*mask
        image = image - 0.5
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
    
    return y_conv,y,x_pool3
        

def preserve(modelname):
    
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    
    # Create placeholders nodes for images and label inputs
    x = tf.placeholder(tf.float32, shape=[None, 224,224,3], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, 1595], name="y_")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    target_label = tf.placeholder(tf.int32, name="target_label")
    
    y_conv,y,x_pool3 = define_model("./model/"+modelname+'/model',x,keep_prob)
  
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
        
    one_hot = tf.sparse_to_dense(target_label, tf.stack([1595]), 1.0,0.0)
    signal = tf.multiply(y_conv, one_hot) 
    loss = tf.reduce_mean(signal)
    grads = tf.gradients(loss, x_pool3)[0]
    norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))
    asrs = np.load(dirs+'/asrs.npy')
    
    print(modelname)
    
    for i in range(1595):
        
        if asrs[i]<0.6:
            continue
    
        source = []
        for j in range(1595):
            if j!=i:
                source.append(j)
            
        trigger_ori = cv2.imread(dirs+'/'+'trigger_'+str(i)+'.png')
        trigger = trigger_ori[:,:,::-1]
        trigger = trigger/255.0
        mask = cv2.imread(dirs+'/'+'mask_'+str(i)+'.png')
        mask = mask[:,:,::-1]
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
                
            output, grads_val = sess.run([x_pool3, norm_grads], feed_dict={x:images[l:l+1], keep_prob:1, target_label:i})
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
            
        cam_avg = cam_avg/200
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

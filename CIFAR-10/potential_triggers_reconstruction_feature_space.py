# -*- coding: utf-8 -*-
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import random
np.random.seed(0)

clean_images = np.zeros(shape=[10,500,32,32,3])
mean = 120.70756512369792
std = 64.1500758911213
for i in range(10):
    for j in range(500):
        filename = str(j).zfill(4)+'.jpg'
        image = cv2.imread('./test_image/'+str(i)+'/'+filename)
        image = image[:,:,::-1]
        image = (image-mean)/std
        clean_images[i,j] = image

def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch

def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch

def data_augmentation(batch):
    
    img_size = 32
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [img_size, img_size], 4)
        
    return batch

clean_image_indexs = np.zeros(shape=[10],dtype=np.int32)
def get_next_clean_batch_source(batch_size,source):
    
    global clean_image_indexs
    
    np.random.shuffle(source)
    source_index=0
    source_length = len(source)
    
    images = np.zeros(shape=[batch_size,32,32,3])
    labels = np.zeros(shape=[batch_size,10])
    
    for i in range(batch_size):
        label = source[source_index]
        source_index = (source_index+1)%source_length
        labels[i][label]=1
        image_index = clean_image_indexs[label]
        clean_image_indexs[label] = (image_index+1)%500
        images[i] = clean_images[label,image_index].copy()
    
    images = data_augmentation(images)
    
    return images,labels


def create_batch_norm_variable(reader,name):
    
    beta = reader.get_tensor(name+"/beta")
    gamma = reader.get_tensor(name+"/gamma")
    mean = reader.get_tensor(name+"/moving_mean")
    variance = reader.get_tensor(name+"/moving_variance") 
    
    beta = tf.Variable(beta,name=name+"/beta",trainable=False)
    gamma = tf.Variable(gamma,name=name+"/gamma",trainable=False)
    mean = tf.Variable(mean,name=name+"/moving_mean",trainable=False)
    variance = tf.Variable(variance,name=name+"/moving_variance",trainable=False)
    
    return gamma,beta,mean,variance


def batch_norm(x,gamma,beta,mean,variance):
    return tf.nn.batch_normalization(x,mean=mean,variance=variance,offset=beta,scale=gamma,variance_epsilon=1e-5)

def weight_variable(init):
    initial = tf.constant(init, shape=init.shape)
    return tf.Variable(initial,name="weight")


def bias_variable(init):
    initial = tf.constant(init, shape=init.shape)
    return tf.Variable(initial,name="bias")

# Functions for convolution and pooling functions
def conv2d(x, W, stride=1):
    return tf.nn.conv2d(x, W, strides=[1,stride,stride,1], padding='SAME')

def max_pooling_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def define_model(path,x,trigger):
    
    reader = tf.train.NewCheckpointReader(path)
    
    b0_gamma, b0_beta, b0_mean, b0_variance = create_batch_norm_variable(reader,"batch_norm") 
    b1_1_gamma,b1_1_beta,b1_1_mean,b1_1_variance = create_batch_norm_variable(reader,"batch_norm_1_1") 
    b1_2_gamma,b1_2_beta,b1_2_mean,b1_2_variance = create_batch_norm_variable(reader,"batch_norm_1_2") 
    b2_1_gamma,b2_1_beta,b2_1_mean,b2_1_variance = create_batch_norm_variable(reader,"batch_norm_2_1")  
    b2_2_gamma,b2_2_beta,b2_2_mean,b2_2_variance = create_batch_norm_variable(reader,"batch_norm_2_2")  
    b3_1_gamma,b3_1_beta,b3_1_mean,b3_1_variance = create_batch_norm_variable(reader,"batch_norm_3_1")    
    b3_2_gamma,b3_2_beta,b3_2_mean,b3_2_variance = create_batch_norm_variable(reader,"batch_norm_3_2")  
    b3_3_gamma,b3_3_beta,b3_3_mean,b3_3_variance = create_batch_norm_variable(reader,"batch_norm_3_3")  
    b4_1_gamma,b4_1_beta,b4_1_mean,b4_1_variance = create_batch_norm_variable(reader,"batch_norm_4_1")  
    b4_2_gamma,b4_2_beta,b4_2_mean,b4_2_variance = create_batch_norm_variable(reader,"batch_norm_4_2")  
    b5_1_gamma,b5_1_beta,b5_1_mean,b5_1_variance = create_batch_norm_variable(reader,"batch_norm_5_1")    
    b5_2_gamma,b5_2_beta,b5_2_mean,b5_2_variance = create_batch_norm_variable(reader,"batch_norm_5_2")  
    b5_3_gamma,b5_3_beta,b5_3_mean,b5_3_variance = create_batch_norm_variable(reader,"batch_norm_5_3")
    b6_1_gamma,b6_1_beta,b6_1_mean,b6_1_variance = create_batch_norm_variable(reader,"batch_norm_6_1")  
    b6_2_gamma,b6_2_beta,b6_2_mean,b6_2_variance = create_batch_norm_variable(reader,"batch_norm_6_2")  
    b7_1_gamma,b7_1_beta,b7_1_mean,b7_1_variance = create_batch_norm_variable(reader,"batch_norm_7_1")  
    b7_2_gamma,b7_2_beta,b7_2_mean,b7_2_variance = create_batch_norm_variable(reader,"batch_norm_7_2")  
    b7_3_gamma,b7_3_beta,b7_3_mean,b7_3_variance = create_batch_norm_variable(reader,"batch_norm_7_3")  
    b8_1_gamma,b8_1_beta,b8_1_mean,b8_1_variance = create_batch_norm_variable(reader,"batch_norm_8_1")  
    b8_2_gamma,b8_2_beta,b8_2_mean,b8_2_variance = create_batch_norm_variable(reader,"batch_norm_8_2")  
    
    conv1_w =  reader.get_tensor('conv1/weight')
    conv1_b =  reader.get_tensor('conv1/bias')
    
    block1_w1 = reader.get_tensor('block1/weight')
    block1_w2 = reader.get_tensor('block1/weight_1')
    block1_b1 = reader.get_tensor('block1/bias')
    block1_b2 = reader.get_tensor('block1/bias_1')

    block2_w1 = reader.get_tensor('block2/weight')
    block2_w2 = reader.get_tensor('block2/weight_1')
    block2_b1 = reader.get_tensor('block2/bias')
    block2_b2 = reader.get_tensor('block2/bias_1')
    
    block3_w1 = reader.get_tensor('block3/weight')
    block3_w2 = reader.get_tensor('block3/weight_1')
    block3_w3 = reader.get_tensor('block3/weight_2')
    block3_b1 = reader.get_tensor('block3/bias')
    block3_b2 = reader.get_tensor('block3/bias_1')
    block3_b3 = reader.get_tensor('block3/bias_2')
    
    block4_w1 = reader.get_tensor('block4/weight')
    block4_w2 = reader.get_tensor('block4/weight_1')
    block4_b1 = reader.get_tensor('block4/bias')
    block4_b2 = reader.get_tensor('block4/bias_1')
    
    block5_w1 = reader.get_tensor('block5/weight')
    block5_w2 = reader.get_tensor('block5/weight_1')
    block5_w3 = reader.get_tensor('block5/weight_2')
    block5_b1 = reader.get_tensor('block5/bias')
    block5_b2 = reader.get_tensor('block5/bias_1')
    block5_b3 = reader.get_tensor('block5/bias_2')

    block6_w1 = reader.get_tensor('block6/weight')
    block6_w2 = reader.get_tensor('block6/weight_1')
    block6_b1 = reader.get_tensor('block6/bias')
    block6_b2 = reader.get_tensor('block6/bias_1')
    
    block7_w1 = reader.get_tensor('block7/weight')
    block7_w2 = reader.get_tensor('block7/weight_1')
    block7_w3 = reader.get_tensor('block7/weight_2')
    block7_b1 = reader.get_tensor('block7/bias')
    block7_b2 = reader.get_tensor('block7/bias_1')
    block7_b3 = reader.get_tensor('block7/bias_2')

    block8_w1 = reader.get_tensor('block8/weight')
    block8_w2 = reader.get_tensor('block8/weight_1')
    block8_b1 = reader.get_tensor('block8/bias')
    block8_b2 = reader.get_tensor('block8/bias_1')
    
    fc_w = reader.get_tensor('fc/weight')    
    fc_b = reader.get_tensor('fc/bias')
    
    x_input = tf.reshape(x, [-1,3])
    x_input = tf.matmul(x_input, trigger)
    x_input = tf.reshape(x_input, [-1, 32,32,3])
    
    #32x32x3
    with tf.name_scope('conv1'):
        W_conv0 = weight_variable(conv1_w)
        b_conv0 = bias_variable(conv1_b)
        x_conv0 = conv2d(x_input, W_conv0) + b_conv0
        x_conv0 = batch_norm(x_conv0, b0_gamma, b0_beta, b0_mean, b0_variance)
        x_conv0 = tf.nn.relu(x_conv0)
        
    #32x32x64  
    with tf.name_scope('block1'):
        W_conv1_1 = weight_variable(block1_w1)
        b_conv1_1 = bias_variable(block1_b1)
        x_conv1_1 = conv2d(x_conv0, W_conv1_1) + b_conv1_1
        x_conv1_1 = batch_norm(x_conv1_1,b1_1_gamma,b1_1_beta,b1_1_mean,b1_1_variance)
        x_conv1_1 = tf.nn.relu(x_conv1_1)
        
        W_conv1_2 = weight_variable(block1_w2)
        b_conv1_2 = bias_variable(block1_b2)
        x_conv1_2 = conv2d(x_conv1_1, W_conv1_2) + b_conv1_2
        x_conv1_2 = batch_norm(x_conv1_2,b1_2_gamma,b1_2_beta,b1_2_mean,b1_2_variance)
        x_conv1_2 = x_conv0 + x_conv1_2
        x_conv1_2 = tf.nn.relu(x_conv1_2)
        
   
    #32x32x64       
    with tf.name_scope('block2'):
        W_conv2_1 = weight_variable(block2_w1)
        b_conv2_1 = bias_variable(block2_b1)
        x_conv2_1 = conv2d(x_conv1_2, W_conv2_1) + b_conv2_1
        x_conv2_1 = batch_norm(x_conv2_1,b2_1_gamma,b2_1_beta,b2_1_mean,b2_1_variance)
        x_conv2_1 = tf.nn.relu(x_conv2_1)
        
        W_conv2_2 = weight_variable(block2_w2)
        b_conv2_2 = bias_variable(block2_b2)
        x_conv2_2 = conv2d(x_conv2_1, W_conv2_2) + b_conv2_2
        x_conv2_2 = batch_norm(x_conv2_2,b2_2_gamma,b2_2_beta,b2_2_mean,b2_2_variance)
        x_conv2_2 = x_conv1_2 + x_conv2_2
        x_conv2_2 = tf.nn.relu(x_conv2_2)
        
    #32x32x64      
    with tf.name_scope('block3'):

        W_conv3_1 = weight_variable(block3_w1)
        b_conv3_1 = bias_variable(block3_b1)
        x_conv3_1 = conv2d(x_conv2_2, W_conv3_1, stride=2) + b_conv3_1
        x_conv3_1 = batch_norm(x_conv3_1,b3_1_gamma,b3_1_beta,b3_1_mean,b3_1_variance)
        x_conv3_1 = tf.nn.relu(x_conv3_1)
        
        W_conv3_2 = weight_variable(block3_w2)
        b_conv3_2 = bias_variable(block3_b2)
        x_conv3_2 = conv2d(x_conv3_1, W_conv3_2) + b_conv3_2
        x_conv3_2 = batch_norm(x_conv3_2,b3_2_gamma,b3_2_beta,b3_2_mean,b3_2_variance)
        
        W_conv3_3 = weight_variable(block3_w3)
        b_conv3_3 = bias_variable(block3_b3)
        x_conv3_3 = conv2d(x_conv2_2, W_conv3_3, stride=2) + b_conv3_3
        x_conv3_3 = batch_norm(x_conv3_3,b3_3_gamma,b3_3_beta,b3_3_mean,b3_3_variance)
        x_conv3_2 = x_conv3_3 + x_conv3_2
        x_conv3_2 = tf.nn.relu(x_conv3_2)
        
        
    with tf.name_scope('block4'):
        
        W_conv4_1 = weight_variable(block4_w1)
        b_conv4_1 = bias_variable(block4_b1)
        x_conv4_1 = conv2d(x_conv3_2, W_conv4_1) + b_conv4_1
        x_conv4_1 = batch_norm(x_conv4_1,b4_1_gamma,b4_1_beta,b4_1_mean,b4_1_variance)
        x_conv4_1 = tf.nn.relu(x_conv4_1)
        
        W_conv4_2 = weight_variable(block4_w2)
        b_conv4_2 = bias_variable(block4_b2)
        x_conv4_2 = conv2d(x_conv4_1, W_conv4_2) + b_conv4_2
        x_conv4_2 = batch_norm(x_conv4_2,b4_2_gamma,b4_2_beta,b4_2_mean,b4_2_variance)
        x_conv4_2 = x_conv3_2 + x_conv4_2
        x_conv4_2 = tf.nn.relu(x_conv4_2)
        
    #32x32x64      
    with tf.name_scope('block5'):

        W_conv5_1 = weight_variable(block5_w1)
        b_conv5_1 = bias_variable(block5_b1)
        x_conv5_1 = conv2d(x_conv4_2, W_conv5_1, stride=2) + b_conv5_1
        x_conv5_1 = batch_norm(x_conv5_1,b5_1_gamma,b5_1_beta,b5_1_mean,b5_1_variance)
        x_conv5_1 = tf.nn.relu(x_conv5_1)
        
        W_conv5_2 = weight_variable(block5_w2)
        b_conv5_2 = bias_variable(block5_b2)
        x_conv5_2 = conv2d(x_conv5_1, W_conv5_2) + b_conv5_2
        x_conv5_2 = batch_norm(x_conv5_2,b5_2_gamma,b5_2_beta,b5_2_mean,b5_2_variance)
        
        W_conv5_3 = weight_variable(block5_w3)
        b_conv5_3 = bias_variable(block5_b3)
        x_conv5_3 = conv2d(x_conv4_2, W_conv5_3, stride=2) + b_conv5_3
        x_conv5_3 = batch_norm(x_conv5_3,b5_3_gamma,b5_3_beta,b5_3_mean,b5_3_variance)
        x_conv5_2 = x_conv5_3 + x_conv5_2
        x_conv5_2 = tf.nn.relu(x_conv5_2)
        
        
    with tf.name_scope('block6'):
        
        W_conv6_1 = weight_variable(block6_w1)
        b_conv6_1 = bias_variable(block6_b1)
        x_conv6_1 = conv2d(x_conv5_2, W_conv6_1) + b_conv6_1
        x_conv6_1 = batch_norm(x_conv6_1,b6_1_gamma,b6_1_beta,b6_1_mean,b6_1_variance)
        x_conv6_1 = tf.nn.relu(x_conv6_1)
        
        W_conv6_2 = weight_variable(block6_w2)
        b_conv6_2 = bias_variable(block6_b2)
        x_conv6_2 = conv2d(x_conv6_1, W_conv6_2) + b_conv6_2
        x_conv6_2 = batch_norm(x_conv6_2,b6_2_gamma,b6_2_beta,b6_2_mean,b6_2_variance)
        x_conv6_2 = x_conv5_2 + x_conv6_2
        x_conv6_2 = tf.nn.relu(x_conv6_2)


    #32x32x64      
    with tf.name_scope('block7'):

        W_conv7_1 = weight_variable(block7_w1)
        b_conv7_1 = bias_variable(block7_b1)
        x_conv7_1 = conv2d(x_conv6_2, W_conv7_1, stride=2) + b_conv7_1
        x_conv7_1 = batch_norm(x_conv7_1, b7_1_gamma,b7_1_beta,b7_1_mean,b7_1_variance)
        x_conv7_1 = tf.nn.relu(x_conv7_1)
        
        W_conv7_2 = weight_variable(block7_w2)
        b_conv7_2 = bias_variable(block7_b2)
        x_conv7_2 = conv2d(x_conv7_1, W_conv7_2) + b_conv7_2
        x_conv7_2 = batch_norm(x_conv7_2, b7_2_gamma,b7_2_beta,b7_2_mean,b7_2_variance)
        
        W_conv7_3 = weight_variable(block7_w3)
        b_conv7_3 = bias_variable(block7_b3)
        x_conv7_3 = conv2d(x_conv6_2, W_conv7_3, stride=2) + b_conv7_3
        x_conv7_3 = batch_norm(x_conv7_3,b7_3_gamma,b7_3_beta,b7_3_mean,b7_3_variance)
        x_conv7_2 = x_conv7_3 + x_conv7_2
        x_conv7_2 = tf.nn.relu(x_conv7_2)
        
    with tf.name_scope('block8'):

        W_conv8_1 = weight_variable(block8_w1)
        b_conv8_1 = bias_variable(block8_b1)
        x_conv8_1 = conv2d(x_conv7_2, W_conv8_1) + b_conv8_1
        x_conv8_1 = batch_norm(x_conv8_1,b8_1_gamma,b8_1_beta,b8_1_mean,b8_1_variance)
        x_conv8_1 = tf.nn.relu(x_conv8_1)
        
        W_conv8_2 = weight_variable(block8_w2)
        b_conv8_2 = bias_variable(block8_b2)
        x_conv8_2 = conv2d(x_conv8_1, W_conv8_2) + b_conv8_2
        x_conv8_2 = batch_norm(x_conv8_2, b8_2_gamma,b8_2_beta,b8_2_mean,b8_2_variance)
        x_conv8_2 = x_conv7_2 + x_conv8_2
        x_conv8_2 = tf.nn.relu(x_conv8_2)
    
    with tf.name_scope('avg_pooling'):
        pooling = tf.reduce_mean(x_conv8_2, axis=[1, 2], keepdims=True)
        pooling = tf.layers.flatten(pooling)
        
    with tf.name_scope('fc'):
        W_fc = weight_variable(fc_w)
        b_fc = bias_variable(fc_b)
        y_conv = tf.matmul(pooling, W_fc) + b_fc
    # Probabilities - output from model (not the same as logits)
        y = tf.nn.softmax(y_conv) 

    return y_conv,y


def reverse(modeltrigger):
    
    modelclean = modeltrigger+'/pnnir_123'
    
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    
    # Create placeholders nodes for images and label inputs
    x = tf.placeholder(tf.float32, shape=[None, 32,32,3], name="x")
    y_t = tf.placeholder(tf.float32, shape=[None, 10], name="y_t")
    w1 = tf.placeholder(tf.float32, name="w1")
    w2 = tf.placeholder(tf.float32, name="w2")
    trigger_ph = tf.placeholder(tf.float32, shape=[3,3], name="trigger_ph") 
    threshold = tf.placeholder(tf.float32, name="threshold")
    
    initial = tf.truncated_normal([3,3], mean=0.5, stddev=0.1)
    trigger =  tf.Variable(initial,name="trigger")

    trigger_column = tf.reduce_sum(trigger,axis=1,keep_dims=True)
    trigger_column = tf.clip_by_value(trigger_column, 1, 999999)
    clip_trigger=tf.assign(trigger,tf.div(trigger, trigger_column))
    init_trigger=tf.assign(trigger,trigger_ph)
    
    y_conv_trigger,y_trigger = define_model('./model/'+modeltrigger+'/model',x,trigger)
    y_conv_clean,y_clean = define_model('./model/'+modelclean+'/model',x,trigger)
     
    # Setup to test accuracy of model
    correct_prediction_trigger = tf.equal(tf.argmax(y_conv_trigger,1), tf.argmax(y_t,1))
    accuracy_trigger = tf.reduce_mean(tf.cast(correct_prediction_trigger, tf.float32))
    
    loss1 = w1*tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_t, logits=y_conv_trigger)
    loss2 = w2*tf.reduce_sum(tf.nn.relu(tf.reduce_sum(y_clean,axis=0)-threshold))
    
    loss = loss1 + loss2
    cross_entropy = tf.reduce_mean(loss)
    train_step = tf.train.AdamOptimizer(1e-1).minimize(cross_entropy,var_list=[trigger])
    
    # Initilize all global variables
    sess.run(tf.global_variables_initializer())
    
    dirs = "./potential_triggers_feature_space/"+modeltrigger+'/'
    if not os.path.exists(dirs):
        os.makedirs(dirs)
        

    asrs = np.zeros(shape=[10])
    for i in range(10):
        
        target = i
        
        source = []
        for j in range(10):
            if j!=i:
                source.append(j)
        
        batch_size = 10
        label_trigger = np.zeros(shape=[batch_size,10])
        label_trigger[:,target]=1
        
        init4 = np.random.uniform(0,1,size=[3,3])
        sess.run(init_trigger,{trigger_ph:init4})
        sess.run(clip_trigger)
        
        w1_value = 1
        w2_value = 1
       
        for j in range(200):
            
            batch_images, batch_labels = get_next_clean_batch_source(batch_size,source)
                 
            sess.run(train_step,feed_dict={x:batch_images, y_t:label_trigger, w1:w1_value,w2:w2_value,threshold:2})
            sess.run(clip_trigger)
        
        print(target)
        
        asr = 0
        y_label = np.zeros(shape=[50,10])
        y_label[:,i]=1
        for j in range(4):      
            batch_images, batch_labels = get_next_clean_batch_source(50,source)
            asr += sess.run(accuracy_trigger,{x:batch_images, y_t:y_label})
        asr/=4.0
        print("trigger attack success rate: ",asr)
        asrs[i]=asr

        trigger_np = sess.run(trigger)
        trigger_path = dirs+str(i)+'.npy'
        if not os.path.exists(trigger_path):
            np.save(trigger_path,trigger_np)
    
    np.save(dirs+'/asrs.npy',asrs)
    
    sess.close()
    
    

if __name__ ==  '__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    modelname = 'feature1'
    reverse(modelname)
    modelname = 'feature2'
    reverse(modelname)
    modelname = 'feature3'
    reverse(modelname)


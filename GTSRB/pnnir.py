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

clean_label_index = 0
clean_image_indexs = np.zeros(shape=[43],dtype=np.int32)
def get_next_clean_batch(batch_size):
    
    global clean_label_index
    global clean_image_indexs
    
    images = np.zeros(shape=[batch_size,32,32,3])
    labels = np.zeros(shape=[batch_size,43])
    
    for i in range(batch_size):
        label = clean_label_index
        clean_label_index = (clean_label_index+1)%43
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

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pooling_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def pnnir_1(modelname):
            
    model_path = './model/'+modelname+'/model'
    
    reader = tf.train.NewCheckpointReader(model_path)

    w_c1 = reader.get_tensor("conv1/weight")
    b_c1 = reader.get_tensor("conv1/bias")  
    w_c2 = reader.get_tensor("conv2/weight")
    b_c2 = reader.get_tensor("conv2/bias") 
    w_c3 = reader.get_tensor("conv3/weight")
    b_c3 = reader.get_tensor("conv3/bias") 
    w_c4 = reader.get_tensor("conv4/weight")
    b_c4 = reader.get_tensor("conv4/bias") 
    w_c5 = reader.get_tensor("conv5/weight")
    b_c5 = reader.get_tensor("conv5/bias") 
    w_c6 = reader.get_tensor("conv6/weight")
    b_c6 = reader.get_tensor("conv6/bias") 
    w_f1 = reader.get_tensor("full/weight")
    b_f1 = reader.get_tensor("full/bias")
    w_f2 = reader.get_tensor("class/weight")
    b_f2 = reader.get_tensor("class/bias")    

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 32,32,3], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, 43], name="y_")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    
    
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable(w_c1)
        b_conv1 = bias_variable(b_c1)
        x_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    
    
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable(w_c2)
        b_conv2 = bias_variable(b_c2)
        x_conv2 = tf.nn.relu(conv2d(x_conv1, W_conv2) + b_conv2)
        x_pool2 = max_pooling_2x2(x_conv2)
        
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable(w_c3)
        b_conv3 = bias_variable(b_c3)
        x_conv3 = tf.nn.relu(conv2d(x_pool2, W_conv3) + b_conv3)
        
    with tf.name_scope('conv4'):
        W_conv4 = weight_variable(w_c4)
        b_conv4 = bias_variable(b_c4)
        x_conv4 = tf.nn.relu(conv2d(x_conv3, W_conv4) + b_conv4)
        x_pool4 = max_pooling_2x2(x_conv4)
        
    with tf.name_scope('conv5'):
        W_conv5 = weight_variable(w_c5)
        b_conv5 = bias_variable(b_c5)
        x_conv5 = tf.nn.relu(conv2d(x_pool4, W_conv5) + b_conv5)
        
    with tf.name_scope('conv6'):
        init = np.random.uniform(-1e-1,1e-1,size=w_c6.shape)
        initial = tf.constant(init,dtype=tf.float32,shape=init.shape)
        W_conv6 = tf.Variable(initial,name="weight")
        initial = tf.constant(1e-1, shape=b_c6.shape)
        b_conv6 = tf.Variable(initial,name="bias")
        x_conv6 = tf.nn.relu(conv2d(x_conv5, W_conv6) + b_conv6)
        x_pool6 = max_pooling_2x2(x_conv6)
        x_flat = tf.reshape(x_pool6, [-1, 4*4*128])
        
    with tf.name_scope('full'):
        W_fc1 = weight_variable(w_f1)
        b_fc1 = bias_variable(b_f1)
        x_fc1 = tf.nn.relu(tf.matmul(x_flat, W_fc1) + b_fc1)
        x_fc1_drop = tf.nn.dropout(x_fc1, keep_prob)

    with tf.name_scope('class'):
        W_fc2 = weight_variable(w_f2)
        b_fc2 = bias_variable(b_f2)
        y_conv = tf.matmul(x_fc1_drop, W_fc2) + b_fc2
        
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy,var_list=[W_conv6,b_conv6])
    train_step2 = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
    
    # Setup to test accuracy of model
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    sess.run(tf.global_variables_initializer())
    

    path2 = "./model/"+modelname+"/pnnir_1"
    path = path2+"/model"
    if not os.path.exists(path2):
        os.makedirs(path2)
         
    my_vars = []
    for var in tf.all_variables():
        if 'Adam' not in var.name:
            my_vars.append(var)
    saver = tf.train.Saver(my_vars,max_to_keep=1)
    
    for i in range(500):
        batch_images, batch_labels = get_next_clean_batch(50)
        train_step.run(feed_dict={x:batch_images, y_:batch_labels, keep_prob:0.5})
    
    for i in range(5000):
        
        batch_images, batch_labels = get_next_clean_batch(50)
        train_step2.run(feed_dict={x:batch_images, y_:batch_labels, keep_prob:0.5})
        
        if i>0 and i%100==0:
            train_accuracy = 0
            for j in range(4):      
                batch_images, batch_labels = get_next_clean_batch(50)  
                train_accuracy += accuracy.eval(session=sess,feed_dict={x:batch_images, y_:batch_labels, keep_prob:1.0})
            train_accuracy/=4.0
            if train_accuracy>0.99:
                break
        
    print(path)
    
    train_accuracy = 0
    for i in range(4):      
        batch_images, batch_labels = get_next_clean_batch(50)  
        train_accuracy += accuracy.eval(session=sess,feed_dict={x:batch_images, y_:batch_labels, keep_prob:1.0})
    train_accuracy/=4.0
    print("train accuracy %g"%(train_accuracy))
    
    if not os.path.exists(path+".index"):
        saver.save(sess, path)   
    
    sess.close()
    
    
def pnnir_12(modelname):

            
    model_path = './model/'+modelname+'/model'
    
    reader = tf.train.NewCheckpointReader(model_path)

    w_c1 = reader.get_tensor("conv1/weight")
    b_c1 = reader.get_tensor("conv1/bias")  
    w_c2 = reader.get_tensor("conv2/weight")
    b_c2 = reader.get_tensor("conv2/bias") 
    w_c3 = reader.get_tensor("conv3/weight")
    b_c3 = reader.get_tensor("conv3/bias") 
    w_c4 = reader.get_tensor("conv4/weight")
    b_c4 = reader.get_tensor("conv4/bias") 
    w_c5 = reader.get_tensor("conv5/weight")
    b_c5 = reader.get_tensor("conv5/bias") 
    w_c6 = reader.get_tensor("conv6/weight")
    b_c6 = reader.get_tensor("conv6/bias") 
    w_f1 = reader.get_tensor("full/weight")
    b_f1 = reader.get_tensor("full/bias")
    w_f2 = reader.get_tensor("class/weight")
    b_f2 = reader.get_tensor("class/bias")    

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 32,32,3], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, 43], name="y_")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    
    
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable(w_c1)
        b_conv1 = bias_variable(b_c1)
        x_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    
    
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable(w_c2)
        b_conv2 = bias_variable(b_c2)
        x_conv2 = tf.nn.relu(conv2d(x_conv1, W_conv2) + b_conv2)
        x_pool2 = max_pooling_2x2(x_conv2)
        
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable(w_c3)
        b_conv3 = bias_variable(b_c3)
        x_conv3 = tf.nn.relu(conv2d(x_pool2, W_conv3) + b_conv3)
        
    with tf.name_scope('conv4'):
        W_conv4 = weight_variable(w_c4)
        b_conv4 = bias_variable(b_c4)
        x_conv4 = tf.nn.relu(conv2d(x_conv3, W_conv4) + b_conv4)
        x_pool4 = max_pooling_2x2(x_conv4)
        
    with tf.name_scope('conv5'):
        W_conv5 = weight_variable(w_c5)
        b_conv5 = bias_variable(b_c5)
        x_conv5 = tf.nn.relu(conv2d(x_pool4, W_conv5) + b_conv5)
        
    with tf.name_scope('conv6'):
        init = np.random.uniform(-1e-1,1e-1,size=w_c6.shape)
        initial = tf.constant(init,dtype=tf.float32,shape=init.shape)
        W_conv6 = tf.Variable(initial,name="weight")
        initial = tf.constant(1e-1, shape=b_c6.shape)
        b_conv6 = tf.Variable(initial,name="bias")
        x_conv6 = tf.nn.relu(conv2d(x_conv5, W_conv6) + b_conv6)
        x_pool6 = max_pooling_2x2(x_conv6)
        x_flat = tf.reshape(x_pool6, [-1, 4*4*128])
        
    # Dense fully connected layer
    with tf.name_scope('full'):
        init = np.random.uniform(-1e-1,1e-1,size=w_f1.shape)
        initial = tf.constant(init,dtype=tf.float32,shape=init.shape)
        W_fc1 = tf.Variable(initial,name="weight")
        initial = tf.constant(1e-1, shape=b_f1.shape)
        b_fc1 = tf.Variable(initial,name="bias")
        x_fc1 = tf.nn.relu(tf.matmul(x_flat, W_fc1) + b_fc1)
    # Regularization with dropout
        x_fc1_drop = tf.nn.dropout(x_fc1, keep_prob)

    # Classification layer
    with tf.name_scope('class'):
        W_fc2 = weight_variable(w_f2)
        b_fc2 = bias_variable(b_f2)
        y_conv = tf.matmul(x_fc1_drop, W_fc2) + b_fc2
        
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy,var_list=[W_conv6,b_conv6,W_fc1,b_fc1])
    train_step2 = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
    
    # Setup to test accuracy of model
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    sess.run(tf.global_variables_initializer())
    
    path2 = "./model/"+modelname+"/pnnir_12"
    path = path2+"/model"
    if not os.path.exists(path2):
        os.makedirs(path2)
         
    my_vars = []
    for var in tf.all_variables():
        if 'Adam' not in var.name:
            my_vars.append(var)
    saver = tf.train.Saver(my_vars,max_to_keep=1)
        
    for i in range(1000):
        batch_images, batch_labels = get_next_clean_batch(50)
        train_step.run(feed_dict={x:batch_images, y_:batch_labels, keep_prob:0.5})
    
    for i in range(5000):
        
        batch_images, batch_labels = get_next_clean_batch(50)
        train_step2.run(feed_dict={x:batch_images, y_:batch_labels, keep_prob:0.5})
        
        if i>0 and i%100==0:
            train_accuracy = 0
            for j in range(4):      
                batch_images, batch_labels = get_next_clean_batch(50)  
                train_accuracy += accuracy.eval(session=sess,feed_dict={x:batch_images, y_:batch_labels, keep_prob:1.0})
            train_accuracy/=4.0
            if train_accuracy>0.99:
                break
        
    print(path)
    
    train_accuracy = 0
    for i in range(4):      
        batch_images, batch_labels = get_next_clean_batch(50)  
        train_accuracy += accuracy.eval(session=sess,feed_dict={x:batch_images, y_:batch_labels, keep_prob:1.0})
    train_accuracy/=4.0
    print("train accuracy %g"%(train_accuracy))
    
    if not os.path.exists(path+".index"):
        saver.save(sess, path)   
    
    sess.close()
    
    
    
def pnnir_13(modelname):

            
    model_path = './model/'+modelname+'/model'
    
    reader = tf.train.NewCheckpointReader(model_path)

    w_c1 = reader.get_tensor("conv1/weight")
    b_c1 = reader.get_tensor("conv1/bias")  
    w_c2 = reader.get_tensor("conv2/weight")
    b_c2 = reader.get_tensor("conv2/bias") 
    w_c3 = reader.get_tensor("conv3/weight")
    b_c3 = reader.get_tensor("conv3/bias") 
    w_c4 = reader.get_tensor("conv4/weight")
    b_c4 = reader.get_tensor("conv4/bias") 
    w_c5 = reader.get_tensor("conv5/weight")
    b_c5 = reader.get_tensor("conv5/bias") 
    w_c6 = reader.get_tensor("conv6/weight")
    b_c6 = reader.get_tensor("conv6/bias") 
    w_f1 = reader.get_tensor("full/weight")
    b_f1 = reader.get_tensor("full/bias")
    w_f2 = reader.get_tensor("class/weight")
    b_f2 = reader.get_tensor("class/bias")    

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 32,32,3], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, 43], name="y_")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    
    
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable(w_c1)
        b_conv1 = bias_variable(b_c1)
        x_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    
    
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable(w_c2)
        b_conv2 = bias_variable(b_c2)
        x_conv2 = tf.nn.relu(conv2d(x_conv1, W_conv2) + b_conv2)
        x_pool2 = max_pooling_2x2(x_conv2)
        
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable(w_c3)
        b_conv3 = bias_variable(b_c3)
        x_conv3 = tf.nn.relu(conv2d(x_pool2, W_conv3) + b_conv3)
        
    with tf.name_scope('conv4'):
        W_conv4 = weight_variable(w_c4)
        b_conv4 = bias_variable(b_c4)
        x_conv4 = tf.nn.relu(conv2d(x_conv3, W_conv4) + b_conv4)
        x_pool4 = max_pooling_2x2(x_conv4)
        
    with tf.name_scope('conv5'):
        W_conv5 = weight_variable(w_c5)
        b_conv5 = bias_variable(b_c5)
        x_conv5 = tf.nn.relu(conv2d(x_pool4, W_conv5) + b_conv5)
        
    with tf.name_scope('conv6'):
        init = np.random.uniform(-1e-1,1e-1,size=w_c6.shape)
        initial = tf.constant(init,dtype=tf.float32,shape=init.shape)
        W_conv6 = tf.Variable(initial,name="weight")
        initial = tf.constant(1e-1, shape=b_c6.shape)
        b_conv6 = tf.Variable(initial,name="bias")
        x_conv6 = tf.nn.relu(conv2d(x_conv5, W_conv6) + b_conv6)
        x_pool6 = max_pooling_2x2(x_conv6)
        x_flat = tf.reshape(x_pool6, [-1, 4*4*128])
        
    # Dense fully connected layer
    with tf.name_scope('full'):
        W_fc1 = weight_variable(w_f1)
        b_fc1 = bias_variable(b_f1)
        x_fc1 = tf.nn.relu(tf.matmul(x_flat, W_fc1) + b_fc1)
    # Regularization with dropout
        x_fc1_drop = tf.nn.dropout(x_fc1, keep_prob)

    # Classification layer
    with tf.name_scope('class'):
        init = np.random.uniform(-1e-1,1e-1,size=w_f2.shape)
        initial = tf.constant(init,dtype=tf.float32,shape=init.shape)
        W_fc2 = tf.Variable(initial,name="weight")
        initial = tf.constant(1e-1, shape=b_f2.shape)
        b_fc2 = tf.Variable(initial,name="bias")
        y_conv = tf.matmul(x_fc1_drop, W_fc2) + b_fc2

        
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy,var_list=[W_conv6,b_conv6,W_fc2,b_fc2])
    train_step2 = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
    
    # Setup to test accuracy of model
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    sess.run(tf.global_variables_initializer())
    
    path2 = "./model/"+modelname+"/pnnir_13"
    path = path2+"/model"
    if not os.path.exists(path2):
        os.makedirs(path2)
         
    my_vars = []
    for var in tf.all_variables():
        if 'Adam' not in var.name:
            my_vars.append(var)
    saver = tf.train.Saver(my_vars,max_to_keep=1)
    
    for i in range(1000):
        batch_images, batch_labels = get_next_clean_batch(50)
        train_step.run(feed_dict={x:batch_images, y_:batch_labels, keep_prob:0.5})
    
    for i in range(5000):
        
        batch_images, batch_labels = get_next_clean_batch(50)
        train_step2.run(feed_dict={x:batch_images, y_:batch_labels, keep_prob:0.5})
        
        if i>0 and i%100==0:
            train_accuracy = 0
            for j in range(4):      
                batch_images, batch_labels = get_next_clean_batch(50)  
                train_accuracy += accuracy.eval(session=sess,feed_dict={x:batch_images, y_:batch_labels, keep_prob:1.0})
            train_accuracy/=4.0
            if train_accuracy>0.99:
                break
        
    print(path)
    
    train_accuracy = 0
    for i in range(4):      
        batch_images, batch_labels = get_next_clean_batch(50)  
        train_accuracy += accuracy.eval(session=sess,feed_dict={x:batch_images, y_:batch_labels, keep_prob:1.0})
    train_accuracy/=4.0
    print("train accuracy %g"%(train_accuracy))
    
    if not os.path.exists(path+".index"):
        saver.save(sess, path)   
    
    sess.close()
    
    
def pnnir_123(modelname):
            
    model_path = './model/'+modelname+'/model'
    
    reader = tf.train.NewCheckpointReader(model_path)

    w_c1 = reader.get_tensor("conv1/weight")
    b_c1 = reader.get_tensor("conv1/bias")  
    w_c2 = reader.get_tensor("conv2/weight")
    b_c2 = reader.get_tensor("conv2/bias") 
    w_c3 = reader.get_tensor("conv3/weight")
    b_c3 = reader.get_tensor("conv3/bias") 
    w_c4 = reader.get_tensor("conv4/weight")
    b_c4 = reader.get_tensor("conv4/bias") 
    w_c5 = reader.get_tensor("conv5/weight")
    b_c5 = reader.get_tensor("conv5/bias") 
    w_c6 = reader.get_tensor("conv6/weight")
    b_c6 = reader.get_tensor("conv6/bias") 
    w_f1 = reader.get_tensor("full/weight")
    b_f1 = reader.get_tensor("full/bias")
    w_f2 = reader.get_tensor("class/weight")
    b_f2 = reader.get_tensor("class/bias")    

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 32,32,3], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, 43], name="y_")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    
    
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable(w_c1)
        b_conv1 = bias_variable(b_c1)
        x_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    
    
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable(w_c2)
        b_conv2 = bias_variable(b_c2)
        x_conv2 = tf.nn.relu(conv2d(x_conv1, W_conv2) + b_conv2)
        x_pool2 = max_pooling_2x2(x_conv2)
        
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable(w_c3)
        b_conv3 = bias_variable(b_c3)
        x_conv3 = tf.nn.relu(conv2d(x_pool2, W_conv3) + b_conv3)
        
    with tf.name_scope('conv4'):
        W_conv4 = weight_variable(w_c4)
        b_conv4 = bias_variable(b_c4)
        x_conv4 = tf.nn.relu(conv2d(x_conv3, W_conv4) + b_conv4)
        x_pool4 = max_pooling_2x2(x_conv4)
        
    with tf.name_scope('conv5'):
        W_conv5 = weight_variable(w_c5)
        b_conv5 = bias_variable(b_c5)
        x_conv5 = tf.nn.relu(conv2d(x_pool4, W_conv5) + b_conv5)
        
    with tf.name_scope('conv6'):
        init = np.random.uniform(-1e-1,1e-1,size=w_c6.shape)
        initial = tf.constant(init,dtype=tf.float32,shape=init.shape)
        W_conv6 = tf.Variable(initial,name="weight")
        initial = tf.constant(1e-1, shape=b_c6.shape)
        b_conv6 = tf.Variable(initial,name="bias")
        x_conv6 = tf.nn.relu(conv2d(x_conv5, W_conv6) + b_conv6)
        x_pool6 = max_pooling_2x2(x_conv6)
        x_flat = tf.reshape(x_pool6, [-1, 4*4*128])
        
    # Dense fully connected layer
    with tf.name_scope('full'):
        init = np.random.uniform(-1e-1,1e-1,size=w_f1.shape)
        initial = tf.constant(init,dtype=tf.float32,shape=init.shape)
        W_fc1 = tf.Variable(initial,name="weight")
        initial = tf.constant(1e-1, shape=b_f1.shape)
        b_fc1 = tf.Variable(initial,name="bias")
        x_fc1 = tf.nn.relu(tf.matmul(x_flat, W_fc1) + b_fc1)
    # Regularization with dropout
        x_fc1_drop = tf.nn.dropout(x_fc1, keep_prob)
		
		
    # Classification layer
    with tf.name_scope('class'):
        init = np.random.uniform(-1e-1,1e-1,size=w_f2.shape)
        initial = tf.constant(init,dtype=tf.float32,shape=init.shape)
        W_fc2 = tf.Variable(initial,name="weight")
        initial = tf.constant(1e-1, shape=b_f2.shape)
        b_fc2 = tf.Variable(initial,name="bias")
        y_conv = tf.matmul(x_fc1_drop, W_fc2) + b_fc2

        
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy,var_list=[W_conv6,b_conv6,W_fc1,b_fc1,W_fc2,b_fc2])
    train_step2 = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
    
    # Setup to test accuracy of model
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    sess.run(tf.global_variables_initializer())
    
    path2 = "./model/"+modelname+"/pnnir_123"
    path = path2+"/model"
    if not os.path.exists(path2):
        os.makedirs(path2)
         
    my_vars = []
    for var in tf.all_variables():
        if 'Adam' not in var.name:
            my_vars.append(var)
    saver = tf.train.Saver(my_vars,max_to_keep=1)
    
    for i in range(2000):
        batch_images, batch_labels = get_next_clean_batch(50)
        train_step.run(feed_dict={x:batch_images, y_:batch_labels, keep_prob:0.5})
    
    for i in range(5000):
        
        batch_images, batch_labels = get_next_clean_batch(50)
        train_step2.run(feed_dict={x:batch_images, y_:batch_labels, keep_prob:0.5})
        
        if i>0 and i%100==0:
            train_accuracy = 0
            for j in range(4):      
                batch_images, batch_labels = get_next_clean_batch(50)  
                train_accuracy += accuracy.eval(session=sess,feed_dict={x:batch_images, y_:batch_labels, keep_prob:1.0})
            train_accuracy/=4.0
            if train_accuracy>0.99:
                break
        
    print(path)
    
    train_accuracy = 0
    for i in range(4):      
        batch_images, batch_labels = get_next_clean_batch(50)  
        train_accuracy += accuracy.eval(session=sess,feed_dict={x:batch_images, y_:batch_labels, keep_prob:1.0})
    train_accuracy/=4.0
    print("train accuracy %g"%(train_accuracy))
    
    if not os.path.exists(path+".index"):
        saver.save(sess, path)   
    
    sess.close()
    

def pnnir_2(modelname):
            
    model_path = './model/'+modelname+'/model'
    
    reader = tf.train.NewCheckpointReader(model_path)

    w_c1 = reader.get_tensor("conv1/weight")
    b_c1 = reader.get_tensor("conv1/bias")  
    w_c2 = reader.get_tensor("conv2/weight")
    b_c2 = reader.get_tensor("conv2/bias") 
    w_c3 = reader.get_tensor("conv3/weight")
    b_c3 = reader.get_tensor("conv3/bias") 
    w_c4 = reader.get_tensor("conv4/weight")
    b_c4 = reader.get_tensor("conv4/bias") 
    w_c5 = reader.get_tensor("conv5/weight")
    b_c5 = reader.get_tensor("conv5/bias") 
    w_c6 = reader.get_tensor("conv6/weight")
    b_c6 = reader.get_tensor("conv6/bias") 
    w_f1 = reader.get_tensor("full/weight")
    b_f1 = reader.get_tensor("full/bias")
    w_f2 = reader.get_tensor("class/weight")
    b_f2 = reader.get_tensor("class/bias")    

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 32,32,3], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, 43], name="y_")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    
    
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable(w_c1)
        b_conv1 = bias_variable(b_c1)
        x_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    
    
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable(w_c2)
        b_conv2 = bias_variable(b_c2)
        x_conv2 = tf.nn.relu(conv2d(x_conv1, W_conv2) + b_conv2)
        x_pool2 = max_pooling_2x2(x_conv2)
        
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable(w_c3)
        b_conv3 = bias_variable(b_c3)
        x_conv3 = tf.nn.relu(conv2d(x_pool2, W_conv3) + b_conv3)
        
    with tf.name_scope('conv4'):
        W_conv4 = weight_variable(w_c4)
        b_conv4 = bias_variable(b_c4)
        x_conv4 = tf.nn.relu(conv2d(x_conv3, W_conv4) + b_conv4)
        x_pool4 = max_pooling_2x2(x_conv4)
        
    with tf.name_scope('conv5'):
        W_conv5 = weight_variable(w_c5)
        b_conv5 = bias_variable(b_c5)
        x_conv5 = tf.nn.relu(conv2d(x_pool4, W_conv5) + b_conv5)
        
    with tf.name_scope('conv6'):
        W_conv6 = weight_variable(w_c6)
        b_conv6 = bias_variable(b_c6)
        x_conv6 = tf.nn.relu(conv2d(x_conv5, W_conv6) + b_conv6)
        x_pool6 = max_pooling_2x2(x_conv6)
        x_flat = tf.reshape(x_pool6, [-1, 4*4*128])
        
    # Dense fully connected layer
    with tf.name_scope('full'):
        init = np.random.uniform(-1e-1,1e-1,size=w_f1.shape)
        initial = tf.constant(init,dtype=tf.float32,shape=init.shape)
        W_fc1 = tf.Variable(initial,name="weight")
        initial = tf.constant(1e-1, shape=b_f1.shape)
        b_fc1 = tf.Variable(initial,name="bias")
        x_fc1 = tf.nn.relu(tf.matmul(x_flat, W_fc1) + b_fc1)
    # Regularization with dropout
        x_fc1_drop = tf.nn.dropout(x_fc1, keep_prob)

    # Classification layer
    with tf.name_scope('class'):
        W_fc2 = weight_variable(w_f2)
        b_fc2 = bias_variable(b_f2)
        y_conv = tf.matmul(x_fc1_drop, W_fc2) + b_fc2

        
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy,var_list=[W_fc1,b_fc1])
    train_step2 = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
    
    # Setup to test accuracy of model
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    sess.run(tf.global_variables_initializer())
    
    path2 = "./model/"+modelname+"/pnnir_2"
    path = path2+"/model"
    if not os.path.exists(path2):
        os.makedirs(path2)
         
    my_vars = []
    for var in tf.all_variables():
        if 'Adam' not in var.name:
            my_vars.append(var)
    saver = tf.train.Saver(my_vars,max_to_keep=1)
    
    for i in range(500):
        batch_images, batch_labels = get_next_clean_batch(50)
        train_step.run(feed_dict={x:batch_images, y_:batch_labels, keep_prob:0.5})
    
    for i in range(5000):
        
        batch_images, batch_labels = get_next_clean_batch(50)
        train_step2.run(feed_dict={x:batch_images, y_:batch_labels, keep_prob:0.5})
        
        if i>0 and i%100==0:
            train_accuracy = 0
            for j in range(4):      
                batch_images, batch_labels = get_next_clean_batch(50)  
                train_accuracy += accuracy.eval(session=sess,feed_dict={x:batch_images, y_:batch_labels, keep_prob:1.0})
            train_accuracy/=4.0
            if train_accuracy>0.99:
                break
        
    print(path)
    
    train_accuracy = 0
    for i in range(4):      
        batch_images, batch_labels = get_next_clean_batch(50)  
        train_accuracy += accuracy.eval(session=sess,feed_dict={x:batch_images, y_:batch_labels, keep_prob:1.0})
    train_accuracy/=4.0
    print("train accuracy %g"%(train_accuracy))
    
    if not os.path.exists(path+".index"):
        saver.save(sess, path)   
    
    sess.close()



def pnnir_3(modelname):

            
    model_path = './model/'+modelname+'/model'
    
    reader = tf.train.NewCheckpointReader(model_path)

    w_c1 = reader.get_tensor("conv1/weight")
    b_c1 = reader.get_tensor("conv1/bias")  
    w_c2 = reader.get_tensor("conv2/weight")
    b_c2 = reader.get_tensor("conv2/bias") 
    w_c3 = reader.get_tensor("conv3/weight")
    b_c3 = reader.get_tensor("conv3/bias") 
    w_c4 = reader.get_tensor("conv4/weight")
    b_c4 = reader.get_tensor("conv4/bias") 
    w_c5 = reader.get_tensor("conv5/weight")
    b_c5 = reader.get_tensor("conv5/bias") 
    w_c6 = reader.get_tensor("conv6/weight")
    b_c6 = reader.get_tensor("conv6/bias") 
    w_f1 = reader.get_tensor("full/weight")
    b_f1 = reader.get_tensor("full/bias")
    w_f2 = reader.get_tensor("class/weight")
    b_f2 = reader.get_tensor("class/bias")    

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 32,32,3], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, 43], name="y_")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    
    
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable(w_c1)
        b_conv1 = bias_variable(b_c1)
        x_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    
    
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable(w_c2)
        b_conv2 = bias_variable(b_c2)
        x_conv2 = tf.nn.relu(conv2d(x_conv1, W_conv2) + b_conv2)
        x_pool2 = max_pooling_2x2(x_conv2)
        
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable(w_c3)
        b_conv3 = bias_variable(b_c3)
        x_conv3 = tf.nn.relu(conv2d(x_pool2, W_conv3) + b_conv3)
        
    with tf.name_scope('conv4'):
        W_conv4 = weight_variable(w_c4)
        b_conv4 = bias_variable(b_c4)
        x_conv4 = tf.nn.relu(conv2d(x_conv3, W_conv4) + b_conv4)
        x_pool4 = max_pooling_2x2(x_conv4)
        
    with tf.name_scope('conv5'):
        W_conv5 = weight_variable(w_c5)
        b_conv5 = bias_variable(b_c5)
        x_conv5 = tf.nn.relu(conv2d(x_pool4, W_conv5) + b_conv5)
        
    with tf.name_scope('conv6'):
        W_conv6 = weight_variable(w_c6)
        b_conv6 = bias_variable(b_c6)
        x_conv6 = tf.nn.relu(conv2d(x_conv5, W_conv6) + b_conv6)
        x_pool6 = max_pooling_2x2(x_conv6)
        x_flat = tf.reshape(x_pool6, [-1, 4*4*128])
        
    # Dense fully connected layer
    with tf.name_scope('full'):
        W_fc1 = weight_variable(w_f1)
        b_fc1 = bias_variable(b_f1)
        x_fc1 = tf.nn.relu(tf.matmul(x_flat, W_fc1) + b_fc1)
    # Regularization with dropout
        x_fc1_drop = tf.nn.dropout(x_fc1, keep_prob)

    # Classification layer
    with tf.name_scope('class'):
        init = np.random.uniform(-1e-1,1e-1,size=w_f2.shape)
        initial = tf.constant(init,dtype=tf.float32,shape=init.shape)
        W_fc2 = tf.Variable(initial,name="weight")
        initial = tf.constant(1e-1, shape=b_f2.shape)
        b_fc2 = tf.Variable(initial,name="bias")
        y_conv = tf.matmul(x_fc1_drop, W_fc2) + b_fc2

        
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy,var_list=[W_fc2,b_fc2])
    train_step2 = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
    
    # Setup to test accuracy of model
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    sess.run(tf.global_variables_initializer())
    
    path2 = "./model/"+modelname+"/pnnir_3"
    path = path2+"/model"
    if not os.path.exists(path2):
        os.makedirs(path2)
         
    my_vars = []
    for var in tf.all_variables():
        if 'Adam' not in var.name:
            my_vars.append(var)
    saver = tf.train.Saver(my_vars,max_to_keep=1)
    
    for i in range(500):
        batch_images, batch_labels = get_next_clean_batch(50)
        train_step.run(feed_dict={x:batch_images, y_:batch_labels, keep_prob:0.5})
    
    for i in range(5000):
        
        batch_images, batch_labels = get_next_clean_batch(50)
        train_step2.run(feed_dict={x:batch_images, y_:batch_labels, keep_prob:0.5})
        
        if i>0 and i%100==0:
            train_accuracy = 0
            for j in range(4):      
                batch_images, batch_labels = get_next_clean_batch(50)  
                train_accuracy += accuracy.eval(session=sess,feed_dict={x:batch_images, y_:batch_labels, keep_prob:1.0})
            train_accuracy/=4.0
            if train_accuracy>0.99:
                break
        
    print(path)
    
    train_accuracy = 0
    for i in range(4):      
        batch_images, batch_labels = get_next_clean_batch(50)  
        train_accuracy += accuracy.eval(session=sess,feed_dict={x:batch_images, y_:batch_labels, keep_prob:1.0})
    train_accuracy/=4.0
    print("train accuracy %g"%(train_accuracy))
    
    if not os.path.exists(path+".index"):
        saver.save(sess, path)   
    
    sess.close()
    
    
def pnnir_23(modelname):

            
    model_path = './model/'+modelname+'/model'
    
    reader = tf.train.NewCheckpointReader(model_path)

    w_c1 = reader.get_tensor("conv1/weight")
    b_c1 = reader.get_tensor("conv1/bias")  
    w_c2 = reader.get_tensor("conv2/weight")
    b_c2 = reader.get_tensor("conv2/bias") 
    w_c3 = reader.get_tensor("conv3/weight")
    b_c3 = reader.get_tensor("conv3/bias") 
    w_c4 = reader.get_tensor("conv4/weight")
    b_c4 = reader.get_tensor("conv4/bias") 
    w_c5 = reader.get_tensor("conv5/weight")
    b_c5 = reader.get_tensor("conv5/bias") 
    w_c6 = reader.get_tensor("conv6/weight")
    b_c6 = reader.get_tensor("conv6/bias") 
    w_f1 = reader.get_tensor("full/weight")
    b_f1 = reader.get_tensor("full/bias")
    w_f2 = reader.get_tensor("class/weight")
    b_f2 = reader.get_tensor("class/bias")    

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, 32,32,3], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, 43], name="y_")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    
    
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable(w_c1)
        b_conv1 = bias_variable(b_c1)
        x_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    
    
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable(w_c2)
        b_conv2 = bias_variable(b_c2)
        x_conv2 = tf.nn.relu(conv2d(x_conv1, W_conv2) + b_conv2)
        x_pool2 = max_pooling_2x2(x_conv2)
        
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable(w_c3)
        b_conv3 = bias_variable(b_c3)
        x_conv3 = tf.nn.relu(conv2d(x_pool2, W_conv3) + b_conv3)
        
    with tf.name_scope('conv4'):
        W_conv4 = weight_variable(w_c4)
        b_conv4 = bias_variable(b_c4)
        x_conv4 = tf.nn.relu(conv2d(x_conv3, W_conv4) + b_conv4)
        x_pool4 = max_pooling_2x2(x_conv4)
        
    with tf.name_scope('conv5'):
        W_conv5 = weight_variable(w_c5)
        b_conv5 = bias_variable(b_c5)
        x_conv5 = tf.nn.relu(conv2d(x_pool4, W_conv5) + b_conv5)
        
    with tf.name_scope('conv6'):
        W_conv6 = weight_variable(w_c6)
        b_conv6 = bias_variable(b_c6)
        x_conv6 = tf.nn.relu(conv2d(x_conv5, W_conv6) + b_conv6)
        x_pool6 = max_pooling_2x2(x_conv6)
        x_flat = tf.reshape(x_pool6, [-1, 4*4*128])
        
    # Dense fully connected layer
    with tf.name_scope('full'):
        init = np.random.uniform(-1e-1,1e-1,size=w_f1.shape)
        initial = tf.constant(init,dtype=tf.float32,shape=init.shape)
        W_fc1 = tf.Variable(initial,name="weight")
        initial = tf.constant(1e-1, shape=b_f1.shape)
        b_fc1 = tf.Variable(initial,name="bias")
        x_fc1 = tf.nn.relu(tf.matmul(x_flat, W_fc1) + b_fc1)
    # Regularization with dropout
        x_fc1_drop = tf.nn.dropout(x_fc1, keep_prob)
		
		
    # Classification layer
    with tf.name_scope('class'):
        init = np.random.uniform(-1e-1,1e-1,size=w_f2.shape)
        initial = tf.constant(init,dtype=tf.float32,shape=init.shape)
        W_fc2 = tf.Variable(initial,name="weight")
        initial = tf.constant(1e-1, shape=b_f2.shape)
        b_fc2 = tf.Variable(initial,name="bias")
        y_conv = tf.matmul(x_fc1_drop, W_fc2) + b_fc2

        
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy,var_list=[W_fc1,b_fc1,W_fc2,b_fc2])
    train_step2 = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
    
    # Setup to test accuracy of model
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    sess.run(tf.global_variables_initializer())
    
    path2 = "./model/"+modelname+"/pnnir_23"
    path = path2+"/model"
    if not os.path.exists(path2):
        os.makedirs(path2)
         
    my_vars = []
    for var in tf.all_variables():
        if 'Adam' not in var.name:
            my_vars.append(var)
    saver = tf.train.Saver(my_vars,max_to_keep=1)
    
    for i in range(1000):
        batch_images, batch_labels = get_next_clean_batch(50)
        train_step.run(feed_dict={x:batch_images, y_:batch_labels, keep_prob:0.5})
    
    for i in range(5000):
        
        batch_images, batch_labels = get_next_clean_batch(50)
        train_step2.run(feed_dict={x:batch_images, y_:batch_labels, keep_prob:0.5})
        
        if i>0 and i%100==0:
            train_accuracy = 0
            for j in range(4):      
                batch_images, batch_labels = get_next_clean_batch(50)  
                train_accuracy += accuracy.eval(session=sess,feed_dict={x:batch_images, y_:batch_labels, keep_prob:1.0})
            train_accuracy/=4.0
            if train_accuracy>0.99:
                break
        
    print(path)
    
    train_accuracy = 0
    for i in range(4):      
        batch_images, batch_labels = get_next_clean_batch(50)  
        train_accuracy += accuracy.eval(session=sess,feed_dict={x:batch_images, y_:batch_labels, keep_prob:1.0})
    train_accuracy/=4.0
    print("train accuracy %g"%(train_accuracy))
    
    if not os.path.exists(path+".index"):
        saver.save(sess, path)   
    
    sess.close()
    
    
def pnnir(modelname):
    
    pnnir_1(modelname)
    pnnir_2(modelname)
    pnnir_3(modelname)
    pnnir_23(modelname)
    pnnir_12(modelname)
    pnnir_13(modelname)
    pnnir_123(modelname)


if __name__ ==  '__main__':
    
     
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    modelname = 'patch1'
    pnnir(modelname)
    modelname = 'patch2'
    pnnir(modelname)
    modelname = 'patch3'
    pnnir(modelname)


    modelname = 'feature1'
    pnnir(modelname)
    modelname = 'feature2'
    pnnir(modelname)
    modelname = 'feature3'
    pnnir(modelname)
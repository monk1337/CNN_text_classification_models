import tensorflow as tf

import numpy as np

dummpy_data  = np.random.randint(0,10,[50,33,356])  #batch_size , sequence_length , embedding_dim
input_x = tf.placeholder(shape=[None,None,356],dtype=tf.float32)


with tf.name_scope("convo_first"):
    weight_f = tf.get_variable(name='convo_first_weight', shape=[3,356,100],
                              initializer=tf.random_uniform_initializer(-0.01, 0.01), dtype=tf.float32)
    bias_f = tf.get_variable(name='convo_first_bias', shape=[100], initializer=tf.random_uniform_initializer(-0.01, 0.01),
                           dtype=tf.float32)
    output_f = tf.nn.conv1d(input_x, weight_f, stride=1, padding="SAME")
    h_f = tf.nn.relu(tf.nn.bias_add(output_f, bias_f), name="relu")
    
    
with tf.name_scope("convo_second"):
    weight_s = tf.get_variable(name='convo_second_weight', shape=[4,100,100],
                              initializer=tf.random_uniform_initializer(-0.01, 0.01), dtype=tf.float32)
    bias_s = tf.get_variable(name='convo_second_bias', shape=[100], initializer=tf.random_uniform_initializer(-0.01, 0.01),
                           dtype=tf.float32)
    output_s = tf.nn.conv1d(h_f, weight_s, stride=1, padding="SAME")
    h_s = tf.nn.relu(tf.nn.bias_add(output_s, bias_s), name="relu")
    
    
with tf.name_scope("convo_third"):
    weight_t = tf.get_variable(name='convo_third_weight', shape=[5,100,100],
                              initializer=tf.random_uniform_initializer(-0.01, 0.01), dtype=tf.float32)
    bias_t = tf.get_variable(name='convo_third_bias', shape=[100], initializer=tf.random_uniform_initializer(-0.01, 0.01),
                           dtype=tf.float32)
    output_t = tf.nn.conv1d(h_s, weight_t, stride=1, padding="SAME")
    h_t = tf.nn.relu(tf.nn.bias_add(output_t, bias_t), name="relu")
    
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(h_t,feed_dict={input_x:dummpy_data}).shape)
    
# pooled = tf.nn.pool(h_t,[1, 30, 1, 1],padding='VALID',
#                         name="pool")

B = tf.nn.pool(h_t, [5], 'MAX', 'SAME', strides = [5])

final_output = tf.reshape(B,[tf.shape(input_x)[0],-1])

# final_output --> dense_layer --> softmax ---> prediction


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(final_output,feed_dict={input_x:dummpy_data}).shape)
    
    
    
  

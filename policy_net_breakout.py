import tensorflow as tf
import numpy as np
import os

ACTIONS = [0, 1, 2, 3]

class Policy_net:
    def __init__(self, name: str, env):
        """
        :param name: string
        :param env: gym env
        """
        ob_space = env.observation_space
        act_space = env.action_space
        print("ob_space")
        print(ob_space)
        print("act_space")
        print(act_space)

        # with tf.variable_scope(name):
        with tf.compat.v1.variable_scope(name):
            
            # self.input = tf.placeholder(shape=[None , 84, 84, 4], dtype=tf.uint8, name='input')
            # self.action = tf.placeholder(shape=[None], dtype=tf.float32, name="action")
            self.input =  tf.compat.v1.placeholder(shape=[None , 84, 84, 4], dtype=tf.uint8, name='input')
            self.action =  tf.compat.v1.placeholder(shape=[None], dtype=tf.float32, name="action")
            #self.input_normalized = tf.to_float(self.input) / 255.0
            self.input_normalized = tf.cast(self.input, tf.float32) / 255.0
            
            #print("self.input_normalized shape :" + str(self.input_normalized.shape))
            #print("self.input shape :" + str(self.input.shape))

            #with tf.variable_scope('policy_net'):
            with tf.compat.v1.variable_scope('policy_net'):
                # # three convolutional layers----------------------------------------------------------------
                conv1 = tf.keras.layers.Conv2D(32, 8, 4, activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer(np.sqrt(2.0)))(self.input_normalized)
                conv2 = tf.keras.layers.Conv2D(64, 4, 2, activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer(np.sqrt(2.0)))(conv1)
                conv3 = tf.keras.layers.Conv2D(64, 3, 1, activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer(np.sqrt(2.0)))(conv2)
                
                # conv1 = tf.keras.layers.Conv2D(32, 8, 4, activation=tf.nn.relu)(self.input_normalized)
                # conv2 = tf.keras.layers.Conv2D(64, 4, 2, activation=tf.nn.relu)(conv1)
                # conv3 = tf.keras.layers.Conv2D(64, 3, 1, activation=tf.nn.relu)(conv2)
                
                flattened = tf.keras.layers.Flatten()(conv3)
                # fully connected layer
                fc1 = tf.keras.layers.Dense(512, activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer(np.sqrt(2.0)))(flattened)
                #fc1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(flattened)
                # predicted actions probabilities / policy
                self.act_probs = tf.keras.layers.Dense(len(ACTIONS), activation=None, kernel_initializer=tf.orthogonal_initializer(0.1))(fc1)
                # self.act_probs = tf.keras.layers.Dense(len(ACTIONS), activation=tf.nn.softmax, kernel_initializer=tf.orthogonal_initializer(0.1))(fc1)
                # self.act_probs = tf.keras.layers.Dense(len(ACTIONS), activation=None)(fc1)

            #with tf.variable_scope('value_net'):
            with tf.compat.v1.variable_scope('value_net'):
            
                conv1 = tf.keras.layers.Conv2D(32, 8, 4, activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer(np.sqrt(2.0)))(self.input_normalized)
                conv2 = tf.keras.layers.Conv2D(64, 4, 2, activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer(np.sqrt(2.0)))(conv1)
                conv3 = tf.keras.layers.Conv2D(64, 3, 1, activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer(np.sqrt(2.0)))(conv2)
                
                # conv1 = tf.keras.layers.Conv2D(32, 8, 4, activation=tf.nn.relu)(self.input_normalized)
                # conv2 = tf.keras.layers.Conv2D(64, 4, 2, activation=tf.nn.relu)(conv1)
                # conv3 = tf.keras.layers.Conv2D(64, 3, 1, activation=tf.nn.relu)(conv2)
                
                flattened = tf.keras.layers.Flatten()(conv3)
                fc1 = tf.keras.layers.Dense(512, activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer(np.sqrt(2.0)))(flattened)
                self.v_preds = tf.keras.layers.Dense(1, kernel_initializer=tf.orthogonal_initializer(0.1))(fc1)
                # fc1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(flattened)
                # self.v_preds = tf.keras.layers.Dense(1)(fc1)

            # self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            # self.act_stochastic = tf.multinomial(tf.math.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.random.categorical(tf.math.log(self.act_probs), num_samples=1)           
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            # self.scope = tf.get_variable_scope().name
            self.scope = tf.compat.v1.get_variable_scope().name

    def act(self, obs):
        return tf.get_default_session().run([self.act_stochastic, self.v_preds], feed_dict={self.input_normalized: obs})
        

    def get_action_prob(self, obs):
        return tf.get_default_session().run(self.act_probs, feed_dict={self.input_normalized: obs})

    def get_variables(self):
        # return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
        # return tf.compat.v1.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, self.scope)


    def get_trainable_variables(self):
        # return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        # return tf.compat.v1.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, self.scope)


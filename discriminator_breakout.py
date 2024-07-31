import tensorflow as tf
import numpy as np 

obs_state_shape = [84,84,4]

class Discriminator:
    def __init__(self, env):
        """
        :param env:
        Output of this Discriminator is reward for learning agent. Not the cost.
        Because discriminator predicts  P(expert|s,a) = 1 - P(agent|s,a).
        """

        with tf.variable_scope('discriminator'):
            self.scope = tf.get_variable_scope().name
            self.expert_s = tf.placeholder(dtype=tf.float32, shape=[None] + obs_state_shape)
            self.expert_a = tf.placeholder(dtype=tf.int32, shape=[None])
            self.agent_s = tf.placeholder(dtype=tf.float32, shape=[None] + obs_state_shape)
            self.agent_a = tf.placeholder(dtype=tf.int32, shape=[None])
            
            with tf.variable_scope('network') as network_scope:
                prob_1 = self.construct_network(obs_input= self.expert_s , action_input = self.expert_a , env = env)
                network_scope.reuse_variables()  # share parameter
                prob_2 = self.construct_network(obs_input= self.agent_s , action_input = self.agent_a ,  env = env)

            with tf.variable_scope('loss'):
                loss_expert = tf.reduce_mean(tf.log(tf.clip_by_value(prob_1, 0.01, 1)))
                loss_agent = tf.reduce_mean(tf.log(tf.clip_by_value(1 - prob_2, 0.01, 1)))
                loss = loss_expert + loss_agent
                loss = -loss
                tf.summary.scalar('discriminator', loss)

            #optimizer = tf.train.AdamOptimizer()
            optimizer = tf.train.AdamOptimizer(learning_rate= 1e-5, epsilon=1e-5)
            self.train_op = optimizer.minimize(loss)

            self.rewards = tf.log(tf.clip_by_value(prob_2, 1e-10, 1))  # log(P(expert|s,a)) larger is better for agent

    def construct_network(self, obs_input , action_input , env):
        action_a_one_hot = tf.one_hot(action_input, depth=env.action_space.n)
        # add noise for stabilise training
        action_a_one_hot += tf.random_normal(tf.shape(action_a_one_hot), mean=0.2, stddev=0.1, dtype=tf.float32)/1.2

        #ACTIVATION = activation=tf.nn.leaky_relu                                    
        conv1 = tf.keras.layers.Conv2D(32, 8, 4, activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer(np.sqrt(2.0)))(obs_input)
        conv2 = tf.keras.layers.Conv2D(64, 4, 2, activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer(np.sqrt(2.0)))(conv1)
        conv3 = tf.keras.layers.Conv2D(64, 3, 1, activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer(np.sqrt(2.0)))(conv2)
        # conv1 = tf.keras.layers.Conv2D(32, 8, 4, activation=tf.nn.relu)(obs_input)
        # conv2 = tf.keras.layers.Conv2D(64, 4, 2, activation=tf.nn.relu)(conv1)
        # conv3 = tf.keras.layers.Conv2D(64, 3, 1, activation=tf.nn.relu)(conv2)
        flattened = tf.keras.layers.Flatten()(conv3)
        concatenated_s_a = tf.concat([flattened, action_a_one_hot], axis=-1)
        #concatenated_s_a = tf.concat([flattened, action_a_one_hot], axis=-0)
        #concatenated_s_a = tf.concat([flattened, action_a_one_hot], axis=1)
        fc1 = tf.keras.layers.Dense(512, activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer(np.sqrt(2.0)))(concatenated_s_a)
        #fc1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(concatenated_s_a)
        #prob = tf.keras.layers.Dense(1, activation = tf.keras.activations.sigmoid , kernel_initializer=tf.orthogonal_initializer(0.1))(fc1)
        prob = tf.keras.layers.Dense(1, activation = None , kernel_initializer=tf.orthogonal_initializer(0.1))(fc1)
        #prob = tf.keras.layers.Dense(1, activation = None )(fc1)
        return prob

    def train(self, expert_s, expert_a, agent_s, agent_a):
        return tf.get_default_session().run(self.train_op, feed_dict={self.expert_s: expert_s,
                                                                      self.expert_a: expert_a,
                                                                      self.agent_s: agent_s,
                                                                      self.agent_a: agent_a})

    def get_rewards(self, agent_s, agent_a):
        return tf.get_default_session().run(self.rewards, feed_dict={self.agent_s: agent_s,
                                                                     self.agent_a: agent_a})

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


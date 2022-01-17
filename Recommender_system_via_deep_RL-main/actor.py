import tensorflow as tf
import numpy as np


class ActorNetwork(tf.keras.Model):
    def __init__(self, embedding_dim, hidden_dim):
        super(ActorNetwork, self).__init__()
        self.inputs = tf.keras.layers.InputLayer(name='input_layer', input_shape=(3 * embedding_dim,))
        self.fc1 = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(embedding_dim, activation='tanh')
        ])

    def call(self, x):
        x = self.inputs(x)
        return self.fc1(x)


class Actor(object):

    def __init__(self, embedding_dim, hidden_dim, learning_rate, state_size, tau):

        self.embedding_dim = embedding_dim
        self.state_size = state_size

        # 两个网络一个输出均值一个输出log方差，其实用多输出网络也一样，但本质没什么不同
        self.mean_network = ActorNetwork(embedding_dim, hidden_dim)
        self.logvar_network = ActorNetwork(embedding_dim, hidden_dim)
        self.target_mean_network = ActorNetwork(embedding_dim, hidden_dim)
        self.target_logvar_network = ActorNetwork(embedding_dim, hidden_dim)
        # 优化器
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        # 소프트 타겟 네트워크 업데이트 하이퍼파라미터 soft target network update hyperparameter
        self.tau = tau

    def build_networks(self):
        # 네트워크들 빌딩 / Build networks
        self.mean_network(np.zeros((1, 3*self.embedding_dim)))
        self.target_mean_network(np.zeros((1, 3*self.embedding_dim)))
        self.logvar_network(np.zeros((1, 3*self.embedding_dim)))
        self.target_logvar_network(np.zeros((1, 3*self.embedding_dim)))



    def update_target_network(self):
        # target网络更新
        cm_theta, tm_theta = self.mean_network.get_weights(), self.target_mean_network.get_weights()
        cl_theta, tl_theta = self.mean_network.get_weights(), self.target_mean_network.get_weights()
        for i in range(len(cm_theta)):
            tm_theta[i] = self.tau * cm_theta[i] + (1 - self.tau) * tm_theta[i]
        self.target_mean_network.set_weights(tm_theta)
        for k in range(len(cl_theta)):
            tl_theta[i] = self.tau * cl_theta[i] + (1 - self.tau) * tl_theta[i]
        self.target_logvar_network.set_weights(tl_theta)

    def train(self, states, dq_das_m,dq_das_l):
        with tf.GradientTape(persistent=True,watch_accessed_variables=True) as g:
            means,logvars = self.mean_network(states),self.logvar_network(states)
            dj_dtheta_m = g.gradient(means, self.mean_network.trainable_weights, -dq_das_m)
            dj_dtheta_l = g.gradient(logvars, self.logvar_network.trainable_weights, -dq_das_l)
            grads_m = zip(dj_dtheta_m, self.mean_network.trainable_weights)
            grads_l = zip(dj_dtheta_l, self.logvar_network.trainable_weights)
            self.optimizer.apply_gradients(grads_m)
            self.optimizer.apply_gradients(grads_l)

    def save_weights(self, mean_path,logvar_path):
        self.mean_network.save_weights(mean_path)
        self.logvar_network.save_weights(logvar_path)

    def load_weights(self, mean_path,logvar_path):
        self.mean_network.load_weights(mean_path)
        self.logvar_network.load_weights(logvar_path)
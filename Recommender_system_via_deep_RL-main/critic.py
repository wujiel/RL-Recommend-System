import tensorflow as tf
import numpy as np

class CriticNetwork(tf.keras.Model):
    def __init__(self, state_dim,action_dim,hidden_dim):
        super(CriticNetwork, self).__init__()
        self.inputs = tf.keras.layers.InputLayer(input_shape=(action_dim,action_dim, state_dim))
        self.fc1 = tf.keras.layers.Dense(action_dim, activation = 'relu')
        self.concat = tf.keras.layers.Concatenate()
        self.fc2 = tf.keras.layers.Dense(hidden_dim, activation = 'relu')
        self.fc3 = tf.keras.layers.Dense(hidden_dim, activation = 'relu')
        self.out = tf.keras.layers.Dense(1, activation = 'linear')
        
    def call(self, x):
        # 第一层处理state
        s = self.fc1(x[2])
        # 连接上均值和方差
        s = self.concat([x[0],x[1],s])
        s = self.fc2(s)
        s = self.fc3(s)
        return self.out(s)


class SacCriticNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(SacCriticNetwork, self).__init__()
        self.inputs = tf.keras.layers.InputLayer(input_shape=(action_dim, action_dim, state_dim))
        self.fc1 = tf.keras.layers.Dense(action_dim, activation='relu')
        self.concat = tf.keras.layers.Concatenate()
        self.fc2 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.fc3 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.out = tf.keras.layers.Dense(1, activation='linear')

    def call(self, x):
        # 第一层处理state
        s = self.fc1(x[2])
        # 连接上均值和方差
        s = self.concat([x[0], x[1], s])
        s = self.fc2(s)
        s = self.fc3(s)
        return self.out(s)


class SacCritic(object):

    def __init__(self, hidden_dim, learning_rate, state_dim,action_dim, target_network_update_rate):
        # 状态向量的维度
        self.state_dim = state_dim
        # 动作向量的维度
        self.action_dim = action_dim
        # 隐藏层的维度
        self.hidden_dim = hidden_dim
        # critic network / target network
        self.network = SacCriticNetwork(state_dim=state_dim,action_dim=action_dim,hidden_dim=hidden_dim)
        self.target_network = SacCriticNetwork(state_dim=state_dim,action_dim=action_dim,hidden_dim=hidden_dim)
        # build并summary
        self.network([np.zeros((1, self.action_dim)), np.zeros((1, self.action_dim)),np.zeros((1, state_dim))])
        self.target_network([np.zeros((1, self.action_dim)), np.zeros((1, self.action_dim)),np.zeros((1, state_dim))])
        print("critic_network已创建")
        self.network.summary()
        print("target_critic_network已创建")
        self.target_network.summary()
        # 优化器
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # 损失函数
        self.loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        # 网络编译
        self.network.compile(self.optimizer, self.loss)
        #  soft target network update hyperparameter
        self.target_network_update_rate = target_network_update_rate

    # 给状态——动作对打分
    def estimate_state_action(self,state):
        self.network.call(state)
    # 给状态——动作对打分（double Q）
    def target_estimate_state_action(self,state):
        self.target_network.call(state)

    # 目标网络更新
    def update_target_network(self):
        c_omega = self.network.get_weights()
        t_omega = self.target_network.get_weights()
        for i in range(len(c_omega)):
            t_omega[i] = self.target_network_update_rate * c_omega[i] + (1 - self.target_network_update_rate) * t_omega[i]
        self.target_network.set_weights(t_omega)

    #  q对a求导（a之后对actor的参数求导让actor朝着最大化q的方向优化）
    def dq_da(self, inputs):
        means = inputs[0]
        logvars = inputs[1]
        states = inputs[2]
        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as g:
            # 转为tensor才能求梯度
            means = tf.convert_to_tensor(means)
            logvars = tf.convert_to_tensor(logvars)
            g.watch([means, logvars])
            qualities = self.network([means, logvars, states])
        q_grads1 = g.gradient(qualities, means)
        q_grads2 = g.gradient(qualities, logvars)
        # print("shit japanese")
        return q_grads1, q_grads2

    # 训练一轮（拟合q）
    def train(self, inputs, td_targets, weight_batch):
        # 单纯的把target作为label去拟合
        weight_batch = tf.convert_to_tensor(weight_batch, dtype=tf.float32)
        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as g:
            outputs = self.network(inputs)
            loss = self.loss(td_targets, outputs)
            weighted_loss = tf.reduce_mean(loss * weight_batch)
        dl_domega = g.gradient(weighted_loss, self.network.trainable_weights)
        grads = zip(dl_domega, self.network.trainable_weights)
        self.optimizer.apply_gradients(grads)
        return weighted_loss

    # 保存权重
    def save_weights(self, path):
        self.network.save_weights(path)

    # 加载权重
    def load_weights(self, path):
        self.network.load_weights(path)

class DdpgCritic(object):

    def __init__(self, hidden_dim, learning_rate, embedding_dim, target_network_update_rate):
        self.embedding_dim = embedding_dim

        # critic network / target network
        self.network = CriticNetwork(embedding_dim, hidden_dim)
        self.target_network = CriticNetwork(embedding_dim, hidden_dim)
        # 优化器
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # 损失函数
        self.loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        #  soft target network update hyperparameter
        self.target_network_update_rate = target_network_update_rate

    def build_networks(self):
        self.network([np.zeros((1, self.embedding_dim)), np.zeros((1, self.embedding_dim)),
                      np.zeros((1, 3 * self.embedding_dim))])
        self.target_network([np.zeros((1, self.embedding_dim)), np.zeros((1, self.embedding_dim)),
                             np.zeros((1, 3 * self.embedding_dim))])
        self.network.compile(self.optimizer, self.loss)

    def update_target_network(self):
        c_omega = self.network.get_weights()
        t_omega = self.target_network.get_weights()
        for i in range(len(c_omega)):
            t_omega[i] = self.target_network_update_rate * c_omega[i] + (1 - self.target_network_update_rate) * t_omega[i]
        self.target_network.set_weights(t_omega)

    def dq_da(self, inputs):
        means = inputs[0]
        logvars = inputs[1]
        states = inputs[2]
        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as g:
            # 转为tensor才能求梯度
            means = tf.convert_to_tensor(means)
            logvars = tf.convert_to_tensor(logvars)

            g.watch([means, logvars])
            qualities = self.network([means, logvars, states])
        q_grads1 = g.gradient(qualities, means)
        q_grads2 = g.gradient(qualities, logvars)
        # print("shit japanese")
        return q_grads1, q_grads2

    def train(self, inputs, td_targets, weight_batch):
        # 单纯的把target作为label去拟合
        weight_batch = tf.convert_to_tensor(weight_batch, dtype=tf.float32)
        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as g:
            outputs = self.network(inputs)
            loss = self.loss(td_targets, outputs)
            weighted_loss = tf.reduce_mean(loss * weight_batch)
        dl_domega = g.gradient(weighted_loss, self.network.trainable_weights)
        grads = zip(dl_domega, self.network.trainable_weights)
        self.optimizer.apply_gradients(grads)
        return weighted_loss

    def train_on_batch(self, inputs, td_targets, weight_batch):
        loss = self.network.train_on_batch(inputs, td_targets, sample_weight=weight_batch)
        return loss

    def save_weights(self, path):
        self.network.save_weights(path)

    def load_weights(self, path):
        self.network.load_weights(path)

class Critic(object):

    def __init__(self, hidden_dim, learning_rate, embedding_dim, target_network_update_rate):
        self.embedding_dim = embedding_dim

        # critic network / target network
        self.network = CriticNetwork(embedding_dim, hidden_dim)
        self.target_network = CriticNetwork(embedding_dim, hidden_dim)
        # 优化器
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # 损失函数
        self.loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        #  soft target network update hyperparameter
        self.target_network_update_rate = target_network_update_rate

    def build_networks(self):
        self.network([np.zeros((1, self.embedding_dim)), np.zeros((1, self.embedding_dim)),
                      np.zeros((1, 3 * self.embedding_dim))])
        self.target_network([np.zeros((1, self.embedding_dim)), np.zeros((1, self.embedding_dim)),
                             np.zeros((1, 3 * self.embedding_dim))])
        self.network.compile(self.optimizer, self.loss)

    def update_target_network(self):
        c_omega = self.network.get_weights()
        t_omega = self.target_network.get_weights()
        for i in range(len(c_omega)):
            t_omega[i] = self.target_network_update_rate * c_omega[i] + (1 - self.target_network_update_rate) * t_omega[i]
        self.target_network.set_weights(t_omega)

    def dq_da(self, inputs):
        means = inputs[0]
        logvars = inputs[1]
        states = inputs[2]
        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as g:
            # 转为tensor才能求梯度
            means = tf.convert_to_tensor(means)
            logvars = tf.convert_to_tensor(logvars)

            g.watch([means, logvars])
            qualities = self.network([means, logvars, states])
        q_grads1 = g.gradient(qualities, means)
        q_grads2 = g.gradient(qualities, logvars)
        # print("shit japanese")
        return q_grads1, q_grads2

    def train(self, inputs, td_targets, weight_batch):
        # 单纯的把target作为label去拟合
        weight_batch = tf.convert_to_tensor(weight_batch, dtype=tf.float32)
        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as g:
            outputs = self.network(inputs)
            loss = self.loss(td_targets, outputs)
            weighted_loss = tf.reduce_mean(loss * weight_batch)
        dl_domega = g.gradient(weighted_loss, self.network.trainable_weights)
        grads = zip(dl_domega, self.network.trainable_weights)
        self.optimizer.apply_gradients(grads)
        return weighted_loss

    def train_on_batch(self, inputs, td_targets, weight_batch):
        loss = self.network.train_on_batch(inputs, td_targets, sample_weight=weight_batch)
        return loss

    def save_weights(self, path):
        self.network.save_weights(path)

    def load_weights(self, path):
        self.network.load_weights(path)
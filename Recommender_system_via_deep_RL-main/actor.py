import keras
import tensorflow
import numpy

# actor网络，输入state，返回action
class ActorNetwork(tensorflow.keras.Model):
    def __init__(self, state_dim,action_dim, hidden_dim):
        super(ActorNetwork, self).__init__()
        self.inputs = tensorflow.keras.layers.InputLayer(name='input_layer', input_shape=(state_dim,))
        self.fc1 = tensorflow.keras.Sequential([
            tensorflow.keras.layers.Dense(hidden_dim, activation='relu'),
            tensorflow.keras.layers.Dense(hidden_dim, activation='relu'),
            tensorflow.keras.layers.Dense(action_dim, activation='tanh')
        ])

    def call(self, x):
        x = self.inputs(x)
        return self.fc1(x)

class SacActor(object):
    def __init__(self, state_dim, hidden_dim,action_dim, learning_rate, target_network_update_rate):
        # 两个网络一个输出均值一个输出log方差，其实用多输出网络也一样，但本质没什么不同
        # 均值网络
        self.mean_network = ActorNetwork(state_dim=state_dim,action_dim=action_dim,hidden_dim=hidden_dim)
        # 方差网络
        self.logvar_network = ActorNetwork(state_dim=state_dim,action_dim=action_dim,hidden_dim=hidden_dim)
        # 目标均值网络
        self.target_mean_network = ActorNetwork(state_dim=state_dim,action_dim=action_dim,hidden_dim=hidden_dim)
        # 目标方差网络
        self.target_logvar_network = ActorNetwork(state_dim=state_dim,action_dim=action_dim,hidden_dim=hidden_dim)
        #  Build networks并summary（）
        self.mean_network(numpy.zeros((1, state_dim)))
        self.target_mean_network(numpy.zeros((1, state_dim)))
        self.logvar_network(numpy.zeros((1, state_dim)))
        self.target_logvar_network(numpy.zeros((1, state_dim)))
        print("均值网络已创建")
        self.mean_network.summary()
        print("目标均值网络已创建")
        self.target_mean_network.summary()
        print("方差网络已创建")
        self.logvar_network.summary()
        print("目标方差网络已创建")
        self.target_logvar_network.summary()
        # 优化器
        self.optimizer = tensorflow.keras.optimizers.Adam(learning_rate)
        # soft target network update hyperparameter
        self.target_network_update_rate = target_network_update_rate

    def act(self, state):
        mean, log_var = self.mean_network(state), self.logvar_network(state)
        action = self._action_from_distribution(mean=mean,log_var=log_var)
        return action, mean, log_var

    def target_act(self,state):
        mean, log_var = self.target_mean_network(state), self.target_logvar_network(state)
        action = self._action_from_distribution(mean=mean, log_var=log_var)
        return action, mean, log_var

    def _action_from_distribution(self,mean,log_var):
        # tensor,numpy互转
        mean_a = keras.backend.eval(mean)
        log_var_a = keras.backend.eval(log_var)
        mean0 = mean_a[0]
        log_var0 = log_var_a[0]
        # 协方差矩阵
        var = numpy.exp(log_var0)
        cov = numpy.diag(var)
        x = numpy.random.multivariate_normal(mean0, cov)
        x = tensorflow.tanh(x)
        x = numpy.array([x])
        action = tensorflow.constant(x, tensorflow.float32)
        return action

    def update_target_network(self):
        # target网络更新
        cm_theta, tm_theta = self.mean_network.get_weights(), self.target_mean_network.get_weights()
        cl_theta, tl_theta = self.mean_network.get_weights(), self.target_mean_network.get_weights()
        for i in range(len(cm_theta)):
            tm_theta[i] = self.target_network_update_rate * cm_theta[i] + (1 - self.target_network_update_rate) * tm_theta[i]
        self.target_mean_network.set_weights(tm_theta)
        for k in range(len(cl_theta)):
            tl_theta[i] = self.target_network_update_rate * cl_theta[i] + (1 - self.target_network_update_rate) * tl_theta[i]
        self.target_logvar_network.set_weights(tl_theta)

    def train(self, states, dq_das_m,dq_das_l):
        with tensorflow.GradientTape(persistent=True,watch_accessed_variables=True) as g:
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

class DdpgActor(object):
    def __init__(self, state_dim, hidden_dim,action_dim, learning_rate, target_network_update_rate):
        # 直接输出action
        self.network = ActorNetwork(state_dim=state_dim,action_dim=action_dim,hidden_dim=hidden_dim)
        self.target_network = ActorNetwork(state_dim=state_dim,action_dim=action_dim,hidden_dim=hidden_dim)
        #  Build networks并summary（）
        self.network(numpy.zeros((1, state_dim)))
        self.target_network(numpy.zeros((1, state_dim)))
        print("ddpg网络已创建")
        self.network.summary()
        print("ddpg目标网络已创建")
        self.target_network.summary()
        # 优化器
        self.optimizer = tensorflow.keras.optimizers.Adam(learning_rate)
        # soft target network update hyperparameter
        self.target_network_update_rate = target_network_update_rate

    def act(self, state):
        action = self.network.call(state)
        return action

    def target_act(self, state):
        action = self.target_network.call(state)
        return action

    def update_target_network(self):
        # target网络更新
        c_theta, t_theta = self.network.get_weights(), self.target_network.get_weights()
        for i in range(len(c_theta)):
            t_theta[i] = self.target_network_update_rate * c_theta[i] + (1 - self.target_network_update_rate) * t_theta[i]
        self.target_network.set_weights(t_theta)

    def train(self, states, dq_das):
        with tensorflow.GradientTape(persistent=True,watch_accessed_variables=True) as g:
            actions = self.network(states)
            dj_dtheta = g.gradient(actions, self.network.trainable_weights, -dq_das)
            grads_m = zip(dj_dtheta, self.network.trainable_weights)
            self.optimizer.apply_gradients(grads_m)

    def save_weights(self,path):
        self.network.save_weights(path)

    def load_weights(self, path):
        self.network.load_weights(path)
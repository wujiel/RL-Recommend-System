import keras

from replay_buffer import PriorityExperienceReplay
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.gen_math_ops import Exp

from actor import Actor
from critic import Critic
from replay_memory import ReplayMemory
from embedding import MovieGenreEmbedding, UserMovieEmbedding
from state_representation import DRRAveStateRepresentation

import matplotlib.pyplot as plt
import wandb


class DRRAgent:
    
    def __init__(self, env, users_num, items_num, state_size, is_test=False, use_wandb=False):
        
        self.env = env

        self.users_num = users_num
        self.items_num = items_num
        
        self.embedding_dim = 100
        self.actor_hidden_dim = 128
        self.actor_learning_rate = 0.001
        self.critic_hidden_dim = 128
        self.critic_learning_rate = 0.001
        self.discount_factor = 0.9
        self.tau = 0.001

        self.replay_memory_size = 1000000
        self.batch_size = 32
        
        self.actor = Actor(self.embedding_dim, self.actor_hidden_dim, self.actor_learning_rate, state_size, self.tau)
        self.critic = Critic(self.critic_hidden_dim, self.critic_learning_rate, self.embedding_dim, self.tau)
        
        # self.m_embedding_network = MovieGenreEmbedding(items_num, 19, self.embedding_dim)
        # self.m_embedding_network([np.zeros((1,)),np.zeros((1,))])
        # self.m_embedding_network.load_weights('/home/diominor/Workspace/DRR/save_weights/m_g_model_weights.h5')

        self.embedding_network = UserMovieEmbedding(users_num, items_num, self.embedding_dim)
        self.embedding_network([np.zeros((1,)),np.zeros((1,))])

        self.embedding_network.load_weights(r'save_weights\user_movie_embedding_case4.h5')

        self.srm_ave = DRRAveStateRepresentation(self.embedding_dim)
        self.srm_ave([np.zeros((1, 100,)),np.zeros((1,state_size, 100))])

        # PER
        self.buffer = PriorityExperienceReplay(self.replay_memory_size, self.embedding_dim)
        self.epsilon_for_priority = 1e-6

        # ε-탐욕 탐색 하이퍼파라미터 ε-greedy exploration hyperparameter
        self.epsilon = 1.
        self.epsilon_decay = (self.epsilon - 0.1)/500000
        self.std = 1.5

        self.is_test = is_test

        # wandb
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="drr", 
            entity="diominor",
            config={'users_num':users_num,
            'items_num' : items_num,
            'state_size' : state_size,
            'embedding_dim' : self.embedding_dim,
            'actor_hidden_dim' : self.actor_hidden_dim,
            'actor_learning_rate' : self.actor_learning_rate,
            'critic_hidden_dim' : self.critic_hidden_dim,
            'critic_learning_rate' : self.critic_learning_rate,
            'discount_factor' : self.discount_factor,
            'tau' : self.tau,
            'replay_memory_size' : self.replay_memory_size,
            'batch_size' : self.batch_size,
            'std_for_exploration': self.std})

    def calculate_td_target(self, rewards, q_values, dones):
        y_t = np.copy(q_values)
        for i in range(q_values.shape[0]):
            y_t[i] = rewards[i] + (1 - dones[i])*(self.discount_factor * q_values[i])
        return y_t

    # 熵的计算
    def calculate_entropy(self,states):
        means = self.actor.mean_network(states)
        log_vars = self.actor.logvar_network(states)
        # 转成numpy数组
        means = keras.backend.eval(means)
        log_vars = keras.backend.eval(log_vars)
        entropies = []
        for i in range(self.batch_size):
            mean = means[i]
            log_var = log_vars[i]
            var = np.exp(log_var)
            cov = np.diag(var)
            # 重采样
            x = np.random.multivariate_normal(mean, cov)
            # 逆协方差矩阵
            cov_inverse = np.linalg.inv(cov)
            alpha = x-mean
            temp1 = np.dot(alpha,cov_inverse)
            temp2 = np.dot(temp1,alpha)/2



            det = np.linalg.det(cov)
            # 多元高斯分布的熵,sac作者代码里用的概率密度的对数。
            # 我对这里保持看法，因为如果计算熵涉及sample取值，而sample本身是随机取的，难道让随机性影响优化结果吗？
            # 分布的熵
            # entropy1 = np.log(det)/2+(self.embedding_dim/2)*(np.log(2*np.pi)+1)
            # 直接-logP，概率密度值
            entropy = np.log( (np.pi**50) * (det**(1/2)) ) + temp2
            # 归一化到一维（但就算是一维-log概率密度值也是可能大于1的）
            entropies.append(0.3*entropy/100)
        entropies = np.array([entropies])
        entropies = entropies.reshape((self.batch_size,1))
        entropies = tf.constant(entropies, tf.float32)

        return entropies
















    def recommend_item(self, action, recommended_items, top_k=False, items_ids=None):
        if items_ids == None:
            items_ids = np.array(list(set(i for i in range(self.items_num)) - recommended_items))

            items_ebs = self.embedding_network.get_layer('movie_embedding')(items_ids)
        action = tf.transpose(action, perm=(1,0))
        if top_k:
            # arg函数返回的是索引
            item_indice = np.argsort(tf.transpose(tf.keras.backend.dot(items_ebs, action), perm=(1,0)))[0][-top_k:]
            return items_ids[item_indice]
        else:
            rank = tf.keras.backend.dot(items_ebs, action)
            item_idx = np.argmax(rank)
            return items_ids[item_idx]
        
    def train(self, max_episode_num, top_k=False, load_model=False):
        # 타겟 네트워크들 초기화
        self.actor.update_target_network()
        self.critic.update_target_network()

        if load_model:
            self.load_model(r"save_weights\actor_50000.h5", r"save_weights\critic_50000.h5")
            print('Completely load weights!')

        episodic_precision_history = []

        for episode in range(max_episode_num):
            # episodic reward 리셋
            episode_reward = 0
            correct_count = 0
            steps = 0
            q_loss = 0
            mean_action = 0
            # Environment 리셋
            user_id, items_ids, done = self.env.reset()
            # print(f'user_id : {user_id}, rated_items_length:{len(self.env.user_items)}')
            # print('items : ', self.env.get_items_names(items_ids))
            while not done:
                
                # Observe current state & Find action
                ## Embedding 해주기
                user_eb = self.embedding_network.get_layer('user_embedding')(np.array(user_id))
                items_eb = self.embedding_network.get_layer('movie_embedding')(np.array(items_ids))
                # items_eb = self.m_embedding_network.get_layer('movie_embedding')(np.array(items_ids))
                ## 状态生成
                state = self.srm_ave([np.expand_dims(user_eb, axis=0), np.expand_dims(items_eb, axis=0)])

                ## Action(ranking score)动作生成
                # action = self.actor.network(state)
                mean, log_var = self.actor.mean_network(state),self.actor.logvar_network(state)
                mean_a = keras.backend.eval(mean)
                log_var_a = keras.backend.eval(log_var)
                mean0 = mean_a[0]
                log_var0 = log_var_a[0]
                var = np.exp(log_var0)
                cov = np.diag(var)

                x = np.random.multivariate_normal(mean0, cov)
                x = np.array([x])
                action = tf.constant(x, tf.float32)




                ## ε-greedy exploration  ε贪婪探索
                # if self.epsilon > np.random.uniform() and not self.is_test:
                #     self.epsilon -= self.epsilon_decay
                #     action += np.random.normal(0,self.std,size=action.shape)

                ## Item 추천
                recommended_item = self.recommend_item(action, self.env.recommended_items, top_k=top_k)
                
                # Calculate reward & observe new state (in env)
                ## Step
                next_items_ids, reward, done, _ = self.env.step(recommended_item, top_k=top_k)
                if top_k:
                    reward = np.sum(reward)

                # get next_state
                next_items_eb = self.embedding_network.get_layer('movie_embedding')(np.array(next_items_ids))
                # next_items_eb = self.m_embedding_network.get_layer('movie_embedding')(np.array(next_items_ids))
                next_state = self.srm_ave([np.expand_dims(user_eb, axis=0), np.expand_dims(next_items_eb, axis=0)])

                # 把经历加入buffer里有，之后训练网络的时候进行经历重放
                self.buffer.append(state, action,mean,log_var, reward, next_state, done)
                
                if self.buffer.crt_idx > 1 or self.buffer.is_full:
                    # Sample a minibatch
                    batch_states, batch_actions,batch_means,batch_logvars, batch_rewards, batch_next_states, batch_dones, weight_batch, index_batch = self.buffer.sample(self.batch_size)

                    # Set TD targets
                    target_next_mean,target_next_logvar= self.actor.target_mean_network(batch_next_states),self.actor.target_logvar_network(batch_next_states)
                    # critic的网络和目标网络分别给下一状态动作打分以作为未来期望奖励
                    qs = self.critic.network([target_next_mean,target_next_logvar, batch_next_states])
                    target_qs = self.critic.target_network([target_next_mean,target_next_logvar, batch_next_states])
                    min_qs = tf.raw_ops.Min(input=tf.concat([target_qs, qs], axis=1), axis=1, keep_dims=True) # Double Q method
                    # 计算熵
                    entropy = self.calculate_entropy(batch_states)

                    # newQ = r+γ*nextQ
                    # nextQ = t_critic(next_s,next_a)
                    # next_a = t_actor(next_s)
                    td_targets = self.calculate_td_target(batch_rewards, min_qs, batch_dones) + entropy
        
                    # Update priority
                    for (p, i) in zip(td_targets, index_batch):
                        self.buffer.update_priority(abs(p[0]) + self.epsilon_for_priority, i)

                    # print(weight_batch.shape)
                    # print(td_targets.shape)
                    # raise Exception
                    # Update critic network
                    q_loss += self.critic.train([batch_means,batch_logvars, batch_states], td_targets, weight_batch)
                    
                    # Update actor network

                    # policy_loss = -entropy- min_qs
                    # tf.train.AdamOptimizer(self.actor_learning_rate).minimize(
                    #     loss=policy_loss,
                    #     var_list=self.actor.network.get_params_internal()
                    # )

                    # q对a求导再a对actor的参数求导最后得到分值对actor网络参数的导数，朝着使分值增大的方向优化，传反梯度
                    s_grads1,s_grads2 = self.critic.dq_da([batch_means,batch_logvars, batch_states])

                    self.actor.train(batch_states, s_grads1,s_grads1)
                    self.actor.update_target_network()
                    self.critic.update_target_network()

                items_ids = next_items_ids
                episode_reward += reward
                mean_action += np.sum(action[0])/(len(action[0]))
                steps += 1

                if reward > 0:
                    correct_count += 1
                
                print(f'recommended items : {len(self.env.recommended_items)},  epsilon : {self.epsilon:0.3f}, reward : {reward:+}', end='\r')

                if done:
                    print()
                    precision = int(correct_count/steps * 100)
                    print(f'{episode}/{max_episode_num}, precision : {precision:2}%, total_reward:{episode_reward}, q_loss : {q_loss/steps}, mean_action : {mean_action/steps}')
                    if self.use_wandb:
                        wandb.log({'precision':precision, 'total_reward':episode_reward, 'epsilone': self.epsilon, 'q_loss' : q_loss/steps, 'mean_action' : mean_action/steps})
                    episodic_precision_history.append(precision)
             
            if (episode+1)%50 == 0:
                plt.plot(episodic_precision_history[len(episodic_precision_history)-50:len(episodic_precision_history)])
                plt.savefig(r'precisionimages1\training_precision__'+str(episode+1)+'__%_top_5.png')
                plt.clf()

            if (episode+1)%100 == 0:
                self.save_model(r'weights\actor_mean'+str(episode+1)+'_fixed.h5',r'weights\actor_logvar'+str(episode+1)+'_fixed.h5',
                                r'weights\critic_'+str(episode+1)+'_fixed.h5')

    def save_model(self, actor_mean_path,actor_logvar_path, critic_path):
        self.actor.save_weights(actor_mean_path,actor_logvar_path)
        self.critic.save_weights(critic_path)
        
    def load_model(self, actor_path, critic_path):
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)
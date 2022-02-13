import copy
import keras
import numpy.random
import tensorflow
import numpy
import matplotlib.pyplot as plt

'''
训练方式：选出历史记录中的好评记录
抛开embedding，本质上是一个在线模型
'''

'''作为一个系统来说应该只从用户那儿获取评价'''
class RecommendSystem:

    def __init__(self, env, representation_model,sac_actor,sac_critic, sac_buffer):
        #模拟环境
        self.env = env
        # 状态表示模块
        self.representation_model = representation_model
        # actor
        self.sac_actor = sac_actor
        # sac_critic
        self.sac_critic = sac_critic
        # sac_buffer(经历重放的容器)
        self.sac_buffer = sac_buffer
        # 从buffer里取出的批量大小
        self.batch_size = 32
        # 未来期望奖励的衰减γ值
        self.discount_factor = 0.9
        # buffer容器的
        self.epsilon_for_priority = 1e-6

        # 推荐空间，1—D array
        # sac方式训练时的推荐空间
        self.sac_recommend_space = copy.deepcopy(self.env.recommend_space_train)
        # Ddpg方式训练时的推荐空间
        self.ddpg_recommend_space = copy.deepcopy(self.env.recommend_space_train)
        # 随机方式训练时的推荐空间
        self.random_recommend_space = copy.deepcopy(self.env.recommend_space_train)

        # 历史记录,字典类型，两个列表，分别是item——id和评分
        # 假设初始记录有10条
        # sac方式训练时的历史记录
        self.sac_recommended_items = copy.deepcopy(self.env.recommended_items_init)
        # Ddpg方式训练时的历史记录
        self.ddpg_recommended_items = copy.deepcopy(self.env.recommended_items_init)
        # 随机方式训练时的历史记录
        self.random_recommended_items = copy.deepcopy(self.env.recommended_items_init)

    # 根据用户id和推荐空间做出推荐，核心方法。返回推荐项目列表，直接传给环境获取真实评价
    def recommend_item_sac(self, user_id, history, top_k=False):
        self.representation_model.user_id = user_id
        self.representation_model.history = history
        sac_state = self.representation_model.get_state()

        sac_action, mean, log_var = self.sac_actor.act(state=sac_state)
        # random_action = numpy.random.normal(0,1,size=sac_action.shape)
        # 待推荐项目的空间，可以考虑训练和应用策略分开
        recommend_space_ebs = self.representation_model.embedding_network.get_layer('movie_embedding')(self.sac_recommend_space)
        action_t = tensorflow.transpose(sac_action, perm=(1, 0))
        if top_k:
            # arg函数返回的是索引
            item_index = numpy.argsort(
                tensorflow.transpose(tensorflow.keras.backend.dot(recommend_space_ebs, action_t), perm=(1, 0)))[0][
                         -top_k:]
            return self.sac_recommend_space[item_index], sac_action, mean, log_var,sac_state
        else:
            rank = tensorflow.keras.backend.dot(recommend_space_ebs, action_t)
            item_idx = numpy.argmax(rank)
            sac_item_id = self.sac_recommend_space[item_idx]
            return sac_item_id, sac_action, mean, log_var, sac_state
    def recommend_item_random(self,top_k=False):
        if top_k:
            random_items = numpy.random.choice(self.random_recommend_space, top_k)
            return random_items
        else:
            random_item = numpy.random.choice(self.random_recommend_space, 1)[0]
            return random_item

    # TD学习的目标值也就是网络应该去拟合的label值
    def calculate_td_target(self, rewards, dones, batch_states, batch_next_states):
        # newQ = r+γ*nextQ
        # nextQ = t_critic(next_s,next_a)
        # next_a = t_actor(next_s)

        # 计算下一动作用target网络
        target_next_mean, target_next_logvar = self.sac_actor.target_mean_network(
            batch_next_states), self.sac_actor.target_logvar_network(batch_next_states)
        # 这里的critic应该直接给actor网络的输出值打分还是应该给加了随机因素的action打分，配合分布熵或者单纯的-logP进行优化，是个问题

        # critic的网络和目标网络分别给下一状态动作打分以作为未来期望奖励 Double Q method
        '''技术分叉点：直接对action打分or对均值和方差打分？直接对action打分没法求导'''
        qs = self.sac_critic.network([target_next_mean, target_next_logvar, batch_next_states])
        target_qs = self.sac_critic.target_network([target_next_mean, target_next_logvar, batch_next_states])
        min_qs = tensorflow.raw_ops.Min(input=tensorflow.concat([target_qs, qs], axis=1), axis=1, keep_dims=True)

        # 计算熵,熵是一个正值代表混乱程度,这里提现多样性
        entropies = self.calculate_entropy(batch_states)

        y_t = numpy.copy(min_qs)
        for i in range(min_qs.shape[0]):
            y_t[i] = rewards[i] + entropies[i] + (1 - dones[i]) * (self.discount_factor * min_qs[i])
        return y_t
    # 熵的计算
    def calculate_entropy(self, states):
        means = self.sac_actor.mean_network(states)
        log_vars = self.sac_actor.logvar_network(states)
        # 转成numpy数组
        means = keras.backend.eval(means)
        log_vars = keras.backend.eval(log_vars)
        entropies = []
        for i in range(self.batch_size):
            mean = means[i]
            log_var = log_vars[i]
            var = numpy.exp(log_var)
            cov = numpy.diag(var)
            # 重采样
            x = numpy.random.multivariate_normal(mean, cov)
            # 逆协方差矩阵
            cov_inverse = numpy.linalg.inv(cov)
            alpha = x - mean
            temp1 = numpy.dot(alpha, cov_inverse)
            temp2 = numpy.dot(temp1, alpha) / 2

            det = numpy.linalg.det(cov)
            '''技术分叉点：熵的计算方式'''
            # 多元高斯分布的熵,sac作者代码里用的概率密度的对数。
            # 我对这里保持看法，因为如果计算熵涉及sample取值，而sample本身是随机取的，难道让随机性影响优化结果吗？
            # 分布的熵
            # entropy1 = numpy.log(det)/2+(self.embedding_dim/2)*(numpy.log(2*numpy.pi)+1)
            # 直接-logP，概率密度值
            entropy = numpy.log((numpy.pi ** 50) * (det ** (1 / 2))) + temp2
            # 归一化到一维（但就算是一维-log概率密度值也是可能大于1的）
            entropies.append(0.3 * entropy / 100)
        entropies = numpy.array([entropies])
        entropies = entropies.reshape((self.batch_size, 1))
        entropies = tensorflow.constant(entropies, tensorflow.float32)
        return entropies


    # 传入评分获取奖励
    def reward_represent(self,rates):
        rewards = rates-3
        return rewards


    # 模型训练
    def train(self, max_episode_num, top_k=False, load_model=False):
        # 重置target网络
        self.sac_actor.update_target_network()
        self.sac_critic.update_target_network()
        print(load_model)
        episodic_precision_history = []
        for episode in range(max_episode_num):
            # episodic reward 每轮置零
            episode_reward = 0
            sac_correct_count = 0
            random_correct_count = 0

            steps = 0
            q_loss = 0
            mean_action = 0
            # Environment 从环境里随机获得一个用户以及历史，即每轮选一个模拟用户进行一次完整的经历
            # recommender进行推荐，user反馈reward（由评分构成，可复合多种指标进行设置）
            user_id, short_term_ids, done = self.env.reset()
            # sac方式训练时的推荐空间
            self.sac_recommend_space = copy.deepcopy(self.env.recommend_space_train)
            # Ddpg方式训练时的推荐空间
            self.ddpg_recommend_space = copy.deepcopy(self.env.recommend_space_train)
            # 随机方式训练时的推荐空间
            self.random_recommend_space = copy.deepcopy(self.env.recommend_space_train)

            # sac方式训练时的历史记录
            self.sac_recommended_items = copy.deepcopy(self.env.recommended_items_init)
            # Ddpg方式训练时的历史记录
            self.ddpg_recommended_items = copy.deepcopy(self.env.recommended_items_init)
            # 随机方式训练时的历史记录
            self.random_recommended_items = copy.deepcopy(self.env.recommended_items_init)







            print("当前用户",user_id)
            # print("推荐空间",self.env.recommend_space_train)
            print("推荐空间长度",len(self.env.recommend_space_train))
            print("推荐空间里的好评记录数",self.env.positive_rates_count)
            while not done:
                # 做出推荐
                sac_item_id, action, mean, log_var,state = self.recommend_item_sac(user_id=user_id,history=self.sac_recommended_items)
                random_item_id = self.recommend_item_random()

                #与用户交互得到评价
                random_recommend_rates = self.env.step(random_item_id, top_k=top_k)
                sac_rates = self.env.step(sac_item_id, top_k=top_k)

                # 推荐空间删除掉已推荐项目
                self.random_recommend_space = self.random_recommend_space[self.random_recommend_space != random_item_id]
                self.sac_recommend_space = self.sac_recommend_space[self.sac_recommend_space != sac_item_id]



                # 推荐历史改变
                self.random_recommended_items['item_ids_list'].append(random_item_id)
                self.random_recommended_items['rates_list'].append(random_recommend_rates)
                self.sac_recommended_items['item_ids_list'].append(sac_item_id)
                self.sac_recommended_items['rates_list'].append(sac_rates)

                # 已推荐项目增加




                # print("推荐空间长度", len(self.env.recommend_space_train))

                # 奖励计算
                sac_rewards = self.reward_represent(sac_rates)
                random_rewards = self.reward_represent(random_recommend_rates)
                # print("用户的第",steps+1,"次推荐")
                if sac_rewards > 0:
                    sac_correct_count += 1
                    # print(sac_correct_count)
                if random_rewards > 0:
                    random_correct_count += 1

                # top_k情况下的reward为累计reward
                if top_k:
                    sac_rewards = numpy.sum(sac_rewards)

                # get next_state,获取下一状态
                next_state = self.representation_model.get_state()
                # 把经历加入buffer里有，之后训练网络的时候进行经历重放
                self.sac_buffer.append(state, action, mean, log_var, sac_rewards, next_state, done)

                # 从buffer里取出批量让网络进行学习
                if self.sac_buffer.crt_idx > 1 or self.sac_buffer.is_full:
                    # 取出一个批量
                    batch_states, batch_actions, batch_means, batch_logvars, batch_rewards, batch_next_states, batch_dones, weight_batch, index_batch = self.sac_buffer.sample(
                        self.batch_size)

                    # TD学习的目标值即q网络应该去拟合的label值
                    td_targets = self.calculate_td_target(batch_rewards, batch_dones, batch_states, batch_next_states)
                    # Update priority
                    for (p, i) in zip(td_targets, index_batch):
                        self.sac_buffer.update_priority(abs(p[0]) + self.epsilon_for_priority, i)

                    q_loss += self.sac_critic.train([batch_means, batch_logvars, batch_states], td_targets, weight_batch)

                    # q对a求导再a对actor的参数求导最后得到分值对actor网络参数的导数，朝着使分值增大的方向优化，传反梯度
                    s_grads1, s_grads2 = self.sac_critic.dq_da([batch_means, batch_logvars, batch_states])
                    self.sac_actor.train(batch_states, s_grads1, s_grads1)
                    self.sac_actor.update_target_network()
                    self.sac_critic.update_target_network()

                episode_reward += sac_rewards
                mean_action += numpy.sum(action[0]) / (len(action[0]))
                steps += 1

                # 计算一次经历的推荐精度

                print(
                    f'recommended items : {steps}, reward : {sac_rewards:+}',
                    end='\r')
                # 是否应该结束推荐
                if steps >=  self.env.positive_rates_count:
                    done = True


                if done:
                    print("sac推荐被好评：",sac_correct_count)
                    print("随机推荐被好评：",random_correct_count)
                    sac_precision = int(sac_correct_count / self.env.positive_rates_count * 100)
                    random_precision = int(random_correct_count / self.env.positive_rates_count * 100)
                    print("推荐项目数量",steps)
                    print(
                        f'{episode}/{max_episode_num}, sac_precision : {sac_precision:2}%, random_precision : {random_precision:2}%, total_reward:{episode_reward}, q_loss : {q_loss / steps}, mean_action : {mean_action / steps}')
                    episodic_precision_history.append(sac_precision)
                    print()

            if (episode + 1) % 50 == 0:
                plt.plot(
                    episodic_precision_history[len(episodic_precision_history) - 50:len(episodic_precision_history)])
                plt.savefig(r'precisionimages1\training_precision__' + str(episode + 1) + '__%_top_5.png')
                plt.clf()

            if (episode + 1) % 100 == 0:
                self.save_model(r'weights\actor_mean' + str(episode + 1) + '_fixed.h5',
                                r'weights\actor_logvar' + str(episode + 1) + '_fixed.h5',
                                r'weights\critic_' + str(episode + 1) + '_fixed.h5')

    def save_model(self, actor_mean_path, actor_logvar_path, critic_path):
        self.sac_actor.save_weights(actor_mean_path, actor_logvar_path)
        self.sac_critic.save_weights(critic_path)

    def load_model(self, actor_path, critic_path):
        self.sac_actor.load_weights(actor_path)
        self.sac_critic.load_weights(critic_path)

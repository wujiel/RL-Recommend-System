import numpy
import numpy as np

'''
是否应该遵循？：
只学有的，不学没的
即：训练时不推荐不在历史记录的项目，不规定其奖励（因为历史记录没有）
不学习没有的短期模式序列（即最近n条交互真的在历史上发生而不是推荐系统推荐后模拟环境就把模拟交互当做短期模式就）

有历史记录的情况下，是否应该直接MC学习而不是TD学习

若TD学习，必引入假设


'''
def count_positive_rates(rates_list):
    count = 0
    for rate in rates_list:
        if rate >= 4:
            count += 1
    return count

class EnvironmentSimulator(object):

    def __init__(self, users_history_dict, lens_list, positive_lens_list, movies_information, short_term_state_size,
                 specify_user_id=None):

        # 用户历史记录，已经以时间排序
        self.users_history_dict = users_history_dict
        # 用户的历史长度
        self.lens_list = lens_list
        # 用户的好评历史长度
        self.positive_lens_list = positive_lens_list
        # 指定用户id
        self.specify_user_id = specify_user_id
        # 表示最近short_term_state_size个历史交互记录来表示短期模式,用来生成可用的模拟用户，（记录长度不够的话就不可用）
        self.short_term_state_size = short_term_state_size
        # 可模拟用户列表
        self.available_users = self._generate_available_users()
        # 指定用户或者从可模拟用户列表中随机选择用户id
        self.user = specify_user_id if specify_user_id else np.random.choice(self.available_users)
        # 当前模拟用户的历史交互项目,包含评价，dict是无序的，但users_history_dict[self.user]是一个元祖列表
        self.user_items = {data[0]: data[1] for data in self.users_history_dict[self.user]}
        # 当前模拟用户的历史交互项目,不包含评价,不包含已推荐项目
        self.recommend_space_train = numpy.array([data[0] for data in self.users_history_dict[self.user][self.short_term_state_size:]])
        # 推荐空间的评价
        self.recommend_space_train_rates = [data[1] for data in self.users_history_dict[self.user][self.short_term_state_size:]]
        # 推荐空间积极评价的数量
        self.positive_rates_count = count_positive_rates(self.recommend_space_train_rates)

        # 短期模式包含的交互项目
        self.short_term_ids = [data[0] for data in self.users_history_dict[self.user][:self.short_term_state_size]]
        # 短期模式包含的交互项目的评分
        self.short_term_rates = [data[1] for data in self.users_history_dict[self.user][:self.short_term_state_size]]
        # 初始交互，包括评价和item_id,有顺序
        self.recommended_items_init = {'item_ids_list':self.short_term_ids, 'rates_list':self.short_term_rates}


        # 是否可用标志位，训练用。即：推荐系统在训练时推荐空间为用户历史记录包含的项目，因为超出历史记录是没有反馈的，人为设置奖励也是不合理的
        self.done = False
        # 初始的已推荐项目，训练时不可再推荐
        self.recommended_items = set(self.short_term_ids)
        self.done_count = 3000

        self.movies_information = movies_information

    # 生成可用的模拟用户，（记录长度不够的话就不可用）
    def _generate_available_users(self):
        available_users = []
        for i, length in zip(self.users_history_dict.keys(), self.lens_list):
            if length > self.short_term_state_size:
                available_users.append(i)
        return available_users

    # 模拟用户重设
    def reset(self,id=None):
        self.user = id if id else np.random.choice(self.available_users)
        self.user_items = {data[0]: data[1] for data in self.users_history_dict[self.user]}
        self.recommend_space_train = numpy.array([data[0] for data in self.users_history_dict[self.user][self.short_term_state_size:]])
        self.short_term_ids = [data[0] for data in self.users_history_dict[self.user][:self.short_term_state_size]]
        # 推荐空间的评价
        self.recommend_space_train_rates = [data[1] for data in self.users_history_dict[self.user][self.short_term_state_size:]]
        # 推荐空间积极评价的数量
        self.positive_rates_count = count_positive_rates(self.recommend_space_train_rates)
        self.done = False
        self.recommended_items = set(self.short_term_ids)
        # 短期模式包含的交互项目
        self.short_term_ids = [data[0] for data in self.users_history_dict[self.user][:self.short_term_state_size]]
        # 短期模式包含的交互项目的评分
        self.short_term_rates = [data[1] for data in self.users_history_dict[self.user][:self.short_term_state_size]]
        # 初始交互，包括评价和item_id,有顺序
        self.recommended_items_init = {'item_ids_list': self.short_term_ids, 'rates_list': self.short_term_rates}
        return self.user, self.short_term_ids, self.done

    ''' 
    返回真正的评价，不做任何归一化
    已推荐项目到达好评历史长度则更换（）
    '''

    def step(self, action, top_k=False):
        # 得到的action应该是一个具体的项目id列表
        if top_k:
            rates = []
            for act in action:
                rates.append(self.user_items[act])
            rate = rates

        else:
            rate = self.user_items[action]  # reward

        return rate

    def get_items_names(self, items_ids):
        items_names = []
        for id in items_ids:
            try:
                items_names.append(self.movies_information[str(id)])
            except:
                items_names.append(list(['Not in list']))
        return items_names

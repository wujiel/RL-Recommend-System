import numpy
import numpy as np
import os


class Dataloader():
    def __init__(self):
        # ac网络
        # 全集
        self.movies_id_to_movies = np.load(r"data/data_ac_train/movies_id_to_movies.npy", allow_pickle=True).item()
        self.users_history_dict = np.load(r"data/users_history_dict.npy", allow_pickle=True).item()
        # 训练集
        self.users_history_dict_train = np.load(r"data/data_ac_train/users_history_dict_train.npy", allow_pickle=True).item()
        self.lens_list_train = np.load(r"data/data_ac_train/lens_list_train.npy", allow_pickle=True)
        self.positive_lens_list_train = np.load(r"data/data_ac_train/positive_lens_list_train.npy", allow_pickle=True)
        # 验证集
        self.users_history_dict_eval = np.load(r"data/data_ac_eval/users_history_dict_eval.npy", allow_pickle=True).item()
        self.lens_list_eval = np.load(r"data/data_ac_eval/lens_list_eval.npy", allow_pickle=True)
        self.positive_lens_list_eval = np.load(r"data/data_ac_eval/positive_lens_list_eval.npy", allow_pickle=True)

        # embedding网络
        # 全集
        self.positive_history_dict = np.load(r"data/positive_history_dict.npy", allow_pickle=True).item()
        self.u_m_positive_pairs = np.load(r"data/u_m_positive_pairs.npy", allow_pickle=True)
        # 训练集
        self.positive_history_dict_train = np.load(r"data/data_embedding_train/positive_history_dict_train.npy", allow_pickle=True).item()
        self.u_m_positive_pairs_train = np.load(r"data/data_embedding_train/u_m_positive_pairs_train.npy", allow_pickle=True)
        # 验证集
        self.positive_history_dict_eval = np.load(r"data/data_embedding_eval/positive_history_dict_eval.npy", allow_pickle=True).item()
        self.u_m_positive_pairs_eval = np.load(r"data/data_embedding_eval/u_m_positive_pairs_eval.npy", allow_pickle=True)

class data_deal():
    def __init__(self):
        self.rates_dict = np.load(r"data/users_history_dict.npy", allow_pickle=True).item()
        self.rates_dict_train = self.rates_dict.copy()
        self.rates_dict_eval = self.rates_dict.copy()

    def get_all_set(self):
        return self.rates_dict

    # 对于一个时序决策序列来说，训练集与测试集划分应遵循时间？
    # 根据时间先后顺序将全集划分为训练集和测试集
    def produce_train_set_and_eval_set(self):
        rates_dict_train = self.rates_dict.copy()
        rates_dict_eval = self.rates_dict.copy()
        users_num = len(self.rates_dict)
        for i in range(users_num):
            user_i = self.rates_dict[i + 1]
            len_user_i = len(user_i)
            len_user_i_train = int(len_user_i * 0.8)
            rates_dict_train[i + 1] = user_i[0:len_user_i_train]
            rates_dict_eval[i + 1] = user_i[len_user_i_train:]
        np.save("data/data_ac_train/users_history_dict_train.npy", rates_dict_train)
        np.save("data/data_ac_eval/users_history_dict_eval.npy", rates_dict_eval)
        return rates_dict_train, rates_dict_eval

    # 计算用户的历史长度，好评历史长度
    # def calculate_user_history_lens:
dataloader = Dataloader()

# embedding网络训练
u_m_positive_pairs_train = dataloader.u_m_positive_pairs_train
positive_history_dict_train = dataloader.positive_history_dict_train
print("shit")

all_set = dataloader.users_history_dict
train_set = dataloader.users_history_dict_train
eval_set = dataloader.users_history_dict_eval

'''
传入一个历史用户历史记录集合
计算并返回每个用户的：
1.历史长度
2.好评历史长度
3.好评字典
4.好评对
'''
def calculate_user_history_lens(history_dict):
    lens_list = []
    positive_lens_list = []
    positive_history_dict = {u: [] for u in range(1, len(history_dict) + 1)}
    u_m_positive_pairs = []
    i = 0
    for history_list in history_dict.values():
        positive_count = 0
        lens_list.append(len(history_list))
        for rate_tuple in history_list:
            if rate_tuple[1] >= 4:
                positive_count += 1
                positive_history_dict[i + 1].append(rate_tuple[0])
                positive_pair = [i + 1, rate_tuple[0]]
                u_m_positive_pairs.append(positive_pair)
        i += 1
        positive_lens_list.append(positive_count)
    return lens_list, positive_lens_list, positive_history_dict, u_m_positive_pairs

# 每个用户的好评字典和好评用户-电影对



lens_list, positive_lens_list, positive_history_dict, u_m_positive_pairs = calculate_user_history_lens(all_set)
u_m_positive_pairs = numpy.array(u_m_positive_pairs)
# ac网络
np.save("data/lens_list.npy", lens_list)
np.save("data/positive_lens_list.npy", positive_lens_list)
# embedding网络
np.save("data/positive_history_dict.npy", positive_history_dict)
np.save("data/u_m_positive_pairs.npy", u_m_positive_pairs)

lens_list_train, positive_lens_list_train, positive_history_dict_train, u_m_positive_pairs_train = calculate_user_history_lens(
    train_set)
u_m_positive_pairs_train = numpy.array(u_m_positive_pairs_train)
# ac网络
np.save("data/data_ac_train/lens_list_train.npy", lens_list_train)
np.save("data/data_ac_train/positive_lens_list_train.npy", positive_lens_list_train)
# embedding网络
np.save("data/data_embedding_train/positive_history_dict_train.npy", positive_history_dict_train)
np.save("data/data_embedding_train/u_m_positive_pairs_train.npy", u_m_positive_pairs_train)

lens_list_eval, positive_lens_list_eval, positive_history_dict_eval, u_m_positive_pairs_eval = calculate_user_history_lens(eval_set)
u_m_positive_pairs_eval = numpy.array(u_m_positive_pairs_eval)
# ac网络
np.save("data/data_ac_eval/lens_list_eval.npy", lens_list_eval)
np.save("data/data_ac_eval/positive_lens_list_eval.npy", positive_lens_list_eval)
# embedding网络
np.save("data/data_embedding_eval/positive_history_dict_eval.npy", positive_history_dict_eval)
np.save("data/data_embedding_eval/u_m_positive_pairs_eval.npy", u_m_positive_pairs_eval)


# lens_list_eval,positive_lens_list_eval = calculate_user_history_lens(eval_set)


# 先别写了，把已经写完的数据加载打包一下，pair要从list转为ndarray

# 生成格式如（3,3156,0）表示用户3没有对3156有过好评
# def generate_user_movie_batch(positive_dict, positive_pairs, batch_size, negative_ratio=0.5):
#     batch = np.zeros((batch_size, 3))
#     positive_batch_size = batch_size - int(batch_size * negative_ratio)
#     max_user_id = 6040
#     max_movie_id = 3900
#     while True:
#         idx = np.random.choice(len(positive_pairs), positive_batch_size)
#         data = positive_pairs[idx]
#         for i, d in enumerate(data):
#             batch[i] = (d[0], d[1], 1)
#         while i + 1 < batch_size:
#             u = np.random.randint(1, max_user_id + 1)
#             m = np.random.randint(1, max_movie_id + 1)
#             if m not in positive_dict[u]:
#                 i += 1
#                 batch[i] = (u, m, 0)
#             else:
#                 tt = batch[i]
#
#         np.random.shuffle(batch)
#         yield batch[:, 0], batch[:, 1], batch[:, 2]
#
# test_generator = generate_user_movie_batch(positive_history_dict_train, u_m_positive_pairs_train, batch_size=64,negative_ratio=0.5)
# u_batch, m_batch, u_m_label_batch = next(test_generator)
#

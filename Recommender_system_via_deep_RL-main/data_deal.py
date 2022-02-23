import pandas as pd
import numpy as np
import os
'''数据处理，训练集，测试集，验证集的分割 
   embedding网络和ac网络 
'''

ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'ml-1m/')
STATE_SIZE = 10
# 加载数据
ratings_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR, 'ratings.dat'), 'r').readlines()]
users_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR, 'users.dat'), 'r').readlines()]
movies_list = [i.strip().split("::") for i in
               open(os.path.join(DATA_DIR, 'movies.dat'), encoding='latin-1').readlines()]

# 用户对电影的评价dataframe，用户id，电影id，评分，时间戳
ratings_dataframe = pd.DataFrame(ratings_list, columns=['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype=np.uint32)
# 电影数量3883
movies_list_len = len(movies_list)
# 空值数量，无，数据已经清洗过
null_sum = ratings_dataframe.isnull().sum()
print(len(set(ratings_dataframe["UserID"])) == max([int(i) for i in set(ratings_dataframe["UserID"])]))
print(max([int(i) for i in set(ratings_dataframe["UserID"])]))
ratings_dataframe = ratings_dataframe.applymap(int)
ratings_dataframe = ratings_dataframe.apply(np.int32)

















# '''ac网络'''
# # 电影id对应的电影信息，字典键是电影id，值是电影信息列表，传进环境里可以检查电影名
# movies_id_to_movies = {movie[0]: movie[1:] for movie in movies_list}
# np.save("./data/movies_id_to_movies.npy", movies_id_to_movies)
#
#
# # 按用户顺序整理记录用以ac网络
# users_dict = {user: [] for user in set(ratings_dataframe["UserID"])}
# # 按时间排序
# ratings_dataframe = ratings_dataframe.sort_values(by='Timestamp', ascending=True)
#
# users_dict1 = {user: [] for user in set(ratings_dataframe["UserID"])}
#
# # 好评历史长度
# ratings_df_gen = ratings_dataframe.iterrows()
# users_dict_for_history_len = {user: [] for user in set(ratings_dataframe["UserID"])}
# users_dict_for_positive_history_len = {user: [] for user in set(ratings_dataframe["UserID"])}
# for data in ratings_df_gen:
#     users_dict[data[1]['UserID']].append((data[1]['MovieID'], data[1]['Rating']))
#     users_dict_for_history_len[data[1]['UserID']].append((data[1]['MovieID'], data[1]['Rating']))
#     if data[1]['Rating'] >= 4:
#         users_dict_for_positive_history_len[data[1]['UserID']].append((data[1]['MovieID'], data[1]['Rating']))
# # 每个用户的观影历史长度，共6040个用户
# users_history_lens = [len(users_dict_for_history_len[u]) for u in set(ratings_dataframe["UserID"])]
# # 每个用户的好评观影历史长度，共6040个用户
# users_positive_history_lens = [len(users_dict_for_positive_history_len[u]) for u in set(ratings_dataframe["UserID"])]
#
#
# #无后缀
# # 用以训练ac网络，全集
# # 每个用户的观影记录（键值对，键：用户id   值：元祖（电影和评分）组成的列表）,好评差评都包括，全集
# np.save("data/users_history_dict.npy", users_dict)
# # 用户的历史长度列表全集(用户1有53条记录历史45条好评，53)
# np.save("./data/users_history_lens.npy", users_history_lens)
# # 用户的好评历史长度列表(依次序)非全集(用户1有53条记录历史45条好评，45)
# np.save("./data/users_positive_history_lens.npy", users_positive_history_lens)
# # 6041，用户总数是6040
# users_num = max(ratings_dataframe["UserID"]) + 1
# # 3953，电影总数是3952
# items_num = max(ratings_dataframe["MovieID"]) + 1
#
#
# # _train
# #训练集
# # 4832
# users_num_train = int(users_num * 0.8)
# # 3953
# items_num_train = items_num
# users_dict_train = {k: users_dict[k] for k in range(1, users_num_train + 1)}
# np.save("./data/users_dict_train.npy", users_dict_train)
# users_history_lens_train = users_history_lens[:users_num_train]
# np.save("./data/users_history_lens_train.npy", users_history_lens_train)
# users_positive_history_lens_train = users_positive_history_lens[:users_num_train]
# np.save("./data/users_positive_history_lens_train.npy", users_positive_history_lens_train)
#
# # _eval
# # 测试集
# users_num_eval = int(users_num * 0.2)
# items_num_eval = items_num
# users_dict_eval = {k: users_dict[k] for k in range(users_num-users_num_eval, users_num)}
# np.save("./data/users_dict_eval.npy", users_dict_train)
# users_history_lens_eval = users_positive_history_lens[users_num - users_num_eval - 1:]
# np.save("./data/users_history_lens_eval.npy", users_history_lens_eval)
# users_positive_history_lens_eval = users_positive_history_lens[users_num - users_num_eval - 1:]
# np.save("./data/users_positive_history_lens_eval.npy", users_positive_history_lens_eval)




'''用以embedding网络'''
# 用户对电影的评价dataframe，用户id，电影id，评分，转化为好评对后预训练embedding用
user_movie_rating_dataframe = ratings_dataframe[['UserID', 'MovieID', 'Rating']]
user_movie_rating_dataframe = user_movie_rating_dataframe.apply(np.int32)
# 筛选出评分>3的好评记录(drop掉差评记录)，需要pairs和dict
index_negative = user_movie_rating_dataframe[user_movie_rating_dataframe['Rating'] < 4].index
positive_rating_df = user_movie_rating_dataframe.drop(index_negative)
positive_u_m_pairs = positive_rating_df.drop('Rating', axis=1).to_numpy()
np.save("./data/positive_u_m_pairs.npy", positive_u_m_pairs)
positive_user_movie_dict = {u: [] for u in range(1, max(positive_rating_df['UserID']) + 1)}
for data in positive_rating_df.iterrows():
    positive_user_movie_dict[data[1][0]].append(data[1][1])
np.save("./data/positive_user_movie_dict.npy", positive_user_movie_dict)

# 筛选出和ac网络统一的训练记录(drop掉>train_users_num的记录)，pairs和dict  （_train）
index_none_train = positive_rating_df[positive_rating_df['UserID'] > 4832].index
positive_rating_df_train = positive_rating_df.drop(index_none_train)
positive_u_m_pairs_train = positive_rating_df_train.drop('Rating', axis=1).to_numpy()
np.save("./data/positive_u_m_pairs_train.npy", positive_u_m_pairs_train)
positive_user_movie_dict_train = {u: [] for u in range(1, max(positive_rating_df_train['UserID']) + 1)}
for data in positive_rating_df_train.iterrows():
    positive_user_movie_dict_train[data[1][0]].append(data[1][1])
np.save("./data/positive_user_movie_dict_train.npy", positive_user_movie_dict_train)




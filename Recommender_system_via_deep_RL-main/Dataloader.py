import numpy

class Dataloader():
    def __init__(self):
        # ac网络
        # 全集
        self.movies_id_to_movies = numpy.load(r"data/data_ac_train/movies_id_to_movies.npy", allow_pickle=True).item()
        self.users_history_dict = numpy.load(r"data/users_history_dict.npy", allow_pickle=True).item()
        # 训练集
        self.users_history_dict_train = numpy.load(r"data/data_ac_train/users_history_dict_train.npy", allow_pickle=True).item()
        self.lens_list_train = numpy.load(r"data/data_ac_train/lens_list_train.npy", allow_pickle=True)
        self.positive_lens_list_train = numpy.load(r"data/data_ac_train/positive_lens_list_train.npy", allow_pickle=True)
        # 验证集
        self.users_history_dict_eval = numpy.load(r"data/data_ac_eval/users_history_dict_eval.npy", allow_pickle=True).item()
        self.lens_list_eval = numpy.load(r"data/data_ac_eval/lens_list_eval.npy", allow_pickle=True)
        self.positive_lens_list_eval = numpy.load(r"data/data_ac_eval/positive_lens_list_eval.npy", allow_pickle=True)

        # embedding网络
        # 全集
        self.positive_history_dict = numpy.load(r"data/positive_history_dict.npy", allow_pickle=True).item()
        self.u_m_positive_pairs = numpy.load(r"data/u_m_positive_pairs.npy", allow_pickle=True)
        # 训练集
        self.positive_history_dict_train = numpy.load(r"data/data_embedding_train/positive_history_dict_train.npy", allow_pickle=True).item()
        self.u_m_positive_pairs_train = numpy.load(r"data/data_embedding_train/u_m_positive_pairs_train.npy", allow_pickle=True)
        # 验证集
        self.positive_history_dict_eval = numpy.load(r"data/data_embedding_eval/positive_history_dict_eval.npy", allow_pickle=True).item()
        self.u_m_positive_pairs_eval = numpy.load(r"data/data_embedding_eval/u_m_positive_pairs_eval.npy", allow_pickle=True)
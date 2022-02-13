import tensorflow as tf
import numpy
from state_representation import DRRAveStateRepresentation
from EmbeddingNetwork import EmbeddingNetwork



class RepresentationModel:
    def __init__(self,short_term_state_size):
        # 短期模式包含最近short_term_state_size条交互
        self.short_term_state_size = short_term_state_size
        # 包括item_id列表和评分列表
        self.history = None
        # 用户id
        self.user_id = None


        # embedding网络
        self.embedding_network = EmbeddingNetwork(len_users=6040, len_movies=3900, embedding_dim=100)
        self.embedding_network([numpy.zeros((1)), numpy.zeros((1))])
        print("已创建embedding网络")
        self.embedding_network.load_weights(r'embedding_weights/embedding_network_weights,.h5')
        print("已加载权重")
        self.embedding_network.summary()
        self.state_represent_network = DRRAveStateRepresentation(embedding_dim=100)
        self.state_represent_network([numpy.zeros((1, 100,)), numpy.zeros((1, self.short_term_state_size, 100))])
        print("已创建state_represent_network")
        self.state_represent_network.summary()



    def calculate_state(self,short_term_item_ids,user_id):
        # embedding向量计算
        user_embedding_tensor = self.embedding_network.get_layer('user_embedding')(numpy.array(user_id))
        short_term_item_ids_embedding_tnnsor = self.embedding_network.get_layer('movie_embedding')(numpy.array(short_term_item_ids))
        ## 组合成状态向量
        state = self.state_represent_network([numpy.expand_dims(user_embedding_tensor, axis=0), numpy.expand_dims(short_term_item_ids_embedding_tnnsor, axis=0)])
        return state

    def get_state(self):
        # embedding向量计算
        user_id =  self.user_id
        item_ids_list = self.history['item_ids_list']
        short_term_item_ids = item_ids_list[len(item_ids_list)-self.short_term_state_size:]
        short_term_item_rates = self.history['rates_list']
        state = self.calculate_state(short_term_item_ids,user_id)
        return state



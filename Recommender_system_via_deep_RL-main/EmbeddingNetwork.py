import tensorflow
import numpy
class EmbeddingNetwork(tensorflow.keras.Model):
    def __init__(self, len_users, len_movies, embedding_dim):
        super(EmbeddingNetwork, self).__init__()
        self.m_u_input = tensorflow.keras.layers.InputLayer(name='input_layer', input_shape=(2,))
        # embedding
        self.u_embedding = tensorflow.keras.layers.Embedding(name='user_embedding', input_dim=len_users,
                                                     output_dim=embedding_dim)
        self.m_embedding = tensorflow.keras.layers.Embedding(name='movie_embedding', input_dim=len_movies,
                                                     output_dim=embedding_dim)
        # dot product
        self.m_u_merge = tensorflow.keras.layers.Dot(name='movie_user_dot', normalize=False, axes=1)
        # output
        # 以后可以多记忆一些特征，多分类
        self.m_u_fc = tensorflow.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.m_u_input(x)
        uemb = self.u_embedding(x[0])
        memb = self.m_embedding(x[1])
        m_u = self.m_u_merge([memb, uemb])
        return self.m_u_fc(m_u)


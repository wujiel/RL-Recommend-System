import keras
import numpy
import tensorflow
from Dataloader import Dataloader
from EmbeddingNetwork import EmbeddingNetwork

dataloader = Dataloader()
# embedding网络训练
u_m_positive_pairs_train = dataloader.u_m_positive_pairs_train
positive_history_dict_train = dataloader.positive_history_dict_train
print("shit")

# 生成格式如（3,3156,0）表示用户3没有对3156有过好评
def generate_user_movie_batch(positive_dict, positive_pairs, batch_size, negative_ratio=0.5):
    batch = numpy.zeros((batch_size, 3))
    positive_batch_size = batch_size - int(batch_size * negative_ratio)
    max_user_id = 6040
    max_movie_id = 3900
    while True:
        idx = numpy.random.choice(len(positive_pairs), positive_batch_size)
        data = positive_pairs[idx]
        for i, d in enumerate(data):
            batch[i] = (d[0], d[1], 1)
        while i + 1 < batch_size:
            u = numpy.random.randint(1, max_user_id + 1)
            m = numpy.random.randint(1, max_movie_id + 1)
            # 在的话不做任何操作用上一次的um0
            if m not in positive_dict[u]:
                i += 1
                batch[i] = (u, m, 0)
        numpy.random.shuffle(batch)
        yield batch[:, 0], batch[:, 1], batch[:, 2]
MAX_EPOCH = 1500
INIT_USER_BATCH_SIZE = 64
FINAL_USER_BATCH_SIZE = 1024
model = EmbeddingNetwork(len_users=6040,len_movies=3900,embedding_dim=100)
# input = [numpy.zeros((1)), numpy.zeros((1))]
# out = model([numpy.zeros((1)), numpy.zeros((1))])

model.build()
model.summary()

input1 = [numpy.zeros((1, 100,)), numpy.zeros((1, 10, 100))]
model.load_weights(r'embedding_weights/embedding_network_weights,.h5')

# 这是在build网络，传一个数据即可
# out = model([numpy.zeros((1)), numpy.zeros((1))])
model.summary()
# optimizer优化器
optimizer = tensorflow.keras.optimizers.Adam()
# loss损失函数
bce = tensorflow.keras.losses.BinaryCrossentropy()
test_train_loss = tensorflow.keras.metrics.Mean(name='train_loss')
test_train_accuracy = tensorflow.keras.metrics.BinaryAccuracy(name='train_accuracy')


@tensorflow.function
def test_train_step(test_inputs, labels):
    with tensorflow.GradientTape() as tape:
        predictions = model(test_inputs, training=True)
        model.summary()
        loss = bce(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    test_train_loss(loss)
    test_train_accuracy(labels, predictions)


test_losses = []

def train(max_epoch,positive_dict,positive_pairs):
    for epoch in range(MAX_EPOCH):
        batch_size = INIT_USER_BATCH_SIZE * (epoch + 1)
        if batch_size > FINAL_USER_BATCH_SIZE:
            batch_size = FINAL_USER_BATCH_SIZE
        test_generator = generate_user_movie_batch(positive_dict=positive_history_dict_train,
                                                   positive_pairs=u_m_positive_pairs_train, batch_size=batch_size)
        for step in range(len(u_m_positive_pairs_train) // batch_size):
            # embedding layer update
            u_batch, m_batch, u_m_label_batch = next(test_generator)
            # predictions = model([u_batch, m_batch], training=True)
            # predictions = keras.backend.eval(predictions)
            # prediction = []
            # for data in predictions:
            #     if data[0] > 0.5:
            #         prediction.append(1)
            #     else:
            #         prediction.append(0)
            test_train_step([u_batch, m_batch], u_m_label_batch)
            print(
                f'{epoch} epoch, Batch size : {batch_size}, {step} steps, Loss: {test_train_loss.result():0.4f}, Accuracy: {test_train_accuracy.result() * 100:0.1f}',
                end='\r')
        test_losses.append(test_train_loss.result())
        model.save_weights('./embedding_weights/embedding_network_weight,.h5')


# embedding层记忆特征
for epoch in range(MAX_EPOCH):
    batch_size = INIT_USER_BATCH_SIZE * (epoch + 1)
    if batch_size > FINAL_USER_BATCH_SIZE:
        batch_size = FINAL_USER_BATCH_SIZE
    test_generator = generate_user_movie_batch(positive_dict=positive_history_dict_train,positive_pairs=u_m_positive_pairs_train,batch_size=batch_size)
    for step in range(len(u_m_positive_pairs_train) // batch_size):
        # embedding layer update
        u_batch, m_batch, u_m_label_batch = next(test_generator)
        # predictions = model([u_batch, m_batch], training=True)
        # predictions = keras.backend.eval(predictions)
        # prediction = []
        # for data in predictions:
        #     if data[0] > 0.5:
        #         prediction.append(1)
        #     else:
        #         prediction.append(0)
        test_train_step([u_batch, m_batch], u_m_label_batch)
        print(
            f'{epoch} epoch, Batch size : {batch_size}, {step} steps, Loss: {test_train_loss.result():0.4f}, Accuracy: {test_train_accuracy.result() * 100:0.1f}',
            end='\r')
    test_losses.append(test_train_loss.result())
    model.save_weights('./embedding_weights/embedding_network_weight,.h5')





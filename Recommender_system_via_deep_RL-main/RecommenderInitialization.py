from replay_buffer import PriorityExperienceReplay
from ExperienceBuffer import DdpgBuffer
from actor import SacActor, DdpgActor
from critic import SacCritic, DdpgCritic
from Dataloader import Dataloader
from EnvironmentSimulator import EnvironmentSimulator
from RepresentationModel import RepresentationModel
from RecommendSystem import RecommendSystem

'''生成推荐系统所需要的:
1.环境
2.状态表示模块
3.actor
4.critic
5.经历重放的容器
'''


def RecommenderInitialization():
    dataloader = Dataloader()
    users_history_dict_train = dataloader.users_history_dict_train
    lens_list_train = dataloader.lens_list_train
    positive_lens_list_train = dataloader.positive_lens_list_train
    movies_id_to_movies = dataloader.movies_id_to_movies

    users_history_dict_eval = dataloader.users_history_dict_eval
    lens_list_eval = dataloader.lens_list_eval
    positive_lens_list_eval = dataloader.positive_lens_list_eval

    # 训练环境
    environment_train = EnvironmentSimulator(short_term_state_size=10, users_history_dict=users_history_dict_train,
                                             lens_list=lens_list_train,
                                             positive_lens_list=positive_lens_list_train,
                                             movies_information=movies_id_to_movies)
    # 验证环境
    environment_eval = EnvironmentSimulator(short_term_state_size=10, users_history_dict=users_history_dict_eval,
                                            lens_list=lens_list_eval,
                                            positive_lens_list=positive_lens_list_eval,
                                            movies_information=movies_id_to_movies,top_k=10)

    # 状态标识模块
    representationModel = RepresentationModel(short_term_state_size=10)

    # actor
    sac_actor = SacActor(state_dim=300, hidden_dim=128, action_dim=100, learning_rate=0.001,
                         target_network_update_rate=0.001)
    # critic
    sac_critic = SacCritic(state_dim=300, action_dim=100, hidden_dim=128, learning_rate=0.001,
                           target_network_update_rate=0.001)
    # 经历重放容器
    sac_buffer = PriorityExperienceReplay(buffer_size=1000000, embedding_dim=100)

    # actor
    ddpg_actor = DdpgActor(state_dim=300, hidden_dim=128, action_dim=100, learning_rate=0.001,
                         target_network_update_rate=0.001)
    # critic
    ddpg_critic = DdpgCritic(state_dim=300, action_dim=100, hidden_dim=128, learning_rate=0.001,
                           target_network_update_rate=0.001)
    # 经历重放容器
    ddpg_buffer = DdpgBuffer(buffer_size=1000000, embedding_dim=100)

    return environment_train, environment_eval, representationModel, sac_critic, sac_actor, sac_buffer,ddpg_critic, ddpg_actor, ddpg_buffer


# 状态表示模块两个网络，actor四个网络，critic两个网络，总共八个网络
environment_train, environment_eval, representationModel, sac_critic, sac_actor, sac_buffer,ddpg_critic, ddpg_actor, ddpg_buffer = RecommenderInitialization()

recommend_system = RecommendSystem(env=environment_train, representation_model=representationModel, sac_actor=sac_actor,
                                   sac_critic=sac_critic, sac_buffer=sac_buffer, ddpg_actor=ddpg_actor,
                                   ddpg_critic=ddpg_critic, ddpg_buffer=ddpg_buffer)

recommend_system_eval = RecommendSystem(env=environment_eval, representation_model=representationModel, sac_actor=sac_actor,
                                   sac_critic=sac_critic, sac_buffer=sac_buffer, ddpg_actor=ddpg_actor,
                                   ddpg_critic=ddpg_critic, ddpg_buffer=ddpg_buffer)

recommend_system.ddpg_critic.load_weights(r'actor_critic_weights/ddpg/critic_5000.h5')
recommend_system.ddpg_actor.load_weights(r'actor_critic_weights/ddpg/actor_5000.h5')
recommend_system.sac_critic.load_weights(r'actor_critic_weights/sac/critic_5000.h5')
recommend_system.sac_actor.load_weights(r'actor_critic_weights/sac/actor_mean5000.h5',r'actor_critic_weights/sac/actor_logvar5000.h5')

recommend_system_eval.ddpg_critic.load_weights(r'actor_critic_weights/ddpg/critic_5000.h5')
recommend_system_eval.ddpg_actor.load_weights(r'actor_critic_weights/ddpg/actor_5000.h5')
recommend_system_eval.sac_critic.load_weights(r'actor_critic_weights/sac/critic_5000.h5')
recommend_system_eval.sac_actor.load_weights(r'actor_critic_weights/sac/actor_mean5000.h5',r'actor_critic_weights/sac/actor_logvar5000.h5')



print("已加载ddpg")
recommend_system_eval.evaluation(max_user_id=6400,top_k=10)
recommend_system.train(max_episode_num=8000)
print("shit")

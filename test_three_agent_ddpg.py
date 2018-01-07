import numpy as np
import tensorflow as tf

import make_env

from model.model_single_agent_ddpg import DDPG
from util.replay_buffer import ReplayBuffer

def create_init_update(oneline_name, target_name, tau=0.99):
    online_var = [i for i in tf.trainable_variables() if oneline_name in i.name]
    target_var = [i for i in tf.trainable_variables() if target_name in i.name]

    target_init = [tf.assign(target, online) for online, target in zip(online_var, target_var)]
    target_update = [tf.assign(target, (1 - tau) * online + tau * target) for online, target in
                                   zip(online_var, target_var)]

    return target_init, target_update

agent1_ddpg = DDPG('agent1')
agent1_ddpg_target = DDPG('agent1_target')

agent2_ddpg = DDPG('agent2')
agent2_ddpg_target = DDPG('agent2_target')

agent3_ddpg = DDPG('agent3')
agent3_ddpg_target = DDPG('agent3_target')

saver = tf.train.Saver()

agent1_actor_target_init, agent1_actor_target_update = create_init_update('agent1_actor', 'agent1_target_actor')
agent1_critic_target_init, agent1_critic_target_update = create_init_update('agent1_critic', 'agent1_target_critic')

agent2_actor_target_init, agent2_actor_target_update = create_init_update('agent2_actor', 'agent2_target_actor')
agent2_critic_target_init, agent2_critic_target_update = create_init_update('agent2_critic', 'agent2_target_critic')

agent3_actor_target_init, agent3_actor_target_update = create_init_update('agent3_actor', 'agent3_target_actor')
agent3_critic_target_init, agent3_critic_target_update = create_init_update('agent3_critic', 'agent3_target_critic')


def get_agents_action(o_n, sess, noise_rate=0):
    agent1_action = agent1_ddpg.action(state=[o_n[0]], sess=sess) + np.random.randn(2) * noise_rate
    agent2_action = agent2_ddpg.action(state=[o_n[1]], sess=sess) + np.random.randn(2) * noise_rate
    agent3_action = agent3_ddpg.action(state=[o_n[2]], sess=sess) + np.random.randn(2) * noise_rate

    return agent1_action, agent2_action, agent3_action


def train_agent(agent_ddpg, agent_ddpg_target, agent_memory, agent_actor_target_update, agent_critic_target_update, sess):
    obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = agent_memory.sample(32)
    new_action_batch = agent_ddpg.action(next_obs_batch, sess)
    target = rew_batch.reshape(-1, 1) + 0.9999 * agent_ddpg_target.Q(state=next_obs_batch, action=new_action_batch,
                                                                      sess=sess)
    agent_ddpg.train_actor(state=obs_batch, sess=sess)
    agent_ddpg.train_critic(state=obs_batch, action=act_batch, target=target, sess=sess)

    sess.run([agent_actor_target_update, agent_critic_target_update])


if __name__ == '__main__':
    env = make_env.make_env('simple_tag')
    o_n = env.reset()

    agent_reward_v = [tf.Variable(0, dtype=tf.float32) for i in range(3)]
    agent_reward_op = [tf.summary.scalar('agent' + str(i) + '_reward', agent_reward_v[i]) for i in range(3)]

    agent_a1 = [tf.Variable(0, dtype=tf.float32) for i in range(3)]
    agent_a1_op = [tf.summary.scalar('agent' + str(i) + '_action_1', agent_a1[i]) for i in range(3)]

    agent_a2 = [tf.Variable(0, dtype=tf.float32) for i in range(3)]
    agent_a2_op = [tf.summary.scalar('agent' + str(i) + '_action_2', agent_a2[i]) for i in range(3)]
    
    reward_100 = [tf.Variable(0, dtype=tf.float32) for i in range(3)]
    reward_100_op = [tf.summary.scalar('agent' + str(i) + '_reward_l100_mean', reward_100[i]) for i in range(3)]

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run([agent1_actor_target_init, agent1_critic_target_init,
              agent2_actor_target_init, agent2_critic_target_init,
              agent3_actor_target_init, agent3_critic_target_init])
    saver.restore(sess, './weight_single/210000.cptk')
    summary_writer = tf.summary.FileWriter('./test_three_summary', graph=tf.get_default_graph())

    agent1_memory = ReplayBuffer(100000)
    agent2_memory = ReplayBuffer(100000)
    agent3_memory = ReplayBuffer(100000)

    e = 1
    
    reward_100_list = [[], [], []]
    for i in range(1000000):
        if i % 1000 == 0:
            o_n = env.reset()

        agent1_action, agent2_action, agent3_action = get_agents_action(o_n, sess, noise_rate=0.1)
        
        env.render()

        a = [[0, i[0][0], 0, i[0][1], 0] for i in [agent1_action, agent2_action, agent3_action]]

        a.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])

        o_n_next, r_n, d_n, i_n = env.step(a)

        o_n = o_n_next
  

    sess.close()

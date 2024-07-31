import gym
import numpy as np
import tensorflow as tf
import argparse
from policy_net_breakout import Policy_net
from collections import deque


target_score = 16


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='directory of model', default='trained_models/gail')
    parser.add_argument('--model', help='trained model to test', default='GAIL_model_breakout-v0_1000iter.ckpt')
    parser.add_argument('--logdir', help='log directory', default='log/test/gail')
    parser.add_argument('--numepisodes', default=int(1e3))
    #parser.add_argument('--numepisodes', default=int(100))
    return parser.parse_args()

def preprocess_obs_state(obs):
    obs = np.resize(obs , (84,84,4))
    obs = np.stack([obs]).astype(dtype=np.float32)
    return obs


def main(args):
    env = gym.make('Breakout-v0')
    env.seed(0)
    Policy = Policy_net('policy', env)

    sum_rewardsperepisode_list = []

    rewards_wndw = deque(maxlen=100) #last 100 scores

    saver = tf.compat.v1.train.Saver()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(args.logdir+'/', sess.graph)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, args.modeldir  +'/' + args.model )

        obs = env.reset()
        reward = 0
        success_num = 0

        for iteration in range(args.numepisodes):
            rewards = []
            run_policy_steps = 0
            reward = 0
            while True:  # run policy RUN_POLICY_STEPS which is much less than episode length
                run_policy_steps += 1
                obs = preprocess_obs_state(obs)  # preprocess the state to feed placeholder Policy.input_normalized
            
                act, _ = Policy.act(obs=obs)

                act = np.asscalar(act)

                rewards.append(reward)

                next_obs, reward, done, info = env.step(act)
                env.render()
                if done:
                    obs = env.reset()
                    break
                else:
                    obs = next_obs
            
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=run_policy_steps)])
                               , iteration)
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rewards))])
                               , iteration)

            
            render = True

            #print(" TOTAL REWARDS IN EPISODE ::::::"  + str(sum(rewards)))

            sum_rewardsperepisode_list.append(sum(rewards))
            rewards_wndw.append(sum(rewards))

            if(iteration%100 == 0 and iteration > 0 ):
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(iteration, np.mean(rewards_wndw)))
            # end condition of test
            if (np.mean(rewards_wndw) >= target_score):
                print('Target Score of 16 reached consistently over a range of 100 episodes.')


        writer.close()
        env.close()


if __name__ == '__main__':
    args = argparser()
    main(args)

import time
import argparse
import pickle
import maddpg.maddpg.common.tf_util as U
from maddpg.maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

import tensorflow as tf
import numpy as np
from pysc2.lib.features import SCREEN_FEATURES
from maddpg.sc2_env.combined_action import get_action

_PLAYER_RELATIVE = SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = SCREEN_FEATURES.unit_type.index
_SELECTED = SCREEN_FEATURES.selected.index

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--map", type=str, default="Reapers", help="name of the scenario script")
    parser.add_argument("--step_mul", type=int, default=1, help="Game steps per agent step.")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.9, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="reapers", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=True)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def get_trainers(action_space, num_units, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_units):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    return trainers

def run_loop(agents, env, max_frames=0):
  """A run loop to have agents and an environment interact."""
  total_frames = 0
  start_time = time.time()
  arglist = parse_args()

  action_spec = env.action_spec()
  observation_spec = env.observation_spec()
  for agent in agents:
    agent.setup(observation_spec, action_spec)

  try:
    with U.single_threaded_session():
        timesteps = env.reset()
        for a in agents:
            a.reset()
        for a, timestep in zip(agents, timesteps):
            a.selected_units(timestep)
            obs_shape_n, timestep = a.build_group(timestep, env)
            action_space = [i for i in range(3)]
            action_space_n = []
            agent_rewards = []
            for i in range(a.num_units):
                agent_rewards.append([0.0])
                action_space_n.append(action_space)
            trainers = get_trainers(action_space_n, a.num_units, obs_shape_n, arglist)
            # Initialize
            U.initialize()
            # Load previous results, if necessary
            if arglist.load_dir == "":
                arglist.load_dir = arglist.save_dir
            if arglist.display or arglist.restore or arglist.benchmark:
                print('Loading previous state...')
                U.load_state(arglist.load_dir) # sum of rewards for all agents
            final_ep_rewards = []  # sum of rewards for training curve
            final_ep_ag_rewards = []  # agent rewards for training curve
            saver = tf.train.Saver()
            loss_n = []
            train_step = 0
            obs_n, timestep = a.get_obs(timestep, env)
            t_start = time.time()
        print('Starting iterations...')
        while True:
            win_pro = timestep.win_pro
            episode_rewards = timestep.episode_rewards
            if len(win_pro) > 1:
                data = np.array(win_pro)
                np.savetxt(arglist.exp_name + '_win_pro.csv', data, delimiter=',')
            if len(loss_n) > 1:
                data = np.array(loss_n)
                np.savetxt(arglist.exp_name + '_loss.csv', data, delimiter=',')
            while True:
              total_frames += 1
              if isinstance(obs_n, list):
                  obs_n = np.array(obs_n)
              action_n = [trainer.action(obs) for trainer, obs in zip(trainers, obs_n)]
              rew_n = []
              for i, action in enumerate(action_n):
                if not timestep:
                    break
                for agent in agents:
                    if agent.group[i] == True:
                        timestep = agent.select_unit(i, timestep, env)
                        if not timestep:
                            break
                    timestep = get_action(action, timestep, env)
                    if not timestep:
                        break
                    new_obs_n, timestep = agent.get_obs(timestep, env)
                    rew_n.append(timestep.reward)

                if max_frames and total_frames >= max_frames:
                  return

              if not timestep:
                  break

              if len(new_obs_n) != 5:
                  for i in range(len(new_obs_n), 5):
                    new_obs_n.append([0] * 20)
              if len(rew_n) != 5:
                  for i in range(len(rew_n), 5):
                      rew_n.append(0)
              for i, agent in enumerate(trainers):
                  agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i])

              obs_n = new_obs_n
              for i, rew in enumerate(rew_n):
                  agent_rewards[i][-1] += rew

              if not arglist.display:

                  train_step += 1
                  # update all trainers, if not in display or benchmark mode
                  loss = None
                  for agent in trainers:
                      agent.preupdate()
                  for agent in trainers:
                      loss = agent.update(trainers, train_step)
                  if isinstance(loss, list):
                      loss_n.append(loss)
                      print('loss:', loss)

              # save model, display training output
              if (len(episode_rewards) % arglist.save_rate == 0):
                  U.save_state(arglist.save_dir, saver=saver)
                  # print statement depends on whether or not there are adversaries
                  print(
                      "steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                          train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                          [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards],
                          round(time.time() - t_start, 3)))
                  t_start = time.time()
                  # Keep track of final episode reward
                  final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                  for rew in agent_rewards:
                      final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))
              # saves final episode reward for plotting training curve later
              if len(episode_rewards) > arglist.num_episodes:
                  rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                  with open(rew_file_name, 'wb') as fp:
                      pickle.dump(final_ep_rewards, fp)
                  agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                  with open(agrew_file_name, 'wb') as fp:
                      pickle.dump(final_ep_ag_rewards, fp)
                  print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                  break
            timesteps = env.reset()
            for a in agents:
                a.reset()
            for a, timestep in zip(agents, timesteps):
                a.selected_units(timestep)
                obs_shape_n, timestep = a.build_group(timestep, env)
  except KeyboardInterrupt:
    pass
  finally:
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds for %s steps: %.3f fps" % (
        elapsed_time, total_frames, total_frames / elapsed_time))
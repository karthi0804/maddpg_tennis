from unityagents import UnityEnvironment
import torch
import numpy as np
from config import Config
import matplotlib.pyplot as plt
from collections import deque
from maddpg import MADDPG
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--train', '-t', action='store_true', help='verbose flag')
args = parser.parse_args()

env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64")
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)
# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)
# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

# config settings
config = Config()
config.update_every = 1
config.batch_size = 512
config.buffer_size = int(1e6)
config.discount = 0.99
config.tau = 0.2
config.seed = 2
config.lr_actor = 1e-4
config.lr_critic = 1e-4
config.action_size = action_size
config.state_size = state_size
config.num_agents = num_agents
ma = MADDPG(config)


def train(n_episode=30000):
    """
    Function to train the agent
    """
    scores = []
    scores_window = deque(maxlen=100)
    for i_episode in range(n_episode):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        ma.reset()
        score = np.zeros(num_agents)
        while True:
            actions = ma.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            ma.step(states, actions, rewards, next_states, dones)
            score += rewards
            states = next_states
            if np.any(dones):
                break
        max_score = np.max(score)
        scores_window.append(max_score)
        scores.append(max_score)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}\tCritic Loss: {:-11.10f}\tActor Loss: {:-10.6f}\t'
              't_step {:-8d}'.
              format(i_episode, np.mean(scores_window), max_score, ma.loss[0], ma.loss[1], ma.t_step),
              end="")
        # periodic model checkpoint
        if i_episode % 50 == 0:
            torch.save(ma.agents[0].actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(ma.agents[0].critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}\tCritic Loss: {:-11.10f}\tActor Loss: {:-10.6f}\t'
                  't_step {:-8d}'.
                  format(i_episode, np.mean(scores_window), ma.loss[0], ma.loss[1], ma.t_step))
        # Stopping the training after the avg score of 30 is reached
        if np.mean(scores_window) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                         np.mean(scores_window)))
            torch.save(ma.agents[0].actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(ma.agents[0].critic_local.state_dict(), 'checkpoint_critic.pth')
            break

    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


def test():
    for player in ma.agents:
        player.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
        player.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    ma.reset()
    while True:
        actions = ma.act(states, False)
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Score (max over agents) {}'.format(np.max(scores)))


if args.train:
    train()
else:
    test()
env.close()

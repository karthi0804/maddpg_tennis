from ddpg_agent import Agent, ReplayBuffer
import numpy as np


class MADDPG:
    def __init__(self, config):
        self.config = config
        # Replay memory
        self.memory = ReplayBuffer(self.config.action_size, self.config.buffer_size,
                                   self.config.batch_size, self.config.seed)
        self.agents = [Agent(self.config) for _ in range(self.config.num_agents)]
        self.t_step = 0
        self.loss = (0.0, 0.0)

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def act(self, states, add_noise=True):
        actions = [agent.act(state, self.t_step, add_noise) for agent, state in zip(self.agents, states)]
        return actions

    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step += 1
        if self.t_step % self.config.update_every == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.config.batch_size:
                closs = []
                aloss = []
                for agent in self.agents:
                    experiences = self.memory.sample()
                    critic_loss, actor_loss = agent.learn(experiences, self.config.discount)
                    closs.append(critic_loss)
                    aloss.append(actor_loss)
                self.loss = (np.mean(closs), np.mean(aloss))

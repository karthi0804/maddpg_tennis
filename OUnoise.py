import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt


class OUNoise1(object):
    def __init__(self, output_dim=1, mu=0, theta=0.15, sigma=0.15):
        self.output_dim = output_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.output_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.output_dim) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state


class OUNoise2(object):
    def __init__(self, n_steps=10000, output_dim=1, theta=0.15, mu=0., sigma=0.3, x0=0, dt=1e-2):
        self.theta = theta
        self.sigma = sigma
        self.n_steps = n_steps
        self.sigma_step = - self.sigma / float(self.n_steps)
        self.x0 = x0
        self.mu = mu
        self.dt = dt
        self.output_dim = output_dim

    def reset(self):
        self.x0 = 0.0

    def sample(self, step):
        sigma = max(0, self.sigma_step * step + self.sigma)
        x = self.x0 + self.theta * (self.mu - self.x0) * self.dt + sigma * np.sqrt(self.dt) * nr.normal(size=self.output_dim)
        self.x0 = x
        return x


if __name__ == '__main__':
    ou1 = OUNoise1()
    ou2 = OUNoise2(theta=0.3, sigma=0.2)
    ou3 = OUNoise2(theta=0.3, sigma=0.2)
    max_epi = 100
    max_step = 100

    noises1 = []
    noises2 = []
    noises3 = []
    for epi in range(max_epi):
        ou3.reset()
        for step in range(max_step):
            noises1.append(ou1.sample())
            noises2.append(ou2.sample(epi*max_step+step))
            noises3.append(ou3.sample(epi*max_step+step))

    plt.plot(noises1, "b", label="ou1 0.15 0.1")
    plt.plot(noises2, "r", label="ou2 0.3 0.2")
    plt.plot(noises3, "g", label="ou3 5.0 0.2")
    plt.legend()
    plt.show()


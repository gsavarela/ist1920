__author__ = 'Guilherme Varela'
__date__ = '2019-11-28'

import numpy as np
from matplotlib import pyplot as plt


class KBandits(object):
    """This class implements the multi-armed bandits

        USAGE:
        ======
        > Q = KBandits(5)
        > Q(5)
        > 0.323

        REFERENCE:
        ==========
        Sutton et Barto,
    """
    def __init__(self, K):
        self.mu = np.random.randn(K)

    def __call__(self, a):
        return self.mu[a - 1] + np.random.randn()


class Greedy(object):

    def __init__(self, K, value):
        self.label = f'Greedy {value:d}'
        self.mu = np.ones((K,), dtype=np.float) * value
        self.steps = np.ones((K,), dtype=np.int)

    @property
    def a(self):
        self._i = np.argmax(self.mu)
        return self._i + 1

    def update(self, r):
        i = self._i
        self.mu[i] = (self.steps[i] * self.mu[i] + r) / (self.steps[i] + 1)
        self.steps[i] += 1


class EpsGreedy(Greedy):

    def __init__(self, K, value, eps):
        super(EpsGreedy, self).__init__(K, value)
        self.label = f'Greedy {int(eps * 100):d}%'
        self.eps = eps

    @property
    def a(self):
        if np.random.rand() < self.eps:
            self._i = np.random.randint(0, len(self.mu))
            return self._i + 1
        return super(EpsGreedy, self).a


class UCB(object):
    def __init__(self):
        pass

    def action(self):
        pass


if __name__ == '__main__':
    K = 10
    Q = KBandits(K)
    best_reward = max(Q.mu)
    best_action = (1 + np.argmax(Q.mu))
    print("True means:", Q.mu)
    print(f"""Best action: {best_action:d}
              Best action's mean: {best_reward:f}""")

    # Greedy policy
    Q1 = Greedy(K, 0)
    Q2 = Greedy(K, 5)

    # Eps-Greedy policy
    Q3 = EpsGreedy(K, 0, 0.1)
    Q4 = EpsGreedy(K, 0, 0.01)

    # 2. run policies for N = 1000
    N = 2000
    trials = np.arange(1, N + 1)
    best_reward *= np.ones((N,), dtype=np.float)

    r1 = np.zeros((N,), dtype=np.float)
    r2 = np.zeros((N,), dtype=np.float)
    r3 = np.zeros((N,), dtype=np.float)
    r4 = np.zeros((N,), dtype=np.float)

    a1 = np.zeros((N,), dtype=np.int)
    a2 = np.zeros((N,), dtype=np.int)
    a3 = np.zeros((N,), dtype=np.int)
    a4 = np.zeros((N,), dtype=np.int)
    for t in trials:
        a1[t - 1] = int(Q1.a == best_action)
        a2[t - 1] = int(Q2.a == best_action)
        a3[t - 1] = int(Q3.a == best_action)
        a4[t - 1] = int(Q4.a == best_action)

        Q1.update(Q(a1[t - 1]))
        Q2.update(Q(a2[t - 1]))
        Q3.update(Q(a3[t - 1]))
        Q4.update(Q(a4[t - 1]))

        r1[t - 1] = np.max(Q1.mu)
        r2[t - 1] = np.max(Q2.mu)
        r3[t - 1] = np.max(Q3.mu)
        r4[t - 1] = np.max(Q4.mu)

    fig, ax = plt.subplots()
    ax.plot(np.cumsum(a1) / trials, color='r',
            label=Q1.label)

    ax.plot(np.cumsum(a2) / trials, color='b',
            label=Q2.label)

    ax.plot(np.cumsum(a3) / trials, color='r',
            label=Q3.label, linestyle='dashed')

    ax.plot(np.cumsum(a4) / trials, color='b',
            label=Q4.label, linestyle='dashed')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Frequency')
    ax.set_title('10-Armed Bandit Optimal Actions Proporsion')
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(best_reward, color='g', label='Best mean')
    ax.plot(np.cumsum(r1) / trials, color='r',
            label=Q1.label)
    ax.plot(np.cumsum(r2) / trials, color='b',
            label=Q2.label)
    ax.plot(np.cumsum(r3) / trials, color='r',
            label=Q3.label, linestyle='dashed')
    ax.plot(np.cumsum(r4) / trials, color='b',
            label=Q4.label, linestyle='dashed')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Reward')
    ax.set_title('10-Armed Bandit Policies Comparison')
    plt.legend()
    plt.show()

    

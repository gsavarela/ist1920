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
        return np.argmax(self.mu) + 1

    def update(self, a, r):
        i = a - 1
        self.mu[i] = (self.steps[i] * self.mu[i] + r) / (self.steps[i] + 1)
        self.steps[i] += 1


class EpsGreedy(Greedy):

    def __init__(self, K, value, eps):
        super(EpsGreedy, self).__init__(K, value)
        self.label = f'Greedy {int(eps * 100):d}%'
        self.eps = eps

    @property
    def a(self):
        greedy_a = super(EpsGreedy, self).a
        if np.random.rand() < self.eps:
            rand_as = [a for a in range(1, len(self.mu) + 1) if a != greedy_a]
            return np.random.choice(rand_as)
        return greedy_a


class UCB(Greedy):
    def __init__(self, K, value):
        super(UCB, self).__init__(K, 0)
        self.label = f'UCB {value:d}'
        self.c = value

    @property
    def a(self):
        t = np.sum(self.steps)
        ucb = self.mu + self.c * np.sqrt(np.log(t) / self.steps)
        return np.argmax(ucb) + 1


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

    # UCB
    Q5 = UCB(K, 1)

    # Perform 2000 independent runs
    # of 1000 steps each
    N = 1000
    M = 1
    trials = np.arange(1, N + 1)
    best_reward *= np.ones((N,), dtype=np.float)

    # rx hold the reward drawn at trial t
    r1 = np.zeros((M, N), dtype=np.float)
    r2 = np.zeros((M, N), dtype=np.float)
    r3 = np.zeros((M, N), dtype=np.float)
    r4 = np.zeros((M, N), dtype=np.float)
    r5 = np.zeros((M, N), dtype=np.float)
    
    # ax hold the action
    a1 = np.zeros((M, N), dtype=np.int)
    a2 = np.zeros((M, N), dtype=np.int)
    a3 = np.zeros((M, N), dtype=np.int)
    a4 = np.zeros((M, N), dtype=np.int)
    a5 = np.zeros((M, N), dtype=np.int)

    # fx hold the frequency that the best
    # action is choosen at draw t
    f1 = np.zeros((M, N), dtype=np.float)
    f2 = np.zeros((M, N), dtype=np.float)
    f3 = np.zeros((M, N), dtype=np.float)
    f4 = np.zeros((M, N), dtype=np.float)
    f5 = np.zeros((M, N), dtype=np.float)

    for m in range(M):
        for t in trials:
            a1[m, t - 1] = Q1.a
            a2[m, t - 1] = Q2.a
            a3[m, t - 1] = Q3.a
            a4[m, t - 1] = Q4.a
            a5[m, t - 1] = Q5.a

            r1[m, t - 1] = Q(a1[m, t - 1])
            r2[m, t - 1] = Q(a2[m, t - 1])
            r3[m, t - 1] = Q(a3[m, t - 1])
            r4[m, t - 1] = Q(a4[m, t - 1])
            r5[m, t - 1] = Q(a5[m, t - 1])

            Q1.update(a1[m, t - 1], r1[m, t - 1])
            Q2.update(a2[m, t - 1], r2[m, t - 1])
            Q3.update(a3[m, t - 1], r3[m, t - 1])
            Q4.update(a4[m, t - 1], r4[m, t - 1])
            Q5.update(a5[m, t - 1], r5[m, t - 1])

            # r1[m, t - 1] = np.max(Q1.mu)
            # r2[m, t - 1] = np.max(Q2.mu)
            # r3[m, t - 1] = np.max(Q3.mu)
            # r4[m, t - 1] = np.max(Q4.mu)

            f1[m, t - 1] = int(a1[m, t - 1] == best_action)
            f2[m, t - 1] = int(a2[m, t - 1] == best_action)
            f3[m, t - 1] = int(a3[m, t - 1] == best_action)
            f4[m, t - 1] = int(a4[m, t - 1] == best_action)
            f5[m, t - 1] = int(a5[m, t - 1] == best_action)

    # Graph rewards
    fig, ax = plt.subplots()
    ax.plot(trials, best_reward, color='g', label='Best mean')
    ax.plot(trials, np.cumsum(r1[0, :]) / trials, color='r',
            label=Q1.label)
    ax.plot(trials, np.cumsum(r2[0, :]) / trials, color='b',
            label=Q2.label)
    ax.plot(np.cumsum(r3[0, :]) / trials, color='r',
            label=Q3.label, linestyle='dashed')
    ax.plot(np.cumsum(r4[0, :]) / trials, color='b',
            label=Q4.label, linestyle='dashed')
    ax.plot(np.cumsum(r5[0, :]) / trials, color='g',
            label=Q5.label)

    ax.set_xlabel('Steps')
    ax.set_ylabel('Average Reward')
    ax.set_title('10-Armed Bandit Policies Comparison')
    plt.legend()
    plt.show()


    # Graph optimal actions proporsion
    fig, ax = plt.subplots()
    ax.plot(trials, np.cumsum(f1[m, :]) / trials, color='r',
            label=Q1.label)

    ax.plot(trials, np.cumsum(f2[m, :]) / trials, color='b',
            label=Q2.label)

    ax.plot(np.cumsum(f3[m, :]) / trials, color='r',
            label=Q3.label, linestyle='dashed')

    ax.plot(np.cumsum(f4[m, :]) / trials, color='b',
            label=Q4.label, linestyle='dashed')

    ax.plot(np.cumsum(f5[m, :]) / trials, color='g',
            label=Q5.label)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Frequency')
    ax.set_title('10-Armed Bandit Optimal Actions Proporsion')
    plt.legend()
    plt.show()


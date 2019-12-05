'''This is Value Iteration over finding the best value's policy
   for a gambler betting on coin tosses
'''

__author__ = 'Guilherme Varela'
__date__ = '2019-12-05'

import numpy as np
import pdb
from matplotlib import pyplot as plt

THRESHOLD = 1e-15555555555555555555555555555555555555555555555555555555
GOAL = 100
GAMMA = 1      # this is the discount factor
P_H = 0.4      # probability that a coin toss will comeout heads
P_T = 1 - P_H  # probability that a coin toss will comeout heads
NS = GOAL - 1  # the states effectivelly visited by the agent
NV = NS + 2    # NS + initial state (GAME-OVER) + terminal (PROFIT)

if __name__ == '__main__':
    VS = {}
    V = np.zeros((NV,), dtype=np.float)
    S = np.arange(1, NS + 1)
    PI = np.zeros((NS,), dtype=np.int)
    sweeps = 0
    delta = 1
    V[-1] = 1  # terminal state -- PROFIT
    while delta > THRESHOLD:
        delta = 0
        # Iterate forwards -- skipping terminal state
        for i, state in enumerate(S):
            v = V[state]
            # maximum bet to reach target
            amax = min(state, GOAL - state)
            A = np.arange(1, amax + 1)

            E = [P_H * GAMMA * V[state + action] +
                 P_T * GAMMA * V[state - action]
                 for action in A]
            V[state] = round(np.max(E), 4)
            PI[state - 1] = A[np.argmax(E)]
            delta = max(delta, np.abs(V[state] - v))

            print(state, delta, state + PI[state - 1], v, V[state - 1])
        sweeps += 1
        VS[sweeps] = V

    _, ax = plt.subplots()
    ax.set_xlabel('state')
    ax.set_ylabel('V[state]')
    ax.plot(S, VS[1][1:-1], 'b-', label='Sweep 1')
    ax.plot(S, VS[2][1:-1], 'c-', label='Sweep 2')
    ax.plot(S, VS[sweeps][1:-1], 'r-', label=f'Sweep {sweeps:02d}')
    plt.show()

    _, ax = plt.subplots()
    ax.set_xlabel('state')
    ax.set_ylabel('policy')
    ax.plot(S, PI, 'r-')
    plt.show()

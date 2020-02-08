"""TD learning with function approximation"""

__author__ = 'Guilherme Varela'
__date__ = '2020-02-07'

import pdb
import numpy as np
from numpy.random import choice
from numpy.linalg import norm
import matplotlib.pyplot as plt

# CONSTANTS
TIME = 500

STATES = [s for s in range(7)]

ACTIONS = [0, 1]

POLICY = [1.0/7, 6.0/7]

T = np.stack(
     [[[0, 0, 0, 0, 0, 0, 1],
      [0, 0, 0, 0, 0, 0, 1],
      [0, 0, 0, 0, 0, 0, 1],
      [0, 0, 0, 0, 0, 0, 1],
      [0, 0, 0, 0, 0, 0, 1],
      [0, 0, 0, 0, 0, 0, 1],
      [0, 0, 0, 0, 0, 0, 1]],
     [[1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 0],
      [1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 0],
      [1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 0],
      [1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 0],
      [1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 0],
      [1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 0],
      [1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 0]]])

r = 0

GAMMA = 0.99

# Feature maps from actions 0 and 1 respectevely
PHI = np.stack(
    [[[2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
      [0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0]],
     [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]])

W0 = np.array(
    [1, 1, 1, 1, 1, 1, 10, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float).T

ALPHA = 0.01

def main(num_iterations, method='Q-Learning'):
    
    norms = np.zeros((num_iterations, TIME), dtype=float)
    for i in range(num_iterations):
        W = W0.copy()
        state = choice(STATES)
        act = choice(ACTIONS, p=POLICY)

        for t in range(TIME):
            qw = np.dot(PHI[act, state], W)

            # print(T[act, state])
            next_state = choice(STATES, p=T[act, state])
            
            # Q-learning
            if method == 'Q-Learning':
                # pdb.set_trace()
                opt = np.argmax(np.dot(PHI[:, next_state], W))
                qw1 = np.dot(PHI[opt, next_state], W)
                next_action = choice(ACTIONS, p=POLICY)

            else:
                # SARSA
                next_action = choice(ACTIONS, p=POLICY)
                qw1 = np.dot(PHI[next_action, next_state], W)

            W += ALPHA * PHI[act, state] * (r + GAMMA * qw1 - qw) 
            state = next_state
            act = next_action
            norms[i, t] = np.linalg.norm(W)
    return norms



if __name__ == '__main__':
    qlnorm = main(100)
    sarsanorm = main(100)
    # axes = plt.gca()
    # axes.set_ylim([-300, -25]) 
    plt.plot(np.mean(qlnorm, axis=0), color='r')
    plt.plot(np.mean(sarsanorm, axis=0), color='c')
    plt.show()
    pdb.set_trace()

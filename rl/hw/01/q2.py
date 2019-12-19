'''This is Value Iteration over finding the best value's policy
   for a gambler betting on coin tosses
'''

__author__ = 'Guilherme Varela'
__date__ = '2019-12-05'

import numpy as np
import pdb
from matplotlib import pyplot as plt

THRESHOLD = 1e-8
GOAL = 100
GAMMA = 1      # this is the discount factor
P_H = 0.4      # probability that a coin toss will comeout heads
P_T = 1 - P_H  # probability that a coin toss will comeout heads

def roundup(x, n=4):
    """ Rounds up
    
    Parameters:
    ----------
    * x : float or array-like
    * n : int

    Returns:
    --------
    * y : float or array-like
        same type as input

    """

    fct = np.power(10, n)
    return np.ceil(x * fct) / fct

if __name__ == '__main__':
    # States: that are effectevely visited by the agent
    # S = (1, 2, 3, ..., 99)
    S = np.arange(1, GOAL)
    # Value: # states + "2 special states"
    # V = (0, 1, 2, ..., 99, 100)
    V = np.zeros((GOAL + 1,), dtype=np.float)
    # Rewards: always zero except on GOAL
    R = np.zeros((GOAL + 1,), dtype=np.float)
    R[GOAL] = 1
    # Policies: same number of states
    PI = np.zeros((GOAL - 1,), dtype=np.int)
    sweeps = 0
    delta = 1
 
    sweeps_value = {1: None, 2: None, 3: None, 32: None}
    sweeps_pi = {1: None, 2: None, 3: None, 32: None}
    while delta > THRESHOLD or sweeps < 32:
        delta = 0
        # Iterate forwards
        for state in S:
            v = V[state]
            # maximum bet to reach target
            max_stake = min(state, GOAL - state)
            stakes = np.arange(1, max_stake + 1)

            H = np.array([
               R[state + stake] + GAMMA * V[state + stake]
               for stake in stakes
            ], dtype=np.float)

            T = np.array([
               R[state - stake] + GAMMA * V[state - stake]
               for stake in stakes
            ], dtype=np.float)

            E = roundup(P_H * H + P_T * T)

            V[state] = np.max(E)
            PI[state - 1] = stakes[np.argmax(E)]
            delta = max(delta, np.abs(V[state] - v))

        sweeps += 1
        if sweeps in sweeps_value:
            sweeps_value[sweeps] = V.copy()
            if sweeps == 32:
                print(f'Sweeps {sweeps} {V}')

        if sweeps in sweeps_pi:
            sweeps_pi[sweeps] = PI.copy()
            if sweeps == 32:
                print(f'Sweeps {sweeps} {PI}')

    legends = ('Sweep 1', 'Sweep 2', 'Sweep 3', f'Sweep {sweeps:02d}')
    _, ax = plt.subplots()
    ax.set_xlabel('Wealth\nState')
    ax.set_ylabel('Win. Probability\nV[State]')
    ax.plot(S, sweeps_value[1][1:-1], 'r-')
    ax.plot(S, sweeps_value[2][1:-1], 'm-')
    ax.plot(S, sweeps_value[3][1:-1], 'c-')
    ax.plot(S, sweeps_value[sweeps][1:-1], 'b-')
    plt.legend(legends)
    plt.title('Gambler\'s Problem:\nWinning Probability')
    plt.show()

    _, ax = plt.subplots()
    ax.set_xlabel('Wealth\n(State)')
    ax.set_ylabel('Stake\n(Policy)')
    plt.title('Gambler\'s Problem:\nStakes')

    ax.bar(range(1, GOAL), sweeps_pi[sweeps], width=1)
    plt.legend((f'Sweeps {sweeps}',))
    plt.show()

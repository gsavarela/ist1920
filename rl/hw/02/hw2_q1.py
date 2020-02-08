"""The cliff problem which is a variation of the GridWorld problem"""

__author__ = 'Guilherme Varela'
__date__ = '2020-02-06'
import pdb
from tqdm import tqdm
from numpy.random import choice, rand
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
TIME = 500
GRID_ROWS = 4
GRID_COLUMNS = 12

# creates slices
START = (GRID_ROWS - 1, 0)
GOAL = (GRID_ROWS - 1, GRID_COLUMNS - 1)
CLIFF = [(GRID_ROWS - 1, j) for j in range(1, GRID_COLUMNS - 1)]
# actions encode the inc for the current
# 2D position wrt rows and columns
# e.g (1, 0) -> `down` and (0, -1) -> `left`
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
STATES = [(x, y) for x in range(GRID_ROWS) for y in range(GRID_COLUMNS)]
Q = {}
ALPHA = 0.5
GAMMA = 1.0

def make_world():
    '''2D matrix representing the rewards'''
    rewards = -np.ones((GRID_ROWS, GRID_COLUMNS), dtype=int)
    rewards[START] = 0
    for cl in CLIFF:
        rewards[cl] = -100

    rewards[GOAL] = 0
    return rewards

def get_actions(coord):
    '''Define available actions given positions
        
       Assures agent is always on board
      
       Params:
        * coordinates
    '''
    actions = [act for act in ACTIONS
              if coord[0] + act[0] >= 0 and coord[0] + act[0] < GRID_ROWS and
                 coord[1] + act[1] >= 0 and coord[1] + act[1] < GRID_COLUMNS]
                    
    return actions

def Q_init(q=Q):
    for state in STATES:
        q[state] = {}
        qs = q[state]
        for action in get_actions(state):
            qs[action] = 0

def Q_accumulate(Qtot, batch):
    for state in STATES:
        qt = Qtot[state]
        for action in get_actions(state):
            qt[action] = (qt[action] * batch + qt[action]) / (batch + 1)

def Q2pol(Qtot):
    data = {}
    for x in range(GRID_ROWS):
        columns = []
        for y in range(GRID_COLUMNS):
            state = (x, y)
            if state in CLIFF:
                columns.append('C')
            elif state == GOAL:
                columns.append('G')
            else:
                opt = optact(Qtot[state])
                if opt == (-1, 0):
                    columns.append('^')
                elif opt == (1, 0):
                    columns.append('V')
                elif opt == (0, -1):
                    columns.append('<')
                else:
                    columns.append('>')
        data[x + 1] = columns

    dfpol = pd.DataFrame.from_dict(data=data, orient='index',
                                   columns=range(1, GRID_COLUMNS + 1))
    return dfpol

def act2dict(actions, actexcl):
    return \
        {i: act for i, act in enumerate(actions) if act != actexcl}

def eps_greedy(actions_values, eps=0.15):
    '''Chooses action according to a randomized strategy'''
    chosen = optact(actions_values)
    if rand() < eps:
            actdict = act2dict(actions_values.keys(), chosen)
            key = choice(list(actdict.keys()))
            chosen = actdict[key]
    return chosen

def optact(actions_values):
    chosen, _ = max(actions_values.items(), key=lambda x: x[1])
    return chosen


def main(batches, method='QL'):
    '''
                 (1, 3)
                    ^
                    |
                    | (-1, 0)
    (2, 2) <---- (2, 3) ----> (2, 4)
            (0, -1) |    (0, 1)
                    |
                    V (1, 0)
    '''
    if method not in ('QL', 'SARSA'):
        raise ValueError("Criteria {method} unknown")
    else:
        is_ql = method == 'QL'

    # initializations
    rewards = make_world()
    returns = np.zeros((batches, TIME), dtype=np.float)
    # this Q is going to be used to generate a new policy
    Qtot = {}
    Q_init(q=Qtot)
    with tqdm(total=batches * TIME) as pb:
        for b in range(batches):
            Q_init()
            for t in range(TIME):
                state = START
                step = 0
                update_action = eps_greedy(Q[START])
                while state != GOAL:
                    step += 1
                    action = eps_greedy(Q[state]) if is_ql else update_action
                    next_state = (action[0] + state[0], action[1] + state[1])

                    reward = rewards[next_state]
                    # in case of cliff teleport
                    if next_state in CLIFF:
                        next_state = START
                    # This action is greedy for QL
                    # and eps-greedy for SARSA
                    update_action = optact(Q[next_state]) \
                        if is_ql else eps_greedy(Q[next_state])

                    Q[state][action] += \
                        ALPHA * (reward + GAMMA *
                                 Q[next_state][update_action] - Q[state][action])
                    returns[b, t] += reward
                    state = next_state
                pb.update(1)
            Q_accumulate(Qtot, b)
    # for each state recover the best action
    dfpol = Q2pol(Qtot)

    print(tabulate(dfpol))
    return returns, dfpol


        
if __name__ == '__main__':
    ql, dfpol = main(5, method='QL')
    sarsa, dfpol = main(5, method='SARSA')
    # ql = main(500, method='QL')
    # sarsa = main(500, method='SARSA')
    axes = plt.gca()
    axes.set_ylim([-300, -25]) 
    plt.plot(np.mean(sarsa, axis=0), color='c')
    plt.plot(np.mean(ql, axis=0), color='r')
    plt.show()
    pdb.set_trace()



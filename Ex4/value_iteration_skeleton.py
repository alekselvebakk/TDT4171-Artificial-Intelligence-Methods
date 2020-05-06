from typing import List, Any, NamedTuple, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Key = State, Value = Reward
reward_matrix: Dict[int, float] = {0: -0.1, 1: -0.1, 2: -0.1, 3: -0.1, 4: -0.1, 5: -1.0, 6: -0.1, 7: -1.0, 8: -0.1,
                                   9: -0.1, 10: - 0.1, 11: -1.0, 12: -1.0, 13: -0.1, 14: -0.1, 15: 1}

# Key = State, Value = Action
# Key = Action, Value = List of transition probability, outcome state tuple

transition_matrix: Dict[int, Dict[int, List[Tuple[float, int]]]] = \
    {0: {0: [(9 / 10, 0), (1 / 10, 4)], 1: [(1 / 10, 0), (8 / 10, 4), (1 / 10, 1)],
         2: [(1 / 10, 4), (8 / 10, 1), (1 / 10, 0)],
         3: [(1 / 10, 1), (9 / 10, 0)]},
     1: {0: [(1 / 10, 1), (8 / 10, 0), (1 / 10, 5)], 1: [(1 / 10, 0), (8 / 10, 5), (1 / 10, 2)],
         2: [(1 / 10, 5), (8 / 10, 2), (1 / 10, 1)],
         3: [(1 / 10, 2), (8 / 10, 1), (1 / 10, 0)]},
     2: {0: [(1 / 10, 2), (8 / 10, 1), (1 / 10, 6)], 1: [(1 / 10, 1), (8 / 10, 6), (1 / 10, 3)],
         2: [(1 / 10, 6), (8 / 10, 3), (1 / 10, 2)],
         3: [(1 / 10, 3), (8 / 10, 2), (1 / 10, 1)]},
     3: {0: [(1 / 10, 3), (8 / 10, 2), (1 / 10, 7)], 1: [(1 / 10, 2), (8 / 10, 7), (1 / 10, 3)],
         2: [(1 / 10, 7), (9 / 10, 3)],
         3: [(9 / 10, 3), (1 / 10, 2)]},
     4: {0: [(1 / 10, 0), (8 / 10, 4), (1 / 10, 8)], 1: [(1 / 10, 4), (8 / 10, 8), (1 / 10, 5)],
         2: [(1 / 10, 8), (8 / 10, 5), (1 / 10, 0)],
         3: [(1 / 10, 5), (8 / 10, 0), (1 / 10, 4)]},
     5: {0: [(1.0, 5)], 1: [(1.0, 5)], 2: [(1.0, 5)], 3: [(1.0, 5)]},
     6: {0: [(1 / 10, 2), (8 / 10, 5), (1 / 10, 10)], 1: [(1 / 10, 5), (8 / 10, 10), (1 / 10, 7)],
         2: [(1 / 10, 10), (8 / 10, 7), (1 / 10, 2)],
         3: [(1 / 10, 7), (8 / 10, 2), (1 / 10, 5)]},
     7: {0: [(1.0, 7)], 1: [(1.0, 7)], 2: [(1.0, 7)], 3: [(1.0, 7)]},
     8: {0: [(1 / 10, 4), (8 / 10, 8), (1 / 10, 12)], 1: [(1 / 10, 8), (8 / 10, 12), (1 / 10, 9)],
         2: [(1 / 10, 12), (8 / 10, 9), (1 / 10, 4)],
         3: [(1 / 10, 9), (8 / 10, 4), (1 / 10, 8)]},
     9: {0: [(1 / 10, 5), (8 / 10, 8), (1 / 10, 13)], 1: [(1 / 10, 8), (8 / 10, 13), (1 / 10, 10)],
         2: [(1 / 10, 13), (8 / 10, 10), (1 / 10, 5)],
         3: [(1 / 10, 10), (8 / 10, 5), (1 / 10, 8)]},
     10: {0: [(1 / 10, 6), (8 / 10, 9), (1 / 10, 14)], 1: [(1 / 10, 9), (8 / 10, 14), (1 / 10, 11)],
          2: [(1 / 10, 14), (8 / 10, 11), (1 / 10, 6)],
          3: [(1 / 10, 11), (8 / 10, 6), (1 / 10, 9)]},
     11: {0: [(1.0, 11)], 1: [(1.0, 11)], 2: [(1.0, 11)], 3: [(1.0, 11)]},
     12: {0: [(1.0, 12)], 1: [(1.0, 12)], 2: [(1.0, 12)], 3: [(1.0, 12)]},
     13: {0: [(1 / 10, 9), (8 / 10, 12), (1 / 10, 13)], 1: [(1 / 10, 12), (8 / 10, 13), (1 / 10, 14)],
          2: [(1 / 10, 13), (8 / 10, 14), (1 / 10, 9)], 3: [(1 / 10, 14), (8 / 10, 9), (1 / 10, 12)]},
     14: {0: [(1 / 10, 10), (8 / 10, 13), (1 / 10, 14)], 1: [(1 / 10, 13), (8 / 10, 14), (1 / 10, 15)],
          2: [(1 / 10, 14), (8 / 10, 15), (1 / 10, 10)], 3: [(1 / 10, 15), (8 / 10, 10), (1 / 10, 13)]},
     15: {0: [(1.0, 15)], 1: [(1.0, 15)], 2: [(1.0, 15)], 3: [(1.0, 15)]}}

moves: Dict[str, int] = {"left": 0, "down": 1, "right": 2, "up": 3}


# ------------------------------- Nothing you need or need to do above this line ---------------------------------------

def get_outcome_states(state: int, action: str) -> List[int]:
    """
    Fetch the possible outcome states given the current state and action taken.
    :param state: The current state which is a number between 0-15 as there are 16 states (16 tiles).
    :param action: The action taken which is a string, either: left, down, right, or up.
    :return: A list of possible outcome states. Each state is a number between 0-15 as there are 16 states (16 tiles).
    """
    assert isinstance(state, int) and 0 <= state < 16, "States must be an integer between 0 - 15."
    assert action in moves.keys(), "Action must be either: left, down, right, or up."

    return [next_state for _, next_state in transition_matrix[state][moves[action]]]


def get_transition_probability(state: int, action: str, outcome_state: int) -> float:
    """
    Fetch the transition probability for the provided outcome states given the current state and action taken.
    :param state: The current state which is a number between 0-15 as there are 16 states (16 tiles)
    :param action: The action taken which is a string, either: left, down, right, or up.
    :param outcome_state: The outcome state which is a number between 0-15 as there are 16 states (16 tiles). However,
    not all of the 16 states are possible, but depends on your current state.
    :return: The transition probability.
    """
    assert isinstance(state, int) and 0 <= state < 16, "States must be an integer between 0 - 15."
    assert action in moves.keys(), "Action must be either: left, down, right, or up."
    assert isinstance(outcome_state, int) and 0 <= outcome_state < 16, "States must be an integer between 0 - 15."

    return {next_state: trans_prob for trans_prob, next_state in transition_matrix[state][moves[action]]}[
        outcome_state]


def get_reward(state: int) -> float:
    """
    Fetch the reward given the current state.
    :param state: The current state which is a number between 0-15 as there are 16 states (16 tiles).
    :return: The reward.
    """
    assert isinstance(state, int) and 0 <= state < 16, "States must be an integer between 0 - 15."

    return reward_matrix[state]


class Constants(NamedTuple):
    """
    This class provides the necessary constants you need to implement the value iteration algorithm.
    In order to access the field you can write e.g. constants.gamma if you need to access the field gamma.

    Attributes
    ----------
    number_states : int
        Number of states in this Markov decision process.
    number_actions : int
        The size of the action space.
    gamma: float
        The discount factor.
    epsilon: float
        The maximum error allowed in the utility of any state.
    """
    number_states: int = 16
    number_actions: int = 4
    epsilon: float = 1e-20
    gamma: float = 0.9


# This variable contains all constants you will need
constants: Constants = Constants()
def action(U_list, state, action):
    sum = 0;
    possible_states = get_outcome_states(state, action)
    for i in possible_states:
        sum = sum + get_transition_probability(state, action, i)*U_list[i]
    return sum



def max_action(U_list, state):
    maximum = max(action(U_list, state, "left"), action(U_list, state, "right"), action(U_list, state, "down"), action(U_list, state, "up"))

    return maximum

def updateU(state, U_list):
    U = get_reward(state)+ constants.gamma*max_action(U_list, state)
    return U

def updateList(U_list):
    U_new = U_list
    for state in range(16):
        U_new[state] = updateU(state, U_list)
    return U_new

def createFig(tableToBePrinted):
    #Rounding off to two decimals
    for i in range(16):
        tableToBePrinted[i] = np.around(tableToBePrinted[i], decimals=2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    col_labels = ['1', '2', '3', '4']
    row_labels = ['1', '2', '3', '4']
    table_vals = np.array([tableToBePrinted[0:4], tableToBePrinted[4:8], tableToBePrinted[8:12], tableToBePrinted[12:16]])
    the_table = plt.table(  cellText=table_vals,
                            rowLabels=row_labels,
                            colLabels=col_labels,
                            colWidths=[0.06] * 4,
                            cellLoc = 'center',
                            loc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(24)
    the_table.scale(4, 4)


    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right','top','bottom','left']:
        plt.gca().spines[pos].set_visible(False)
    plt.savefig('matplotlib-table.png')


def value_iteration() -> Any:
    """
    Implement the value iteration algorithm described in Figure 17.4 in the book.

    Note: Everything you need to implement this algorithm is either stored in the variable "constants" above or can be
    accessed through one of the functions above.

    :return: The converged utility values of all states.
    """
    # TODO: Implement the method.
    U_old =np.array([   0, #state 0
                            0, #state 1
                            0, #state 2
                            0, #state 3
                            0, #state 4
                            0, #state 5
                            0, #state 6
                            0, #state 7
                            0, #state 8
                            0, #state 9
                            0, #state 10
                            0, #state 11
                            0, #state 12
                            0, #state 13
                            0, #state 14
                            0], float) #state 15
    U_new = np.copy(U_old)
    error = float(10)
    j = 1;
    while error >= constants.epsilon:
        U_new = updateList(U_new)
        diff = U_new-U_old
        error = np.linalg.norm(diff)
        U_old = np.copy(U_new)
        j = j+1

    return U_old

def createTextFig(tableToBePrinted):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    col_labels = ['1', '2', '3', '4']
    row_labels = ['1', '2', '3', '4']
    table_vals = np.array([tableToBePrinted[0:4], tableToBePrinted[4:8], tableToBePrinted[8:12], tableToBePrinted[12:16]])
    the_table = plt.table(  cellText=table_vals,
                            rowLabels=row_labels,
                            colLabels=col_labels,
                            colWidths=[0.06] * 4,
                            cellLoc = 'center',
                            loc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(24)
    the_table.scale(4, 4)


    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right','top','bottom','left']:
        plt.gca().spines[pos].set_visible(False)
    plt.savefig('matplotlib-table-text.png')


def return_max_action(value_table, state):
    maxvalue = 0
    maxaction = ''
    right = action(value_table, state, 'right')
    if right > maxvalue:
        maxvalue = right
        maxaction = 'right'
    left = action(value_table, state, 'left')
    if left > maxvalue:
        maxvalue = left
        maxaction = 'left'
    up = action(value_table, state, 'up')
    if up > maxvalue:
        maxvalue = up
        maxaction = 'up'
    down = action(value_table, state, 'down')
    if down > maxvalue:
        maxvalue = down
        maxaction = 'down'
    return maxaction




def extract_policy(value_table: Any) -> Any:
    """
    Extract policy based on the given value_table.
    :param value_table: Some data structure containing the converged utility values.
    :return: The extracted policy.
    """
    # TODO: Implement the method.
    pi_list = [ '', #STATE0
                '', #STATE1
                '', #STATE2
                '', #STATE3
                '', #STATE4
                '', #STATE5
                '', #STATE6
                '', #STATE7
                '', #STATE8
                '', #STATE9
                '', #STATE10
                '', #STATE11
                '', #STATE12
                '', #STATE13
                '', #STATE14
                ''] #STATE15
    for i in range(16):
        pi_list[i] = return_max_action(value_table, i)
    return pi_list


def main() -> None:
    """
    Run the script.
    :return: Nothing.
    """
    value_table = value_iteration()

    createFig(value_table)



    optimal_policy = extract_policy(value_table)
    createTextFig(optimal_policy)

if __name__ == '__main__':
    main()

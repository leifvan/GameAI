import numpy as np
from itertools import groupby
from operator import itemgetter
from random import choice
from functools import partial
from numba import njit

from game import TurnBasedGame, run_multiple_games


class ConnectFourGame(TurnBasedGame):
    num_players = 2

    def get_initial_game_state(self):
        return np.zeros((6, 7), dtype=int)

    @staticmethod
    def victory_condition(game_state, zero_padded_state=np.zeros((7,8))):
        #zero_padded_state = np.pad(game_state, ((0, 0), (1, 1)), mode='constant')
        zero_padded_state[:-1,:-1] = game_state

        # check rows
        p, max_len = get_max_consecutive_row_length(zero_padded_state.ravel())
        if max_len == 4:
            return p

        # check cols
        p, max_len = get_max_consecutive_row_length(np.transpose(zero_padded_state).ravel())
        if max_len == 4:
            return p

        # check diagonals
        diagonals = concat_diagonals(zero_padded_state)
        p, max_len = get_max_consecutive_row_length(diagonals)
        if max_len == 4:
            return p

        # check anti-diagonals
        diagonals = concat_diagonals(np.fliplr(zero_padded_state))
        p, max_len = get_max_consecutive_row_length(diagonals)
        if max_len == 4:
            return p

        return None

    @staticmethod
    def end_condition(game_state):
        return np.count_nonzero(game_state == 0) == 0

    @staticmethod
    def legal_state_condition(game_state):
        diff = np.diff((game_state != 0).astype(int), axis=0)
        return np.all(diff >= 0)


@njit
def concat_diagonals(arr):
    max_len = min(arr.shape)
    num_items = 0

    # count
    for o in range(-3, 5):
        for i in range(max_len):
            if 0 <= i+o < arr.shape[1]:
                num_items += 1

    diags = np.zeros(num_items, dtype=np.int32)

    c = 0
    for o in range(-3, 5):
        for i in range(max_len):
            if 0 <= i+o < arr.shape[1]:
                diags[c] = arr[i, i + o]
                c += 1

    return diags


@njit
def get_max_consecutive_row_length(flat_rows):
    max_len = 0
    max_x = 0

    cur_len = 0
    cur_x = -1

    for x in flat_rows:
        if x != cur_x:
            if cur_x != 0 and max_len < cur_len:
                max_len = cur_len
                max_x = cur_x
            cur_len = 0
            cur_x = x
        cur_len += 1

    return max_x, max_len


def get_legal_positions(game_state):
    one_padded_state = np.pad(game_state, ((0,1),(0,0)), mode='constant', constant_values=1)
    diff = np.diff((one_padded_state != 0).astype(int), axis=0)
    return np.transpose(np.nonzero(diff))


def random_strategy(game_state, p):
    positions = get_legal_positions(game_state)
    pos = choice(positions)
    game_state[pos[0], pos[1]] = p
    return game_state


def winning_or_random_move(game_state, p):
    positions = get_legal_positions(game_state)

    # check if victory is possible
    for pos in positions:
        game_state[pos[0], pos[1]] = p
        if ConnectFourGame.victory_condition(game_state):
            return game_state
        game_state[pos[0], pos[1]] = 0

    # check if victory of opponent has to be prevented
    op = 1 if p == 2 else 2
    for pos in positions:
        game_state[pos[0], pos[1]] = op
        if ConnectFourGame.victory_condition(game_state):
            game_state[pos[0], pos[1]] = p
            return game_state
        game_state[pos[0], pos[1]] = 0

    # pick random
    pos = choice(positions)
    game_state[pos[0], pos[1]] = p
    return game_state


if __name__ == '__main__':
    # game = ConnectFourGame()
    # game.game_state = np.array([[0,0,0,0,0,0,0],
    #                             [0,0,0,0,0,0,0],
    #                             [0,0,0,0,0,0,0],
    #                             [2,0,0,0,0,0,1],
    #                             [0,2,0,0,0,1,0],
    #                             [0,0,2,0,1,0,0],
    #                             [0,0,0,2,0,0,0]])
    # print(game.victory_condition(game.game_state))
    #
    # exit(0)

    strats = [partial(winning_or_random_move, p=1),
              partial(winning_or_random_move, p=2)]

    stats = run_multiple_games(ConnectFourGame, strats, num_games=1000)

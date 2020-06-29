from collections import defaultdict
from functools import partial
from itertools import product

import numpy as np
from numba import njit
from tqdm import tqdm

from games.game import TurnBasedGame

rng = np.random.default_rng()

# python dictionary to map integers (1, -1, 0) to characters ('x', 'o', ' ')
symbols = {1: 'x', -1: 'o', 0: ' '}


class TicTacToeGame(TurnBasedGame):
    num_players = 2

    @staticmethod
    def get_initial_game_state():
        return np.zeros((3, 3))

    @staticmethod
    def victory_condition(game_state):
        if move_was_winning_move(game_state, 1):
            return 0
        elif move_was_winning_move(game_state, -1):
            return 1
        return None

    @staticmethod
    def end_condition(game_state):
        return not move_still_possible(game_state)

    @staticmethod
    def legal_state_condition(game_state):
        return True

    def print_state(self):
        for row in self.game_state:
            print(' '.join(symbols[v] for v in row))
        print("-" * 5)

    @staticmethod
    @njit
    def possible_moves(game_state):
        positions = np.argwhere(game_state == 0)
        cur_player = 1 if len(positions) % 2 == 1 else -1
        return [(cur_player, pos) for pos in positions]

    @staticmethod
    @njit
    def make_move(game_state, move):
        new_state = np.copy(game_state)
        pl, pos = move
        new_state[pos[0], pos[1]] = pl
        return new_state


@njit
def move_still_possible(S):
    return np.count_nonzero(S) != 9


@njit
def move_was_winning_move(S, p):
    diag_acc = antidiag_acc = 0
    for i in range(3):
        row_acc = col_acc = 0
        for j in range(3):
            row_acc += S[i, j] == p
            col_acc += S[j, i] == p

        if row_acc == 3 or col_acc == 3:
            return True

        diag_acc += S[i, i] == p
        antidiag_acc += S[2 - i, i] == p

    if diag_acc == 3 or antidiag_acc == 3:
        return True

    return False


def hash_flat_state(flat_state):
    return sum((3 ** i) * (2 if s == -1 else s) for i, s in enumerate(flat_state))


class TicTacToeStateHasher:
    # TODO is not working btw
    def __init__(self):
        self.flat_hash_group_map = dict()
        cur_state = 0

        for state_flat in product([0, 1, -1], repeat=9):
            state = np.reshape(state_flat, (3, 3))

            if hash_flat_state((state_flat)) in self.flat_hash_group_map:
                continue

            #similar_states = [state, np.fliplr(state.copy()), np.flipud(state.copy())]
            similar_states = [state]

            rot = state.copy()
            for _ in range(3):
                rot = np.rot90(rot.copy())
                similar_states.append(rot)

            unique_hashes = [hash_flat_state(s.ravel()) for s in similar_states]

            # # check if one of the states is already in map
            # g = None
            #
            # for h in unique_hashes:
            #     if h in self.flat_hash_group_map:
            #         g = self.flat_hash_group_map[h]
            #         break
            #
            # if not g:
            g = cur_state
            cur_state += 1

            self.flat_hash_group_map.update({h: g for h in unique_hashes})

        # group_flat_hash_map = defaultdict(list)
        #
        # for h, g in self.flat_hash_group_map.items():
        #     group_flat_hash_map[g].append(h)

    def __call__(self, game_state):
        return self.flat_hash_group_map[hash_flat_state(game_state.ravel())]


def move_at_random(S, p):
    xs, ys = np.where(S == 0)
    i = rng.integers(0, xs.size)
    S[xs[i], ys[i]] = p
    return S


def move_at_random_weighted(game_state, p, weights):
    positions = np.flatnonzero(game_state == 0)

    relevant_weights = np.ravel(weights)[positions]
    relevant_weights /= np.sum(relevant_weights)

    target = rng.choice(positions, p=relevant_weights)
    target = np.unravel_index(target, game_state.shape)
    game_state[target] = p
    return game_state


def select_winning_move_or_weighted(game_state, p, weights):
    positions = np.transpose(np.nonzero(game_state == 0))

    # check if this player can win
    for px, py in positions:
        game_state[px, py] = p
        if move_was_winning_move(game_state, p):
            return game_state
        game_state[px, py] = 0

    # check if other player can win and block
    for px, py in positions:
        game_state[px, py] = -p
        if move_was_winning_move(game_state, -p):
            game_state[px, py] = p
            return game_state
        game_state[px, py] = 0

    # choose randomly
    return move_at_random_weighted(game_state, p, weights)


class SelectBestMinMaxStrategy:
    def __init__(self, player_type, tie_strategy, game_tree, depth=2 ** 32, eval_fn=None):
        self.game_tree = game_tree
        self.cur_node = self.game_tree.root
        self.player_type = player_type
        self.tie_strategy = tie_strategy
        self.depth = depth
        self.eval_fn = eval_fn

    def __call__(self, game_state):
        # reset if game_state is initial
        if np.count_nonzero(game_state) < 2:
            self.cur_node = self.game_tree.root

        # find the node for corresponding game state
        for child in self.cur_node.children:
            if np.allclose(self.game_tree.get_state(child), game_state):
                self.cur_node = child
                break

        _, best_node = self.cur_node.mmv(self.player_type, self.tie_strategy, self.depth, self.eval_fn)
        new_state = TicTacToeGame.make_move(game_state, best_node.move)
        self.cur_node = best_node
        return new_state


class SelectBestAlphaBetaStrategy:
    def __init__(self, player_type, game_tree):
        self.game_tree = game_tree
        self.cur_node = self.game_tree.root
        self.player_type = player_type

    def __call__(self, game_state):
        # reset if game_state is initial
        if np.count_nonzero(game_state) < 2:
            self.cur_node = self.game_tree.root

        # find the node for corresponding game state
        for child in self.cur_node.children:
            if np.allclose(self.game_tree.get_state(child), game_state):
                self.cur_node = child
                break

        _, best_node = self.cur_node.alpha_beta(self.player_type)
        new_state = TicTacToeGame.make_move(game_state, best_node.move)
        self.cur_node = best_node
        return new_state


if __name__ == '__main__':

    # completely random strategies
    random_strats = [partial(move_at_random, p=p) for p in [1, -1]]

    # player 1 chooses randomly with distribution from auspicious positions after 100,000 random games
    win_probs = np.array([[0.11811497, 0.09795438, 0.11796465],
                          [0.09668166, 0.14078844, 0.09746834],
                          [0.11683974, 0.09710506, 0.11708276]])

    auspicious_strats = [partial(move_at_random_weighted, p=1, weights=win_probs),
                         partial(move_at_random, p=-1)]

    # player 1 checks for a winning move before using auspicious position strategy
    winning_move_strats = [partial(select_winning_move_or_weighted, p=1, weights=win_probs),
                           partial(move_at_random, p=-1)]

    # player 1 plays with min max
    game_tree = TicTacToeGame.get_full_game_tree(hash_fn=TicTacToeStateHasher())

    minmax_strats = [SelectBestMinMaxStrategy('max', 'best', game_tree), partial(move_at_random, p=-1)]
    alpha_beta_strats = [SelectBestAlphaBetaStrategy('max', game_tree), partial(move_at_random, p=-1)]

    # player 1 plays with depth-restricted min max

    @njit
    def count_win_lines(game_state, player):
        row_hits = [0, 0, 0]
        col_hits = [0, 0, 0]
        diag_hits = anti_diag_hits = 0

        for i in range(3):
            for j in range(3):
                if game_state[i,j] == player or game_state[i,j] == 0:
                    row_hits[i] += 1
                    col_hits[j] += 1

                    if i == j:
                        diag_hits += 1
                    elif i - 2 == j:
                        anti_diag_hits += 1

        hits = int(diag_hits == 3) + int(anti_diag_hits == 3)
        for r,c in zip(row_hits, col_hits):
            hits += int(r == 3) + int(c == 3)

        return hits

    def eval_fn(node):
        game_state = game_tree.get_state(node)
        return count_win_lines(game_state, 1) - count_win_lines(game_state, -1)

    dr_minmax_strats = [SelectBestMinMaxStrategy('max', 'best', game_tree, depth=10, eval_fn=eval_fn),
                        partial(move_at_random, p=-1)]

    strats_to_use = minmax_strats

    # --------------------------------------
    # run a single game and print all states
    # --------------------------------------

    winner, _ = TicTacToeGame().run(strats_to_use, print_states=True)

    print("player", winner, "won the game")

    # ------------------------------
    # run 10,000 games and evaluate:
    # ------------------------------

    wins = defaultdict(int)
    auspicious_positions = np.zeros((3, 3), dtype=int)

    for i in tqdm(range(10000)):
        winner, final_state = TicTacToeGame().run(strats_to_use)
        wins[winner] += 1

        if winner == 0:
            auspicious_positions[final_state == 1] += 1
        elif winner == 1:
            auspicious_positions[final_state == -1] += 1

    print("x:   ", wins[0])
    print("o:   ", wins[1])
    print("draw:", wins[None])

    print("auspicious positions:")
    print(auspicious_positions / np.sum(auspicious_positions))

    # --------------------------------------------
    # generate a game tree and get some statistics
    # --------------------------------------------

    tree = TicTacToeGame.get_full_game_tree()
    num_nodes, num_leafs, num_branches = tree.get_statistics()
    b = num_branches / (num_nodes - num_leafs)

    print()
    print()
    print(num_nodes, "nodes == state space complexity")
    print(num_leafs, "leafs == game tree size")
    print(f"{b:.5} average branching factor")

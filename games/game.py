from abc import ABC, abstractmethod
from collections import defaultdict
from collections import deque
from operator import itemgetter
from typing import Type

from tqdm import tqdm


# victory_condition(game_state) => winner or None
# end_condition(game_state) => True (no more moves) or False
# player_strategy(game_state) => new game state


class GameEndedException(Exception):
    pass


class IllegalMoveException(Exception):
    pass


class TurnBasedGame(ABC):
    num_players = 1

    def __init__(self):
        self.game_state = self.get_initial_game_state()
        self.player = 0
        self.winner = None
        self.game_ended = False

    @staticmethod
    @abstractmethod
    def get_initial_game_state():
        ...

    @staticmethod
    @abstractmethod
    def victory_condition(game_state):
        ...

    @staticmethod
    @abstractmethod
    def end_condition(game_state):
        ...

    @staticmethod
    @abstractmethod
    def legal_state_condition(game_state):
        ...

    @staticmethod
    @abstractmethod
    def possible_moves(game_state):
        ...

    @staticmethod
    @abstractmethod
    def make_move(game_state, move):
        ...

    def print_state(self):
        print(self.game_state)

    def move(self, new_game_state):
        if self.game_ended:
            raise GameEndedException

        if not self.legal_state_condition(new_game_state):
            raise IllegalMoveException

        self.game_state = new_game_state
        self.player = (self.player + 1) % self.num_players

        self.winner = self.victory_condition(self.game_state)
        self.game_ended = self.winner or self.end_condition(self.game_state)

    def run(self, player_strategies, print_states=False):
        if print_states:
            self.print_state()

        while not self.winner and not self.game_ended:
            self.move(player_strategies[self.player](self.game_state))

            if print_states:
                self.print_state()

        return self.winner, self.game_state

    @classmethod
    def run_multiple_games(cls, player_strategies, num_games):
        wins = defaultdict(int)

        with tqdm(range(num_games)) as pbar:
            for i in pbar:
                winner, _ = cls().run(player_strategies)
                wins[winner] += 1
                if i % 10 == 0:
                    pbar.set_postfix({f'p{p}' if p else 'draw': w for p, w in wins.items()})

        return wins

    @classmethod
    def get_full_game_tree(cls, initial_state=None, starting_player=0):
        initial_state = initial_state or cls.get_initial_game_state()
        game_tree = GameTree(cls, initial_state, starting_player)

        with tqdm() as pbar:
            expand_queue = deque([(game_tree.root, initial_state)])

            while len(expand_queue) > 0:
                node, state = expand_queue.popleft()
                game_tree.expand_node(node, state)
                expand_queue.extend([(child, cls.make_move(state, child.move)) for child in node.children])
                pbar.update(1)

        return game_tree


class GameTreeNode:
    def __init__(self, move, parent, utility=None):
        self.move = move
        self.parent = parent
        self.children = []

        self.utility = utility

    def path_to_root(self):
        path = [self]
        while path[-1].parent:
            path.append(path[-1].parent)
        return path

    def mmv(self, node_type, tie_strategy='choose_first'):
        if len(self.children) == 0:
            return self.utility, self

        if tie_strategy == 'choose_first':
            if node_type == 'max':
                return max(((child.mmv('min', tie_strategy)[0], child)
                            for child in self.children),
                           key=itemgetter(0))
            if node_type == 'min':
                return min(((child.mmv('max', tie_strategy)[0], child)
                            for child in self.children),
                           key=itemgetter(0))

            raise ValueError("node_type has to be one of 'max', 'min'.")

        elif tie_strategy == 'choose_best':
            if node_type == 'max':
                return max(((child.mmv('min', tie_strategy)[0], child, child.mmv('max', tie_strategy)[0])
                            for child in self.children),
                           key=itemgetter(0, 2))[:2]
            if node_type == 'min':
                return min(((child.mmv('max', tie_strategy)[0], child, child.mmv('min', tie_strategy)[0])
                            for child in self.children),
                           key=itemgetter(0, 2))[:2]

            raise ValueError("node_type has to be one of 'max', 'min'.")

        raise ValueError("tie_strategy has to be one of 'choose_first', 'choose_best'.")


class GameTree:
    def __init__(self, game_type: Type[TurnBasedGame], initial_state=None, starting_player=0):
        self.game_type = game_type

        if initial_state is None:
            self.initial_state = game_type.get_initial_game_state()
        else:
            self.initial_state = initial_state

        self.starting_player = starting_player
        self.root = GameTreeNode(move=None, parent=None)

    def get_state(self, node: GameTreeNode):
        path = node.path_to_root()[::-1]
        state = self.initial_state

        for path_node in path[1:]:
            state = self.game_type.make_move(state, path_node.move)

        return state

    def get_statistics(self):
        num_nodes = num_leafs = num_branches = 0
        stack = [self.root]

        while len(stack) > 0:
            node = stack.pop()
            num_nodes += 1

            if len(node.children) == 0:
                num_leafs += 1
            else:
                num_branches += len(node.children)

            stack.extend(node.children)

        return num_nodes, num_leafs, num_branches

    def expand_node(self, node: GameTreeNode, state=None):
        if state is None:
            state = self.get_state(node)

        if self.game_type.victory_condition(state) or self.game_type.end_condition(state):
            return

        possible_moves = self.game_type.possible_moves(state)

        for move in possible_moves:
            node.children.append(GameTreeNode(move=move, parent=node))


if __name__ == '__main__':
    # create min max tree for task 3.1
    tree = GameTree(game_type=None, initial_state=0)
    # n1
    n1 = GameTreeNode(None, parent=tree.root)
    tree.root.children.append(n1)
    n1.children.append(GameTreeNode(None, parent=n1, utility=15))
    n1.children.append(GameTreeNode(None, parent=n1, utility=20))
    n1.children.append(GameTreeNode(None, parent=n1, utility=1))
    n1.children.append(GameTreeNode(None, parent=n1, utility=3))

    # n2
    n2 = GameTreeNode(None, parent=tree.root)
    tree.root.children.append(n2)
    n2.children.append(GameTreeNode(None, parent=n2, utility=3))
    n2.children.append(GameTreeNode(None, parent=n2, utility=4))

    # n3
    n3 = GameTreeNode(None, parent=tree.root)
    tree.root.children.append(n3)
    n3.children.append(GameTreeNode(None, parent=n3, utility=15))
    n3.children.append(GameTreeNode(None, parent=n3, utility=10))

    # n4
    n4 = GameTreeNode(None, parent=tree.root)
    tree.root.children.append(n4)
    n4.children.append(GameTreeNode(None, parent=n4, utility=16))
    n4.children.append(GameTreeNode(None, parent=n4, utility=4))
    n4.children.append(GameTreeNode(None, parent=n4, utility=12))

    # n5
    n5 = GameTreeNode(None, parent=tree.root)
    tree.root.children.append(n5)
    n5.children.append(GameTreeNode(None, parent=n5, utility=15))
    n5.children.append(GameTreeNode(None, parent=n5, utility=12))
    n5.children.append(GameTreeNode(None, parent=n5, utility=8))

    print("mmv(n0) =", tree.root.mmv('max')[0])

    # -------------------------------------------------------------------------------------------
    # task 3.2 (but here n3 is better, because n1 would be chosen anyway because it is the first)
    # -------------------------------------------------------------------------------------------

    tree = GameTree(game_type=None, initial_state=0)

    n1 = GameTreeNode("n1", parent=tree.root)
    tree.root.children.append(n1)
    n1.children.append(GameTreeNode("n5", parent=n1, utility=16))
    n1.children.append(GameTreeNode("n6", parent=n1, utility=6))
    n1.children.append(GameTreeNode("n7", parent=n1, utility=16))
    n1.children.append(GameTreeNode("n8", parent=n1, utility=6))
    n1.children.append(GameTreeNode("n9", parent=n1, utility=5))

    n2 = GameTreeNode("n2", parent=tree.root)
    tree.root.children.append(n2)
    n2.children.append(GameTreeNode("n10", parent=n2, utility=7))
    n2.children.append(GameTreeNode("n11", parent=n2, utility=1))

    n3 = GameTreeNode("n3", parent=tree.root)
    tree.root.children.append(n3)
    n3.children.append(GameTreeNode("n12", parent=n3, utility=16))
    n3.children.append(GameTreeNode("n13", parent=n3, utility=18))
    n3.children.append(GameTreeNode("n14", parent=n3, utility=5))

    n4 = GameTreeNode("n4", parent=tree.root)
    tree.root.children.append(n4)
    n4.children.append(GameTreeNode("n15", parent=n4, utility=10))
    n4.children.append(GameTreeNode("n16", parent=n4, utility=2))

    fmmv, fnode = tree.root.mmv('max')
    bmmv, bnode = tree.root.mmv('max', 'choose_best')
    print("first mmv(n0) =", fmmv, fnode.move)
    print(" best mmv(n0) =", bmmv, bnode.move)

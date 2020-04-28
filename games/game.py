import asyncio
from inspect import iscoroutinefunction
from abc import ABC, abstractmethod

# victory_condition(game_state) => winner or None
# end_condition(game_state) => True (no more moves) or False
# player_strategy(game_state) => new game state


class TurnBasedGame(ABC):
    num_players = 1

    def __init__(self):
        self.game_state = self.get_initial_game_state()
        self.player = 0
        self.winner = None
        self.game_ended = False

    @abstractmethod
    def get_initial_game_state(self):
        ...

    @abstractmethod
    def victory_condition(self):
        ...

    @abstractmethod
    def end_condition(self):
        ...

    @abstractmethod
    def legal_state_condition(self):
        ...

    def print_state(self):
        print(self.game_state)

    def move(self, new_game_state):
        if self.game_ended:
            raise Exception("Game has already ended.")

        self.game_state = new_game_state

        if not self.legal_state_condition():
            raise Exception("Illegal state change.")

        self.player = (self.player + 1) % self.num_players

        self.winner = self.victory_condition()
        self.game_ended = self.end_condition()

    def run(self, player_strategies, print_states=False):
        if print_states:
            self.print_state()

        while not self.winner and not self.game_ended:
            self.move(player_strategies[self.player](self.game_state))

            if print_states:
                self.print_state()

        return self.winner, self.game_state


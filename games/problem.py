from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Collection, Sequence, Hashable

ST = TypeVar("ST", bound=Hashable)


class Problem(ABC, Generic[ST]):

    @abstractmethod
    def get_initial_state(self) -> ST:
        ...

    @abstractmethod
    def successor_function(self, state: ST) -> Collection[ST]:
        ...

    @abstractmethod
    def goal_test_function(self, state: ST) -> bool:
        ...

    @abstractmethod
    def path_cost_function(self, path: Sequence[ST]):
        ...


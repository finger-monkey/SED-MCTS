from sortedcontainers import SortedList
from typing import Any, Tuple, Iterable
from abc import ABC, abstractmethod
import math
import random

class Queue_Base(ABC):





    def __init__(self, max_size: int):
        self.max_size = max_size

        self.list: SortedList[Tuple[Any, float]] = SortedList(key=lambda x: -x[1])

        self._reward_values: SortedList[float] = SortedList()
        self.min_reward: float = float('-inf')

    @abstractmethod
    def append(self, state: Any, reward: float) -> bool:
        pass

    def __len__(self) -> int:
        return len(self.list)

    def __iter__(self) -> Iterable[Tuple[Any, float]]:
        return iter(self.list)


    def best_reward(self) -> float:
        if not self.list:
            return 0.0
        return self.list[0][1]

    def best(self) -> Any:
        if not self.list:
            return None, None
        return self.list[0]

    def random_sample(self) -> Any:
        if not self.list:
            return None, None
        return random.choice(self.list)

    def is_empty(self) -> bool:
        return not self.list

class Exp_Queue(Queue_Base):









    def append(self, state: Any, reward: float, threshold: float = 1e-5) -> bool:







        if math.isinf(reward) or math.isnan(reward):
            return False

        reward_values = self._reward_values

        pos = reward_values.bisect_left(reward)
        if pos > 0 and abs(reward_values[pos - 1] - reward) < threshold:
            return False
        if pos < len(reward_values) and abs(reward_values[pos] - reward) < threshold:
            return False

        lst = self.list
        if len(lst) < self.max_size:
            lst.add((state, reward))
            reward_values.add(reward)
        else:

            if reward <= self.min_reward:
                return False

            removed_state, removed_reward = lst.pop(-1)

            reward_values.remove(removed_reward)

            lst.add((state, reward))
            reward_values.add(reward)


        self.min_reward = lst[-1][1] if lst else float('-inf')
        return True

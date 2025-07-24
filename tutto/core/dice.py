from dataclasses import dataclass
from typing import List, Tuple, Dict
import random
from collections import Counter


@dataclass
class DiceRoll:
    """Represents a single roll of multiple dice."""
    values: List[int]
    
    def __post_init__(self):
        if not all(1 <= v <= 6 for v in self.values):
            raise ValueError("All dice values must be between 1 and 6")
    
    @property
    def count(self) -> int:
        """Number of dice in this roll."""
        return len(self.values)
    
    @property
    def value_counts(self) -> Counter:
        """Count of each dice value."""
        return Counter(self.values)
    
    def __str__(self) -> str:
        return f"DiceRoll({self.values})"


class Dice:
    """Manages dice rolling and state."""
    
    def __init__(self, num_dice: int = 6):
        self.num_dice = num_dice
        self.remaining_dice = num_dice
        
    def roll(self, num_to_roll: int = None) -> DiceRoll:
        """Roll dice and return the result."""
        if num_to_roll is None:
            num_to_roll = self.remaining_dice
            
        if num_to_roll > self.remaining_dice:
            raise ValueError(f"Cannot roll {num_to_roll} dice, only {self.remaining_dice} remaining")
        
        values = [random.randint(1, 6) for _ in range(num_to_roll)]
        return DiceRoll(values)
    
    def roll_specific(self, values: List[int]) -> DiceRoll:
        """Create a roll with specific values (for testing/input)."""
        if len(values) > self.remaining_dice:
            raise ValueError(f"Cannot roll {len(values)} dice, only {self.remaining_dice} remaining")
        return DiceRoll(values)
    
    def remove_dice(self, count: int):
        """Remove dice from the available pool."""
        if count > self.remaining_dice:
            raise ValueError(f"Cannot remove {count} dice, only {self.remaining_dice} remaining")
        self.remaining_dice -= count
    
    def reset(self):
        """Reset to full dice count."""
        self.remaining_dice = self.num_dice
    
    @property
    def is_tutto(self) -> bool:
        """Check if all dice have been successfully used (Tutto!)."""
        return self.remaining_dice == 0
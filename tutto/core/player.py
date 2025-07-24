"""Player management for Tutto."""
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class PlayerType(Enum):
    """Types of players."""
    HUMAN = "human"
    AI_OPTIMAL = "ai_optimal"
    AI_GREEDY = "ai_greedy"
    AI_CONSERVATIVE = "ai_conservative"


@dataclass
class Player:
    """Represents a player in the game."""
    name: str
    player_type: PlayerType = PlayerType.HUMAN
    score: int = 0
    turn_history: List[int] = field(default_factory=list)
    
    def add_score(self, points: int):
        """Add points to player's score."""
        self.score = max(0, self.score + points)  # Score can't go below 0
        self.turn_history.append(points)
    
    def deduct_score(self, points: int):
        """Deduct points from player's score."""
        self.score = max(0, self.score - points)
    
    @property
    def total_turns(self) -> int:
        """Number of turns played."""
        return len(self.turn_history)
    
    @property
    def average_score_per_turn(self) -> float:
        """Average score per turn."""
        if not self.turn_history:
            return 0.0
        return sum(self.turn_history) / len(self.turn_history)
    
    def __str__(self):
        return f"{self.name} ({self.score} points)"
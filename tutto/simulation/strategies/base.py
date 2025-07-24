"""
Base classes for strategy implementations.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from ...core.dice import DiceRoll
from ...core.scoring import ScoringEngine
from ...core.game_state import DecisionState


@dataclass
class StrategyConfig:
    """Configuration for a strategy."""
    name: str
    description: str
    parameters: Dict[str, Any]


class Strategy(ABC):
    """Abstract base class for game strategies."""
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or self.get_default_config()
        self.setup(**self.config.parameters)
    
    @abstractmethod
    def setup(self, **kwargs):
        """Initialize strategy with parameters."""
        pass
    
    @abstractmethod
    def should_continue(self, game_state: DecisionState) -> bool:
        """Decide whether to continue rolling based on game state."""
        pass
    
    @abstractmethod
    def select_dice(self, roll: DiceRoll, scoring_engine: ScoringEngine) -> List[int]:
        """Select which dice to keep from a roll."""
        pass
    
    @classmethod
    @abstractmethod
    def get_default_config(cls) -> StrategyConfig:
        """Get default configuration for this strategy."""
        pass
    
    def get_description(self) -> str:
        """Get human-readable description of the strategy."""
        return f"{self.config.name}: {self.config.description}"
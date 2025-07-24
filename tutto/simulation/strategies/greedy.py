"""
Greedy strategy implementation.
"""
from typing import List
from ...core.dice import DiceRoll
from ...core.scoring import ScoringEngine
from ...core.game_state import DecisionState
from .base import Strategy, StrategyConfig


class GreedyStrategy(Strategy):
    """Always continue rolling while possible."""
    
    def setup(self, **kwargs):
        pass
    
    def should_continue(self, game_state: DecisionState) -> bool:
        return (game_state.remaining_dice > 0 and 
                game_state.round_status == "in_progress")
    
    def select_dice(self, roll: DiceRoll, scoring_engine: ScoringEngine) -> List[int]:
        """Select all scoring dice."""
        score, combinations = scoring_engine.calculate_score(roll)
        selected_dice = []
        for combo in combinations:
            selected_dice.extend(combo.dice_used)
        return selected_dice
    
    @classmethod
    def get_default_config(cls) -> StrategyConfig:
        return StrategyConfig(
            name="Greedy",
            description="Always continue rolling, selecting all scoring dice",
            parameters={}
        )
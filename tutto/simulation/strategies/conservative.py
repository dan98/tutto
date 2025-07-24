"""
Conservative strategy implementation.
"""
from typing import List
from ...core.dice import DiceRoll
from ...core.scoring import ScoringEngine
from ...core.game_state import DecisionState
from .base import Strategy, StrategyConfig


class ConservativeStrategy(Strategy):
    """Stop after reaching a score threshold."""
    
    def setup(self, threshold: int = 300):
        self.threshold = threshold
    
    def should_continue(self, game_state: DecisionState) -> bool:
        if game_state.accumulated_score >= self.threshold:
            return False
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
            name="Conservative",
            description="Stop after reaching score threshold",
            parameters={"threshold": 300}
        )
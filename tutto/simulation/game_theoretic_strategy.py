"""Game-theoretic optimal strategy for multi-player Tutto."""
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from .strategies import Strategy, StrategyConfig
from ..core.dice import DiceRoll
from ..core.scoring import ScoringEngine
from ..core.cards import CardType
from ..core.game_state import DecisionState, GameContext


class GameTheoreticOptimalStrategy(Strategy):
    """
    Optimal strategy that considers game state and opponent positions.
    
    This strategy makes decisions based on:
    1. Current position relative to opponents
    2. Distance to winning score
    3. Risk tolerance based on game state
    4. Expected opponent progress
    """
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        """
        Initialize game-theoretic optimal strategy.
        """
        super().__init__(config)
    
    def setup(self, **kwargs):
        """Initialize strategy with parameters."""
        self.risk_adjustment = kwargs.get("risk_adjustment", 1.0)
        
        # Cache for expected values at different dice counts
        self._ev_cache = {}
        self._initialize_ev_cache()
    
    def _initialize_ev_cache(self):
        """Pre-calculate expected values for different dice counts."""
        # Simplified expected values per die
        # These are approximations based on Tutto scoring rules
        self._ev_cache = {
            1: 25,    # 1/6 * 100 + 1/6 * 50 + bust risk
            2: 75,    # Higher with more dice
            3: 150,   # Triples become possible
            4: 250,   # Better triple chances
            5: 375,   # High probability of scoring
            6: 500,   # Fresh roll with all dice
        }
    
    def should_continue(self, game_state: Union[Dict, DecisionState]) -> bool:
        """
        Decide whether to continue based on game theory considerations.
        """
        # Convert to DecisionState if needed
        if isinstance(game_state, dict):
            decision_state = DecisionState.from_dict(game_state)
        else:
            decision_state = game_state
        
        # Check if we have game context
        if not decision_state.game_context:
            # Fall back to single-player optimal strategy
            return self._single_player_decision(decision_state)
        
        # Use DecisionState properties
        total_if_stop = decision_state.total_score_if_stop
        
        # Get current player's total score from context
        my_total_score = decision_state.game_context.current_player_score
        my_score_if_stop = my_total_score + total_if_stop
        
        # Analyze game position using GameContext
        position_analysis = self._analyze_game_position_from_context(
            decision_state.game_context
        )
        
        # Calculate continuation value with game context
        continuation_ev = self._calculate_contextual_continuation_value(
            decision_state, position_analysis
        )
        
        # Adjust decision based on game state
        threshold = total_if_stop
        
        # If we're behind, be more aggressive
        if position_analysis["position"] == "behind":
            threshold *= (0.8 * self.risk_adjustment)
        # If we're close to winning, be more conservative
        elif position_analysis["distance_to_win"] < 1000:
            threshold *= (1.2 / self.risk_adjustment)
        # If opponents are close to winning, be more aggressive
        elif position_analysis["min_opponent_distance"] < 1000:
            threshold *= (0.7 * self.risk_adjustment)
        
        return continuation_ev > threshold
    
    def _analyze_game_position_from_context(self, context: GameContext) -> Dict:
        """Analyze current game position from context."""
        return {
            "position": context.position,
            "distance_to_win": context.current_player_distance_to_win,
            "lead_margin": context.lead_margin,
            "min_opponent_distance": context.min_opponent_distance_to_win,
            "catch_up_pressure": self._calculate_catch_up_pressure(context)
        }
    
    def _calculate_catch_up_pressure(self, context: GameContext) -> float:
        """Calculate catch-up pressure based on position."""
        if context.position == "behind":
            # Higher pressure if we're far behind
            return min(1.0, abs(context.lead_margin) / 1000)
        return 0
    def _analyze_game_position(
        self, 
        my_score: int, 
        all_scores: Dict[str, int], 
        target: int,
        current_player: str
    ) -> Dict:
        """Legacy method for backward compatibility."""
        # Remove current player from opponent analysis
        opponent_scores = {k: v for k, v in all_scores.items() if k != current_player}
        
        if not opponent_scores:
            # Single player game
            return {
                "position": "leading",
                "distance_to_win": target - my_score,
                "lead_margin": 0,
                "min_opponent_distance": target,
                "catch_up_pressure": 0
            }
        
        max_opponent_score = max(opponent_scores.values())
        min_opponent_distance = min(target - s for s in opponent_scores.values())
        
        # Determine position
        if my_score > max_opponent_score:
            position = "leading"
            lead_margin = my_score - max_opponent_score
        elif my_score == max_opponent_score:
            position = "tied"
            lead_margin = 0
        else:
            position = "behind"
            lead_margin = my_score - max_opponent_score  # Negative
        
        # Calculate catch-up pressure (how urgently we need points)
        catch_up_pressure = 0
        if position == "behind":
            # Higher pressure if we're far behind
            catch_up_pressure = min(1.0, abs(lead_margin) / 1000)
        
        return {
            "position": position,
            "distance_to_win": target - my_score,
            "lead_margin": lead_margin,
            "min_opponent_distance": min_opponent_distance,
            "catch_up_pressure": catch_up_pressure
        }
    
    def _calculate_contextual_continuation_value(
        self, 
        decision_state: DecisionState,
        position_analysis: Dict
    ) -> float:
        """Calculate continuation value considering game context."""
        remaining_dice = decision_state.remaining_dice
        current_score = decision_state.accumulated_score
        card = decision_state.card
        
        # Base expected value
        base_ev = self._ev_cache.get(remaining_dice, 0)
        
        # Adjust for current score (momentum)
        if current_score > 0:
            momentum_bonus = min(0.2, current_score / 1000)
            base_ev *= (1 + momentum_bonus)
        
        # Adjust for card type
        if card:
            if card.card_type == CardType.BONUS:
                # Bonus cards incentivize going for tutto
                if decision_state.achieved_tutto:
                    base_ev *= 1.5  # Already achieved tutto, safe to continue
                else:
                    base_ev *= 1.2  # Incentive to achieve tutto
            elif card.card_type == CardType.X2:
                # x2 cards heavily incentivize tutto
                if not decision_state.achieved_tutto:
                    base_ev *= 1.5
            elif card.card_type == CardType.PLUS_MINUS:
                # Must achieve tutto, no choice
                base_ev = float('inf')
            elif card.card_type == CardType.CLOVERLEAF:
                # Must achieve 2 tuttos
                if decision_state.tutto_count < 2:
                    base_ev = float('inf')
        
        # Adjust for game position
        if position_analysis["position"] == "behind":
            # Need to take more risks when behind
            base_ev *= (1 + position_analysis["catch_up_pressure"] * 0.5)
        elif position_analysis["distance_to_win"] < 500:
            # Very close to winning, be conservative
            base_ev *= 0.5
        elif position_analysis["min_opponent_distance"] < 500:
            # Opponent close to winning, need to be aggressive
            base_ev *= 1.5
        
        # Add current accumulated score
        total_ev = current_score + base_ev
        
        return total_ev
    
    def _single_player_decision(self, decision_state: DecisionState) -> bool:
        """Fallback to single-player optimal decision."""
        current_score = decision_state.accumulated_score
        total_if_stop = decision_state.total_score_if_stop
        remaining_dice = decision_state.remaining_dice
        
        # Simple expected value calculation
        expected_gain = self._ev_cache.get(remaining_dice, 0)
        continuation_value = current_score + expected_gain
        
        return continuation_value > total_if_stop
    
    def select_dice(self, roll: DiceRoll, scoring_engine: ScoringEngine) -> List[int]:
        """
        Select dice to keep based on maximizing expected value.
        
        In game-theoretic play, we might be more or less aggressive
        based on game state, but dice selection follows the same
        principles as single-player optimal.
        """
        # For now, use the same selection logic as OptimalStrategy
        # This could be enhanced to consider game state
        return self._select_optimal_dice(roll, scoring_engine)
    
    def _select_optimal_dice(self, roll: DiceRoll, scoring_engine: ScoringEngine) -> List[int]:
        """Select dice that maximize expected value."""
        score, combinations = scoring_engine.calculate_score(roll)
        
        if score == 0:
            return []
        
        # Find all possible selections
        possible_selections = []
        
        # Try each combination
        for combo in combinations:
            selection = []
            remaining_dice = roll.values.copy()
            
            # Extract dice for this combination
            for die in combo.dice_used:
                if die in remaining_dice:
                    selection.append(die)
                    remaining_dice.remove(die)
            
            dice_kept = len(selection)
            dice_remaining = len(roll.values) - dice_kept
            
            # Calculate expected value
            future_ev = self._ev_cache.get(dice_remaining, 0) if dice_remaining > 0 else 0
            total_ev = combo.points + future_ev
            
            possible_selections.append({
                "dice": selection,
                "immediate_score": combo.points,
                "expected_total": total_ev,
                "dice_remaining": dice_remaining
            })
        
        # Also consider keeping all scoring dice
        all_scoring = []
        for val in roll.values:
            if val == 1 or val == 5:
                all_scoring.append(val)
        
        if all_scoring:
            imm_score = len([d for d in all_scoring if d == 1]) * 100 + \
                       len([d for d in all_scoring if d == 5]) * 50
            dice_remaining = len(roll.values) - len(all_scoring)
            future_ev = self._ev_cache.get(dice_remaining, 0) if dice_remaining > 0 else 0
            
            possible_selections.append({
                "dice": all_scoring,
                "immediate_score": imm_score,
                "expected_total": imm_score + future_ev,
                "dice_remaining": dice_remaining
            })
        
        # Select the option with highest expected value
        if possible_selections:
            best = max(possible_selections, key=lambda x: x["expected_total"])
            return best["dice"]
        
        return []

    @classmethod
    def get_default_config(cls) -> StrategyConfig:
        """Get default configuration for this strategy."""
        return StrategyConfig(
            name="GameTheoretic",
            description="Optimal strategy considering opponent positions and game state",
            parameters={"risk_adjustment": 1.0}
        )


class AdaptiveStrategy(Strategy):
    """
    Adaptive strategy that changes behavior based on game phase and position.
    
    Early game: Build steady points (conservative)
    Mid game: Balanced approach
    Late game: Aggressive if behind, conservative if leading
    """
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config)
    
    def setup(self, **kwargs):
        """Initialize strategy with parameters."""
        self.early_game_threshold = kwargs.get("early_game_threshold", 2000)
        self.late_game_threshold = kwargs.get("late_game_threshold", 4500)
    
    def should_continue(self, game_state: Union[Dict, DecisionState]) -> bool:
        """Adapt strategy based on game phase."""
        # Convert to DecisionState if needed
        if isinstance(game_state, dict):
            decision_state = DecisionState.from_dict(game_state)
        else:
            decision_state = game_state
        
        if not decision_state.game_context:
            # Default to conservative without context
            return decision_state.accumulated_score < 300
        
        # Use GameContext properties
        context = decision_state.game_context
        my_total = context.current_player_score
        max_score = max(context.player_scores.values()) if context.player_scores else 0
        
        # Determine phase
        if max_score < self.early_game_threshold:
            # Early game - be conservative
            threshold = 300
        elif max_score < self.late_game_threshold:
            # Mid game - balanced
            threshold = 400
        else:
            # Late game - depends on position
            if my_total >= max_score:
                # Leading - be conservative
                threshold = 500
            else:
                # Behind - be aggressive
                threshold = 200
        
        # Special card handling
        if decision_state.card:
            if decision_state.card_type in [CardType.PLUS_MINUS, CardType.CLOVERLEAF]:
                # Must continue for these cards
                return True
            elif decision_state.card_type == CardType.X2 and not decision_state.achieved_tutto:
                # Strong incentive to get tutto with x2
                threshold *= 0.5
        
        return decision_state.accumulated_score < threshold
    
    def select_dice(self, roll: DiceRoll, scoring_engine: ScoringEngine) -> List[int]:
        """Select dice based on current strategy phase."""
        # For now, always select all scoring dice
        # Could be enhanced to be more selective based on game phase
        score, combinations = scoring_engine.calculate_score(roll)
        
        if score == 0:
            return []
        
        # Find the highest scoring combination
        if combinations:
            best_combo = max(combinations, key=lambda c: c.points)
            return best_combo.dice_used
        
        return []
    
    @classmethod
    def get_default_config(cls) -> StrategyConfig:
        """Get default configuration for this strategy."""
        return StrategyConfig(
            name="Adaptive",
            description="Adapts strategy based on game phase and position",
            parameters={
                "early_game_threshold": 2000,
                "late_game_threshold": 4500
            }
        )
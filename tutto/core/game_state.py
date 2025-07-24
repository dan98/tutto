"""Game state classes for Tutto."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
from .cards import Card, CardType
from .dice import DiceRoll
from .player import Player


@dataclass
class GameContext:
    """Context information about the overall game state."""
    player_scores: Dict[str, int]
    current_player: str
    target_score: int
    leading_players: List[str]
    deck_remaining: int
    rounds_played: int
    
    @property
    def max_opponent_score(self) -> int:
        """Get the highest opponent score."""
        opponent_scores = [score for name, score in self.player_scores.items() 
                          if name != self.current_player]
        return max(opponent_scores) if opponent_scores else 0
    
    @property
    def min_opponent_distance_to_win(self) -> int:
        """Get the minimum distance any opponent has to winning."""
        opponent_scores = [score for name, score in self.player_scores.items() 
                          if name != self.current_player]
        if not opponent_scores:
            return self.target_score
        return min(self.target_score - score for score in opponent_scores)
    
    @property
    def current_player_score(self) -> int:
        """Get current player's score."""
        return self.player_scores.get(self.current_player, 0)
    
    @property
    def current_player_distance_to_win(self) -> int:
        """Get current player's distance to winning."""
        return self.target_score - self.current_player_score
    
    @property
    def is_current_player_leading(self) -> bool:
        """Check if current player is leading."""
        return self.current_player in self.leading_players
    
    @property
    def position(self) -> str:
        """Get current player's position: 'leading', 'tied', or 'behind'."""
        max_score = self.max_opponent_score
        current_score = self.current_player_score
        
        if current_score > max_score:
            return "leading"
        elif current_score == max_score:
            return "tied"
        else:
            return "behind"
    
    @property
    def lead_margin(self) -> int:
        """Get lead margin (positive if leading, negative if behind)."""
        return self.current_player_score - self.max_opponent_score


@dataclass
class DecisionState:
    """Complete state information for decision making."""
    # Round state
    accumulated_score: int
    total_score_if_stop: int
    remaining_dice: int
    achieved_tutto: bool
    card: Card
    round_status: str
    
    # Roll information
    current_roll: Optional[DiceRoll] = None
    roll_history: List[DiceRoll] = field(default_factory=list)
    
    # Special card state
    straight_collected: Set[int] = field(default_factory=set)
    straight_complete: bool = False
    tutto_count: int = 0
    forced_continue: bool = False
    
    # Player and game context
    player: Optional[Player] = None
    game_context: Optional[GameContext] = None
    
    @property
    def card_type(self) -> CardType:
        """Get card type."""
        return self.card.card_type
    
    @property
    def bonus_value(self) -> int:
        """Get bonus value (for backward compatibility)."""
        if self.card.card_type == CardType.BONUS:
            return self.card.value
        return 0
    
    @property
    def must_continue(self) -> bool:
        """Check if player must continue (no choice)."""
        return self.card_type in [CardType.PLUS_MINUS, CardType.CLOVERLEAF, CardType.FIREWORKS]
    
    @property
    def is_special_card(self) -> bool:
        """Check if this is a special (non-bonus) card."""
        return self.card_type != CardType.BONUS
    
    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility."""
        result = {
            "accumulated_score": self.accumulated_score,
            "total_score_if_stop": self.total_score_if_stop,
            "remaining_dice": self.remaining_dice,
            "achieved_tutto": self.achieved_tutto,
            "bonus_card": self.bonus_value,  # Backward compatibility
            "card": self.card,
            "card_type": self.card_type,
            "round_status": self.round_status,
            "current_roll": self.current_roll,
            "roll_history": self.roll_history,
            "straight_collected": self.straight_collected,
            "straight_complete": self.straight_complete,
            "tutto_count": self.tutto_count,
            "player": self.player,
        }
        
        if self.game_context:
            result["game_context"] = {
                "player_scores": self.game_context.player_scores,
                "current_player": self.game_context.current_player,
                "target_score": self.game_context.target_score,
                "leading_players": self.game_context.leading_players,
            }
        
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> "DecisionState":
        """Create from dictionary."""
        # Extract game context if present
        game_context = None
        if "game_context" in data:
            gc = data["game_context"]
            game_context = GameContext(
                player_scores=gc.get("player_scores", {}),
                current_player=gc.get("current_player", ""),
                target_score=gc.get("target_score", 6000),
                leading_players=gc.get("leading_players", []),
                deck_remaining=gc.get("deck_remaining", 56),
                rounds_played=gc.get("rounds_played", 0),
            )
        
        # Handle card
        card = data.get("card")
        if not card:
            # Create from legacy format
            from .cards import Card
            card_type = data.get("card_type", CardType.BONUS)
            if card_type == CardType.BONUS:
                card = Card(CardType.BONUS, data.get("bonus_card", 300))
            else:
                card = Card(card_type)
        
        return cls(
            accumulated_score=data["accumulated_score"],
            total_score_if_stop=data["total_score_if_stop"],
            remaining_dice=data["remaining_dice"],
            achieved_tutto=data["achieved_tutto"],
            card=card,
            round_status=data["round_status"],
            current_roll=data.get("current_roll"),
            roll_history=data.get("roll_history", []),
            straight_collected=data.get("straight_collected", set()),
            straight_complete=data.get("straight_complete", False),
            tutto_count=data.get("tutto_count", 0),
            forced_continue=data.get("forced_continue", False),
            player=data.get("player"),
            game_context=game_context,
        )
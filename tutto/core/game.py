from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Set, Dict
from enum import Enum
from .dice import Dice, DiceRoll
from .scoring import ScoringEngine, ScoringCombination
from .cards import CardType, Card, Deck
from .player import Player
from .game_state import DecisionState, GameContext


class RoundStatus(Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BUSTED = "busted"


@dataclass
class RoundState:
    """Represents the current state of a round."""
    card: Card
    accumulated_score: int = 0
    current_roll: Optional[DiceRoll] = None
    roll_history: List[DiceRoll] = field(default_factory=list)
    selected_dice_history: List[List[int]] = field(default_factory=list)
    status: RoundStatus = RoundStatus.IN_PROGRESS
    achieved_tutto: bool = False
    straight_collected: Set[int] = field(default_factory=set)
    straight_complete: bool = False
    # For special cards
    forced_continue: bool = False  # For Fireworks
    tutto_count: int = 0  # For Cloverleaf (need 2 in a row)
    points_before_card: int = 0  # Points accumulated before revealing current card

    @property
    def total_score(self) -> int:
        """Total score including card effects."""
        if self.card.card_type == CardType.STOP:
            return 0
        elif self.card.card_type == CardType.BONUS and self.achieved_tutto:
            return self.accumulated_score + self.card.value
        elif self.card.card_type == CardType.STRAIGHT and self.straight_complete:
            return 2000
        elif self.card.card_type == CardType.X2 and self.achieved_tutto:
            return self.accumulated_score * 2
        elif self.card.card_type == CardType.PLUS_MINUS and self.achieved_tutto:
            return 1000  # Plus/Minus always gives exactly 1000
        elif self.card.card_type == CardType.FIREWORKS:
            return self.accumulated_score  # All accumulated points count
        elif self.card.card_type == CardType.CLOVERLEAF and self.tutto_count >= 2:
            return float('inf')  # Special marker for instant win
        return self.accumulated_score


class Round:
    """Manages a single round of Tutto."""

    def __init__(self, card: Card, player: Player):
        self.dice = Dice()
        self.scoring_engine = ScoringEngine()
        self.state = RoundState(card=card)
        self.player = player

    def roll_dice(self, specific_values: Optional[List[int]] = None) -> Tuple[DiceRoll, int, List[ScoringCombination]]:
        """Roll remaining dice and calculate possible scores."""
        if self.state.status != RoundStatus.IN_PROGRESS:
            raise ValueError("Cannot roll dice - round is not in progress")

        if specific_values:
            roll = self.dice.roll_specific(specific_values)
        else:
            roll = self.dice.roll()

        self.state.current_roll = roll
        self.state.roll_history.append(roll)

        if self.state.card.card_type == CardType.STRAIGHT:
            # For Straight, check for new unique values
            new_values = set(roll.values) - self.state.straight_collected

            if not new_values:
                # No new values - bust!
                self.state.status = RoundStatus.BUSTED
                return roll, 0, []

            # Return pseudo-score for display (not added to accumulated)
            score = len(new_values) * 100  # Just for display
            combinations = [ScoringCombination(
                dice_used=list(new_values),
                points=score,
                description=f"New values: {sorted(new_values)}"
            )]
            return roll, score, combinations
        else:
            # Regular scoring
            score, combinations = self.scoring_engine.calculate_score(roll)

            # Check if busted (no scoring dice)
            if score == 0:
                self.state.status = RoundStatus.BUSTED
                # For Fireworks, busting ends the turn but keeps accumulated score
                if self.state.card.card_type != CardType.FIREWORKS:
                    self.state.accumulated_score = 0

            return roll, score, combinations

    def select_dice(self, selected_values: List[int]) -> int:
        """Select dice to keep and calculate score."""
        if self.state.status != RoundStatus.IN_PROGRESS:
            raise ValueError("Cannot select dice - round is not in progress")

        if not self.state.current_roll:
            raise ValueError("No current roll to select from")

        if self.state.card.card_type == CardType.STRAIGHT:
            # For Straight, only keep one die per new unique value
            new_values = set(selected_values) - self.state.straight_collected
            if not new_values:
                raise ValueError("Must select at least one new value for Straight")

            # Add new values to collected
            self.state.straight_collected.update(new_values)

            # Remove one die per new value
            self.dice.remove_dice(len(new_values))
            self.state.selected_dice_history.append(list(new_values))

            # Check if completed Straight
            if len(self.state.straight_collected) == 6:
                self.state.straight_complete = True
                self.state.achieved_tutto = True  # Straight counts as Tutto
                if self.state.card.card_type != CardType.CLOVERLEAF:
                    self.state.status = RoundStatus.COMPLETED

            return len(new_values) * 100  # Display score
        else:
            # Regular selection
            selected_roll = DiceRoll(selected_values)
            if not self.scoring_engine.is_valid_selection(self.state.current_roll, selected_values):
                raise ValueError("Invalid selection - must include at least one scoring die")

            # Calculate score for selected dice
            score, _ = self.scoring_engine.calculate_score(selected_roll)
            self.state.accumulated_score += score

            # Update dice state
            self.dice.remove_dice(len(selected_values))
            self.state.selected_dice_history.append(selected_values)

            # Check if achieved Tutto
            if self.dice.is_tutto:
                self.state.achieved_tutto = True
                self.state.tutto_count += 1
                self.dice.reset()  # Reset for potential continuation

                # Check for Cloverleaf win condition
                if self.state.card.card_type == CardType.CLOVERLEAF and self.state.tutto_count >= 2:
                    self.state.status = RoundStatus.COMPLETED

            return score

    def stop_round(self):
        """Stop the round and keep accumulated score."""
        if self.state.status != RoundStatus.IN_PROGRESS:
            raise ValueError("Round is not in progress")

        self.state.status = RoundStatus.COMPLETED

    def get_decision_info(self) -> dict:
        """Get information needed for decision making."""
        # Create DecisionState for cleaner interface
        decision_state = DecisionState(
            accumulated_score=self.state.accumulated_score,
            total_score_if_stop=self.state.total_score,
            remaining_dice=self.dice.remaining_dice,
            achieved_tutto=self.state.achieved_tutto,
            card=self.state.card,
            round_status=self.state.status.value,
            current_roll=self.state.current_roll,
            roll_history=self.state.roll_history,
            straight_collected=self.state.straight_collected,
            straight_complete=self.state.straight_complete,
            tutto_count=self.state.tutto_count,
            forced_continue=self.state.forced_continue,
            player=self.player,
        )
        
        # Return as dict for backward compatibility
        return decision_state.to_dict()
    
    def get_decision_state(self, game: Optional['Game'] = None) -> DecisionState:
        """Get decision state object with optional game context."""
        decision_state = DecisionState(
            accumulated_score=self.state.accumulated_score,
            total_score_if_stop=self.state.total_score,
            remaining_dice=self.dice.remaining_dice,
            achieved_tutto=self.state.achieved_tutto,
            card=self.state.card,
            round_status=self.state.status.value,
            current_roll=self.state.current_roll,
            roll_history=self.state.roll_history,
            straight_collected=self.state.straight_collected,
            straight_complete=self.state.straight_complete,
            tutto_count=self.state.tutto_count,
            forced_continue=self.state.forced_continue,
            player=self.player,
        )
        
        # Add game context if available
        if game:
            decision_state.game_context = GameContext(
                player_scores={p.name: p.score for p in game.players},
                current_player=self.player.name,
                target_score=game.target_score,
                leading_players=[p.name for p in game.leading_players],
                deck_remaining=game.deck.cards_remaining,
                rounds_played=game.rounds_played,
            )
        
        return decision_state


class Game:
    """Manages a full game of Tutto."""

    def __init__(self, players: List[Player], target_score: int = 6000):
        self.players = players
        self.target_score = target_score
        self.deck = Deck()
        self.current_player_index = 0
        self.rounds_played = 0
        self.game_over = False
        self.winner: Optional[Player] = None
        self.round_history: List[Dict] = []
        self.current_round: Optional[Round] = None

    @property
    def current_player(self) -> Player:
        """Get the current player."""
        return self.players[self.current_player_index]

    @property
    def leading_players(self) -> List[Player]:
        """Get the player(s) with the highest score."""
        if not self.players:
            return []
        max_score = max(p.score for p in self.players)
        return [p for p in self.players if p.score == max_score]

    def start_new_round(self) -> Tuple[Round, Card]:
        """Start a new round for the current player."""
        card = self.deck.draw()

        # Handle Stop card immediately
        if card.card_type == CardType.STOP:
            self.current_player.add_score(0)
            self.advance_turn()
            return None, card

        self.current_round = Round(card, self.current_player)
        return self.current_round, card

    def complete_round(self):
        """Complete the current round and update scores."""
        if not self.current_round:
            return

        round_obj = self.current_round
        player = round_obj.player

        if round_obj.state.status == RoundStatus.IN_PROGRESS:
            round_obj.stop_round()

        # Calculate final score
        final_score = round_obj.state.total_score

        # Handle special card effects
        if round_obj.state.card.card_type == CardType.CLOVERLEAF and final_score == float('inf'):
            # Instant win!
            self.game_over = True
            self.winner = player
            player.add_score(self.target_score - player.score)  # Set to target score
        elif round_obj.state.card.card_type == CardType.PLUS_MINUS and round_obj.state.achieved_tutto:
            # Player gets 1000 points
            player.add_score(1000)
            # Leading players lose 1000 points
            for leader in self.leading_players:
                if leader != player:  # Don't deduct from self if player is leader
                    leader.deduct_score(1000)
        else:
            # Normal scoring
            player.add_score(final_score)

        # Record round history
        self.round_history.append({
            "player": player.name,
            "card": str(round_obj.state.card),
            "score": final_score,
            "achieved_tutto": round_obj.state.achieved_tutto,
        })

        # Check for game end
        if player.score >= self.target_score and not self.game_over:
            # Player reached target - everyone else gets one more turn
            self.game_over = True

        self.deck.discard(round_obj.state.card)
        self.current_round = None
        self.advance_turn()

    def advance_turn(self):
        """Move to the next player."""
        self.current_player_index = (self.current_player_index + 1) % len(self.players)

        # Check if we've completed a full round of turns
        if self.current_player_index == 0:
            self.rounds_played += 1

            # If game_over flag is set and everyone has had equal turns, end game
            if self.game_over and not self.winner:
                # Determine winner (highest score)
                self.winner = max(self.players, key=lambda p: p.score)

    def get_game_state(self) -> dict:
        """Get current game state."""
        return {
            "players": [{"name": p.name, "score": p.score} for p in self.players],
            "current_player": self.current_player.name,
            "rounds_played": self.rounds_played,
            "game_over": self.game_over,
            "winner": self.winner.name if self.winner else None,
            "deck_remaining": self.deck.cards_remaining,
            "target_score": self.target_score,
        }

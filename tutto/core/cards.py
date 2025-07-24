"""Card types and deck management for Tutto."""
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
import random


class CardType(Enum):
    """Types of cards in Tutto."""
    BONUS = "bonus"
    STRAIGHT = "straight"  # Renamed from STRASSE for consistency
    STOP = "stop"
    FIREWORKS = "fireworks"
    PLUS_MINUS = "plus_minus"
    X2 = "x2"
    CLOVERLEAF = "cloverleaf"
    
    @property
    def description(self):
        """Get card description."""
        descriptions = {
            CardType.BONUS: "Collect bonus points if you achieve Tutto",
            CardType.STRAIGHT: "Roll 1,2,3,4,5,6 to win 2000 points",
            CardType.STOP: "End your turn immediately",
            CardType.FIREWORKS: "Keep rolling until you bust",
            CardType.PLUS_MINUS: "Score 1000 points and leader loses 1000",
            CardType.X2: "Double your points if you achieve Tutto",
            CardType.CLOVERLEAF: "Win instantly with two Tuttos in a row"
        }
        return descriptions[self]


@dataclass
class Card:
    """Represents a single card in the deck."""
    card_type: CardType
    value: Optional[int] = None  # For bonus cards
    
    def __str__(self):
        if self.card_type == CardType.BONUS:
            return f"Bonus {self.value}"
        return self.card_type.value.replace("_", " ").title()
    
    @property
    def display_name(self):
        """Get display name for the card."""
        if self.card_type == CardType.BONUS:
            return f"{self.value} Bonus"
        names = {
            CardType.STRAIGHT: "Straight",
            CardType.STOP: "Stop",
            CardType.FIREWORKS: "Fireworks",
            CardType.PLUS_MINUS: "Plus/Minus",
            CardType.X2: "x2",
            CardType.CLOVERLEAF: "Cloverleaf"
        }
        return names.get(self.card_type, self.card_type.value)


class Deck:
    """Manages the deck of cards."""
    
    def __init__(self):
        self.cards: List[Card] = []
        self.discard_pile: List[Card] = []
        self._create_deck()
        self.shuffle()
    
    def _create_deck(self):
        """Create a standard Tutto deck with 56 cards."""
        # 25 Bonus cards (5 each of 200, 300, 400, 500, 600)
        for value in [200, 300, 400, 500, 600]:
            for _ in range(5):
                self.cards.append(Card(CardType.BONUS, value))
        
        # 5 Straight cards
        for _ in range(5):
            self.cards.append(Card(CardType.STRAIGHT))
        
        # 10 Stop cards
        for _ in range(10):
            self.cards.append(Card(CardType.STOP))
        
        # 5 Fireworks cards
        for _ in range(5):
            self.cards.append(Card(CardType.FIREWORKS))
        
        # 5 Plus/Minus cards
        for _ in range(5):
            self.cards.append(Card(CardType.PLUS_MINUS))
        
        # 5 x2 cards
        for _ in range(5):
            self.cards.append(Card(CardType.X2))
        
        # 1 Cloverleaf card
        self.cards.append(Card(CardType.CLOVERLEAF))
    
    def shuffle(self):
        """Shuffle the deck."""
        random.shuffle(self.cards)
    
    def draw(self) -> Card:
        """Draw a card from the deck."""
        if not self.cards:
            # Reshuffle discard pile back into deck
            self.cards = self.discard_pile
            self.discard_pile = []
            self.shuffle()
        
        if not self.cards:
            raise ValueError("No cards left in deck or discard pile")
        
        return self.cards.pop()
    
    def discard(self, card: Card):
        """Discard a card."""
        self.discard_pile.append(card)
    
    @property
    def cards_remaining(self) -> int:
        """Number of cards remaining in deck."""
        return len(self.cards)
    
    @property
    def total_cards(self) -> int:
        """Total cards in deck and discard pile."""
        return len(self.cards) + len(self.discard_pile)
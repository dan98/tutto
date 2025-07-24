#!/usr/bin/env python3
"""
Run strategy comparison tests for Tutto game.
"""
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from tests.test_strategy_comparison import TestStrategyComparison
from tutto.simulation import GameSimulator


def main():
    """Run all strategy comparison tests."""
    print("="*80)
    print("TUTTO STRATEGY COMPARISON TESTS")
    print("="*80)
    
    # Create test instance and simulator
    test = TestStrategyComparison()
    simulator = GameSimulator(num_workers=4)
    
    # Test 1: Basic 3-player comparison
    print("\n1. THREE PLAYER STRATEGY COMPARISON")
    print("-"*40)
    test.test_three_player_strategy_comparison(simulator)
    
    # Test 2: All strategies comparison
    print("\n\n2. ALL STRATEGIES COMPARISON")
    print("-"*40)
    test.test_all_strategies_comparison(simulator)
    
    # Test 3: Optimal vs Probability Threshold
    print("\n\n3. OPTIMAL VS PROBABILITY THRESHOLD")
    print("-"*40)
    test.test_optimal_vs_probability_threshold(simulator)
    
    # Test 4: Performance by player count
    print("\n\n4. STRATEGY PERFORMANCE BY PLAYER COUNT")
    print("-"*40)
    test.test_strategy_performance_by_player_count(simulator)
    
    print("\n" + "="*80)
    print("All tests completed!")


if __name__ == "__main__":
    main()
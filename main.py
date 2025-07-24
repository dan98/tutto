#!/usr/bin/env python3
"""
Tutto - A dice game simulator with probability analysis
"""

from tutto.cli.interface import InteractiveCLI


def main():
    """Main entry point for Tutto."""
    cli = InteractiveCLI()
    cli.run()


if __name__ == '__main__':
    main()

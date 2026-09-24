#!/usr/bin/env python3
"""Evaluate GOF: all scenes by default; --scene selects one scene."""
if __package__:
    from .evaluate_method import main
else:
    from evaluate_method import main

if __name__ == '__main__':
    raise SystemExit(main('gof'))

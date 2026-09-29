#!/usr/bin/env python3
"""Evaluate DPR_ordered benchmark runs by default; override with --output-root."""
if __package__:
    from .evaluate_method import main
else:
    from evaluate_method import main

if __name__ == '__main__':
    raise SystemExit(main('ours'))

#!/usr/bin/env python3
"""Evaluate ordered beta-surfel workshop runs against ground-truth meshes."""
if __package__:
    from .evaluate_method import main
else:
    from evaluate_method import main

if __name__ == '__main__':
    raise SystemExit(main('workshop'))

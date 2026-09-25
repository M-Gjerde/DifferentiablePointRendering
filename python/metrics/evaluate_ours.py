#!/usr/bin/env python3
"""Evaluate paper runs using points_final.ply and mesh/fuse_post.ply."""
if __package__:
    from .evaluate_method import main
else:
    from evaluate_method import main

if __name__ == '__main__':
    raise SystemExit(main('ours'))

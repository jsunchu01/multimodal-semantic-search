"""Components package: one subpackage per pipeline stage, each behind a shared
ABC in its own base.py and registered with mmss.registry so backends are
swappable via config alone.
"""

"""Private learner mechanics reused by particular methods.

Shared does not mean universal: d4pg supports QR-D4PG and EX-D4PG, while
pathwise supports Deep Hedging and no-transaction bands. These are internal
building blocks, not additional baselines or a required learner framework.

New algorithms integrate through hedging_gym.interfaces and the environment
APIs. Reusing these helpers is optional when their mechanics fit the method.
"""

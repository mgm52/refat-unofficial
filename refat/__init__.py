"""
ReFAT: Refusal Feature Adversarial Training (Unofficial Implementation)

An unofficial implementation of the ReFAT paper for adversarial training against
jailbreak attempts in language models.
"""

from refat.pipeline.refat import Refat, RefatConfig
from refat.pipeline.refusal_direction import RefusalDirectionCalculator

__version__ = "0.1.0"

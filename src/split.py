"""Stable entity partitions; never split candidate rows independently."""
import hashlib


def hash_id(entity_id, seed=2026):
    return int.from_bytes(hashlib.blake2b(f'{seed}:{entity_id}'.encode(), digest_size=8).digest(), 'big') % (2**63-1)


def partition(entity_id, seed=2026):
    bucket = hash_id(entity_id, seed) % 100
    return 'holdout' if bucket < 10 else 'dev' if bucket < 20 else 'tune' if bucket < 30 else 'fit'

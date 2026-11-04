# heal-guard/world/snapshots.py
import random
from dataclasses import dataclass

@dataclass
class CloudSnapshot:
    buckets: dict
    iam_roles: dict
    security_groups: dict
    task_type: str # F1, F2, F4
    expected_vulnerable: list[str] # Ground truth

def generate_snapshot(seed: int, family: str) -> CloudSnapshot:
    """Generates a deterministic fake cloud account.
    F1: Public buckets
    F2: Wildcard IAM roles
    F4: Open security groups
    """
    # TODO: Implement seeded logic on Wednesday
    pass
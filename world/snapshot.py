"""Seeded fake cloud account. World v2.

What changed from v1 and why
----------------------------
v1 drew one coin (`is_clean`) and then made family index 0 of every family
vulnerable when that coin said "vulnerable". Three consequences, all bad:

* a vulnerable task always had **exactly three** findings, one per family, so
  the answer set was either `[]` or `[bucket-N-0, role-N-0, sg-N-0]`;
* the three families were perfectly correlated, so the per-family regression
  table the gate is built on measured one variable three times;
* the vulnerable resource was always index 0, so "check the first resource" was
  a free shortcut.

v2 draws each family independently, picks vulnerable positions uniformly, and
varies how many resources are vulnerable per family. Answer sets now range from
0 to 8 findings, and a task can be vulnerable in one family only.

The subset we model, unchanged and still ten lines:

1. A bucket is public when its ACL is public **or** its policy is public.
2. A public bucket policy with an explicit deny makes the bucket safe.
3. A public-access block makes the bucket safe, overriding ACL and policy.
4. No conditions, no `NotAction`, no resource policies on roles.
5. A role is risky when it allows `action: *` on `resource: *`.
6. A security group is risky when ingress is `0.0.0.0/0` on 22, 3389 or 3306.
7. Bucket/role/group counts vary, so nothing is positional.
8. Decoys exist in every family: public-looking resources that are safe for a
   reason the agent must actually read (block, deny, narrow action, private CIDR).
9. Every tool returns a dict, so the model has one shape to learn.
10. Roughly 42% of seeds are `none` naturally; splits are then balanced to
    exactly half by seed selection, which is what the plan asks for.

Determinism: `generate_snapshot(seed)` is a pure function of the seed. Writes
nothing, reads nothing.
"""

from __future__ import annotations

import random

# How many resources of a family are vulnerable, when the family is vulnerable.
# (count, weight). Zero-heavy on purpose: most tasks have one or two findings.
_COUNT_WEIGHTS = ((0, 75), (1, 15), (2, 7), (3, 3))

RISKY_PORTS = (22, 3389, 3306)
SAFE_PORTS = (80, 443, 8080)


def _draw_count(rng: random.Random, available: int) -> int:
    """How many resources in this family are vulnerable. Clamped to available."""
    roll = rng.randrange(100)
    total = 0
    for count, weight in _COUNT_WEIGHTS:
        total += weight
        if roll < total:
            return min(count, available)
    return 0


def _draw_buckets(rng: random.Random, seed: int) -> list[dict]:
    size = rng.choice((2, 3, 4))
    vulnerable = set(rng.sample(range(size), _draw_count(rng, size)))

    buckets = []
    for index in range(size):
        bucket = {
            "id": f"bucket-{seed}-{index}",
            "acl_public": False,
            "policy_public": False,
            "policy_deny": False,
            "public_access_block": False,
        }

        if index in vulnerable:
            # Public through the ACL, through the policy, or both. A policy-only
            # bucket cannot be found by reading the ACL, which is the point.
            mode = rng.choices(("acl", "policy", "both"), weights=(5, 4, 3))[0]
            bucket["acl_public"] = mode in ("acl", "both")
            bucket["policy_public"] = mode in ("policy", "both")
        else:
            decoy = rng.randrange(100)
            if decoy < 30:
                # Looks public, closed by a public-access block.
                bucket["acl_public"] = True
                bucket["public_access_block"] = True
            elif decoy < 55:
                # Looks public, closed by an explicit deny.
                bucket["policy_public"] = True
                bucket["policy_deny"] = True
            elif decoy < 70:
                bucket["policy_deny"] = True
            # else: plainly clean

        buckets.append(bucket)

    return buckets


def _draw_roles(rng: random.Random, seed: int) -> list[dict]:
    size = rng.choice((2, 3, 3))
    vulnerable = set(rng.sample(range(size), _draw_count(rng, size)))

    roles = []
    for index in range(size):
        policies = []
        if index in vulnerable:
            policies.append({"action": "*", "resource": "*", "effect": "Allow"})
        else:
            decoy = rng.randrange(100)
            if decoy < 35:
                # Wildcard resource, narrow action.
                policies.append(
                    {"action": "s3:GetObject", "resource": "*", "effect": "Allow"}
                )
            elif decoy < 70:
                # Wildcard action, narrow resource.
                policies.append(
                    {"action": "*", "resource": "arn:aws:s3:::safe", "effect": "Allow"}
                )
            else:
                policies.append(
                    {
                        "action": "s3:GetObject",
                        "resource": "arn:aws:s3:::safe",
                        "effect": "Allow",
                    }
                )

        # Sometimes a second policy: the agent must read the whole list.
        if rng.randrange(100) < 25:
            policies.append(
                {"action": "logs:DescribeLogGroups", "resource": "*", "effect": "Allow"}
            )

        roles.append({"id": f"role-{seed}-{index}", "policies": policies})

    return roles


def _draw_security_groups(rng: random.Random, seed: int) -> list[dict]:
    size = rng.choice((2, 3, 3))
    vulnerable = set(rng.sample(range(size), _draw_count(rng, size)))

    groups = []
    for index in range(size):
        ingress = []
        if index in vulnerable:
            ingress.append(
                {"port": rng.choice(RISKY_PORTS), "cidr": "0.0.0.0/0"}
            )
        else:
            decoy = rng.randrange(100)
            if decoy < 40:
                # Open to the world, but on a port that is not risky.
                ingress.append({"port": rng.choice(SAFE_PORTS), "cidr": "0.0.0.0/0"})
            elif decoy < 80:
                # Risky port, but private.
                ingress.append(
                    {"port": rng.choice(RISKY_PORTS), "cidr": "10.0.0.0/8"}
                )
            else:
                ingress.append({"port": 443, "cidr": "0.0.0.0/0"})
                ingress.append({"port": 22, "cidr": "10.0.0.0/8"})

        # Sometimes a second rule: the agent must scan every rule, not the first.
        if rng.randrange(100) < 25:
            ingress.append({"port": 443, "cidr": "0.0.0.0/0"})

        groups.append({"id": f"sg-{seed}-{index}", "ingress": ingress})

    return groups


def generate_snapshot(seed: int) -> dict:
    """Deterministic fake cloud account for a seed. Same seed, same bytes."""
    rng = random.Random(seed)

    return {
        "seed": seed,
        "buckets": _draw_buckets(rng, seed),
        "roles": _draw_roles(rng, seed),
        "security_groups": _draw_security_groups(rng, seed),
    }

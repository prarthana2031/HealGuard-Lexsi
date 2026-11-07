# world — the deterministic cloud environment (v2)

Eight tools, one seeded fake account, no AWS, no network.

| File | Role |
|---|---|
| `snapshot.py` | `generate_snapshot(seed)` → buckets, IAM roles, security groups. Pure function of the seed. |
| `tools.py` | `CloudEnv` with the eight tool callables and `TOOL_SCHEMAS`. |
| `specs.py` | Strict argument specs shared by the environment, the classifier and the healer. |
| `harness.py` | Reserved; the rollout harness lives in `eval/run_agent.py`. |

## The subset we implement

Ten lines, because judges ask:

1. A bucket is public when its ACL is public **or** its policy is public.
2. A bucket policy with an explicit deny makes the bucket safe.
3. A public-access block makes the bucket safe, overriding ACL and policy.
4. No conditions, no `NotAction`, no resource policies on roles.
5. A role is risky when it allows `action: *` on `resource: *`.
6. A security group is risky when ingress is `0.0.0.0/0` on 22, 3389 or 3306.
7. Bucket, role and group counts vary (2–4 / 2–3 / 2–3), so nothing is positional.
8. Every family has decoys: public-looking resources that are safe for a reason
   the agent must actually read — a block, a deny, a narrow action, a private CIDR.
9. Unknown tool names return a structured error; they never raise.
10. Every tool returns a dict, so the model has one shape to learn.

## What v1 got wrong, and what v2 does

v1 drew one coin, `is_clean`, then made index 0 of every family vulnerable when
that coin said "vulnerable". Every vulnerable task therefore had exactly three
findings, the three families were perfectly correlated, and the vulnerable
resource was always the first one. The measured consequences on the sealed set
were answer sizes of `{0: 27, 3: 23}` and an `always-none` policy worth the
penalty rate.

v2 draws each family independently and picks positions uniformly. Measured over
4,000 seeds:

| Property | v1 | v2 |
|---|---|---|
| Answer-set sizes | `{0, 3}` only | `{0, 1, 2, 3, 4, 5, 6, 8}` |
| Natural `none` rate | 50% | 41.4% |
| Single-family task | never | 43% of vulnerable tasks |
| All three families | always | 1.7% of tasks |
| Vulnerable index | always 0 | uniform |
| `always-none` task success | 0.60 | 0.50 |

## Answer balance

The plan asks for about half of every split to have the empty answer. v2's
generator is naturally ~41% `none`, so `data/splits/build_splits.py` selects
seeds until a split has **exactly** half `none` and half vulnerable, then
interleaves them. `generate_snapshot(seed)` stays pure; only which seeds a split
claims is chosen. Every split is now 50/50, asserted in
`tests/test_no_leakage.py`.

# world — the deterministic cloud environment

Eight tools, one seeded fake account, no AWS, no network.

| File | Role |
|---|---|
| `snapshot.py` | `generate_snapshot(seed)` → buckets, IAM roles, security groups. Same seed, same bytes. |
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
7. `list_buckets` is paginated on purpose, so stopping at page one is a real
   clean-task failure and not only a drift artefact.
8. Unknown tool names return a structured error; they never raise.
9. Every tool returns a dict, so the model has one shape to learn.
10. About half of all tasks have the answer `none`, by construction.

## Answer balance

The oracle returns the exact set of vulnerable resource ids, or the empty set.
Every split sits between 40% and 60% `none`, asserted in
`tests/test_no_leakage.py`.

**Known weakness, stated rather than hidden:** the current generator flips all
three families together, so a vulnerable task always has exactly three
vulnerable resources, one per family, at index 0. That makes `always-none` worth
roughly the none-rate and lets a model shortcut "check bucket 0". Fixing it means
regenerating every split including the sealed set, which is a freeze decision —
see `AUDIT.md`.

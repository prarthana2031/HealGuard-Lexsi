# Sealed set freeze record

Append-only. Generated from `data/splits/sealed_history.json` by
`python -m data.splits.build_splits`; do not hand-edit.

- **Current hash:** `19abf859754430298beedf84a08c7c1db7bcad93355669537cf869f3c384118a`
- **Task count:** 50 (inside the plan's 40-60 window)
- **Answer balance:** 25 `none`, 25 vulnerable, interleaved
- **Opened before any re-freeze?** No. No score, no table and no
  article number was produced from the world-v1 sealed set.

| Version | sha256 of `sealed_traces.jsonl` |
|---|---|
| world v1 | `fc7dcf2d675620db4aa12077ac9a8cef175b46ab7960965939a6864ace3a46a7` |
| world v2 | `19abf859754430298beedf84a08c7c1db7bcad93355669537cf869f3c384118a` |

The v2 hash is frozen for the 14 Nov submission. Any later change
invalidates the article's claim that the set was opened exactly once.

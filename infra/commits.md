# Commit ledger

Every pin was read back from the checkout, not copied from a README.

| Repo | Commit | Subject |
|---|---|---|
| AgentTune | `d4db4c12d52342227c4113a70183d0f64aa2f345` | Fix model compatibility across chat templates, processors and LoRA targets (#4) |
| AuditKIT | `71316da345784ff220d5ed59337441a59ab31526` | Release 1.1.3 |

Re-verify with:

```powershell
git -C ../AgentTune log -1 --format="%H %s"
git -C ../AuditKIT  log -1 --format="%H %s"
```

If either hash moves, `requirements.txt` and the article's section 2 must move
with it. Two training jobs must never point at two different pins.

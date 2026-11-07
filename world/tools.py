import json

# Every tool's returned text is capped. Our real payloads are far below these,
# but the cap is the guarantee: a snapshot cannot be made so large that it blows
# the 8k window mid-episode.
MAX_LIST_ITEMS = 25
MAX_TEXT_CHARS = 800

# A call after `finish` is ignored and flagged. It is *not* an environment error:
# `env_error` is reserved for injected 5xx/timeout, and mislabelling a post-finish
# call as one used to shove it into the wrong training bucket.
FINISHED_PAYLOAD = {
    "ignored": True,
    "reason": "episode already finished; call ignored",
}


def _cap(payload: dict) -> dict:
    """Bound list length and string length without changing the shape."""
    capped = {}
    truncated = 0
    for key, value in payload.items():
        if isinstance(value, list) and len(value) > MAX_LIST_ITEMS:
            truncated += len(value) - MAX_LIST_ITEMS
            capped[key] = value[:MAX_LIST_ITEMS]
        elif isinstance(value, str) and len(value) > MAX_TEXT_CHARS:
            capped[key] = value[:MAX_TEXT_CHARS] + "...[truncated]"
        else:
            capped[key] = value
    if truncated:
        capped["truncated_items"] = truncated
    return capped


class CloudEnv:
    def __init__(self, snapshot: dict):
        self.snapshot = snapshot
        self.finished = False
        self.ignored_calls = 0

    # -- helpers ---------------------------------------------------------
    def _ignore(self) -> dict:
        self.ignored_calls += 1
        return dict(FINISHED_PAYLOAD)

    def list_buckets(self, limit: int = 10, next_token: str = None) -> dict:
        if self.finished: return self._ignore()
        if not isinstance(limit, int) or isinstance(limit, bool):
            return {"error": "schema_violation: limit must be int"}

        buckets = [b["id"] for b in self.snapshot["buckets"]]
        start_idx = int(next_token) if next_token else 0
        end_idx = min(start_idx + limit, len(buckets))

        return _cap({
            "items": buckets[start_idx:end_idx],
            "next_token": str(end_idx) if end_idx < len(buckets) else None
        })

    def get_bucket_acl(self, bucket: str) -> dict:
        if self.finished: return self._ignore()
        for b in self.snapshot["buckets"]:
            if b["id"] == bucket: return _cap({"acl_public": b["acl_public"]})
        return {"error": "NoSuchBucket"}

    def get_bucket_policy(self, bucket: str) -> dict:
        if self.finished: return self._ignore()
        for b in self.snapshot["buckets"]:
            if b["id"] == bucket:
                return _cap({"policy_public": b["policy_public"], "policy_deny": b["policy_deny"]})
        return {"error": "NoSuchBucket"}

    def get_public_access_block(self, bucket: str) -> dict:
        if self.finished: return self._ignore()
        for b in self.snapshot["buckets"]:
            if b["id"] == bucket: return _cap({"public_access_block": b["public_access_block"]})
        return {"error": "NoSuchBucket"}

    def list_iam_roles(self, limit: int = 10) -> dict:
        if self.finished: return self._ignore()
        return _cap({"roles": [r["id"] for r in self.snapshot["roles"]]})

    def get_role_policies(self, role_name: str) -> dict:
        if self.finished: return self._ignore()
        for r in self.snapshot["roles"]:
            if r["id"] == role_name: return _cap({"policies": r["policies"]})
        return {"error": "NoSuchRole"}

    def describe_security_groups(self, group_ids: list = None) -> dict:
        if self.finished: return self._ignore()
        sgs = self.snapshot["security_groups"]
        if group_ids:
            sgs = [sg for sg in sgs if sg["id"] in group_ids]
        return _cap({"security_groups": sgs})

    def finish(self, answer: list) -> dict:
        if self.finished:
            return self._ignore()
        self.finished = True
        return {"status": "success", "answer": answer}

# Schema definition for Person A/C
TOOL_SCHEMAS = [
    {"name": "list_buckets", "description": "Lists S3 buckets. Args: limit (int), next_token (str)"},
    {"name": "get_bucket_acl", "description": "Gets ACL. Args: bucket (str)"},
    {"name": "get_bucket_policy", "description": "Gets policy. Args: bucket (str)"},
    {"name": "get_public_access_block", "description": "Gets access block. Args: bucket (str)"},
    {"name": "list_iam_roles", "description": "Lists IAM roles. Args: limit (int)"},
    {"name": "get_role_policies", "description": "Gets policies. Args: role_name (str)"},
    {"name": "describe_security_groups", "description": "Gets security groups. Args: group_ids (list)"},
    {"name": "finish", "description": "Submits answer. Args: answer (list of str)"}
]

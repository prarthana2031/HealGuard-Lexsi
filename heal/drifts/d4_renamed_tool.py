"""D4: rename get_bucket_acl to get_acl."""

from world.tools import CloudEnv, TOOL_SCHEMAS


class D4DriftEnv(CloudEnv):
    def __init__(self, env: CloudEnv):
        self._base_env = env
        self.snapshot = env.snapshot

    @property
    def finished(self):
        return self._base_env.finished

    @finished.setter
    def finished(self, value):
        self._base_env.finished = value

    def get_acl(self, bucket):
        """Replacement for get_bucket_acl."""
        return self._base_env.get_bucket_acl(bucket=bucket)

    def get_bucket_acl(self, bucket=None, **kwargs):
        """Old name intentionally returns an unknown-tool error."""
        return {
            "error": (
                "unknown_tool: Tool 'get_bucket_acl' not found. "
                "Did you mean 'get_acl'?"
            )
        }


TOOL_SCHEMAS_D4 = []
for _schema in TOOL_SCHEMAS:
    if _schema["name"] == "get_bucket_acl":
        TOOL_SCHEMAS_D4.append({
            "name": "get_acl",
            "description": _schema["description"].replace(
                "get_bucket_acl", "get_acl"
            ),
        })
    else:
        TOOL_SCHEMAS_D4.append(dict(_schema))


def apply_d4(env: CloudEnv, enabled: bool = True):
    return D4DriftEnv(env) if enabled else env

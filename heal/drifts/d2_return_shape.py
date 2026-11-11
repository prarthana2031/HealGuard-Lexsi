"""LEGACY: D2 return-shape drift (retired).

D2 is now a pagination drift in `heal.drifts.d2_pagination`. This module is not
registered and must not be used as the active D2 behaviour; it is kept only so
the old import path does not disappear from history.
"""

from world.tools import CloudEnv, TOOL_SCHEMAS


class D2DriftEnv(CloudEnv):
    def __init__(self, env: CloudEnv):
        self._base_env = env
        self.snapshot = env.snapshot

    @property
    def finished(self):
        return self._base_env.finished

    @finished.setter
    def finished(self, value):
        self._base_env.finished = value

    def list_buckets(self, limit=10, next_token=None):
        """Return the D2 response contract."""
        result = self._base_env.list_buckets(
            limit=limit,
            next_token=next_token,
        )

        # Preserve structured errors *and* the post-finish ignore payload from
        # the underlying environment; neither has an `items` key.
        if "error" in result or "ignored" in result:
            return result

        return {
            "bucket_list": result["items"],
            "continuation": result["next_token"],
        }


TOOL_SCHEMAS_D2 = []
for _schema in TOOL_SCHEMAS:
    _updated = dict(_schema)
    if _updated["name"] == "list_buckets":
        _updated["description"] = (
            "Lists S3 buckets. Args: limit (int), next_token (str). "
            "Returns: bucket_list (list), continuation (str or null)."
        )
    TOOL_SCHEMAS_D2.append(_updated)


def apply_d2(env: CloudEnv, enabled: bool = True):
    """Enable D2 or return the original environment unchanged."""
    return D2DriftEnv(env) if enabled else env

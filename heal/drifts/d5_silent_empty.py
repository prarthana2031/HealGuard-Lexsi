"""D5: fail-closed bucket tools.

The stale client still sends ``bucket``. Instead of a structured error, the API
returns an empty object: a syntactically valid response with no fields. The
model cannot distinguish "this resource has no findings" from "the call was
silently rejected", so it misses the bucket without ever seeing an error.

A call with ``bucket_name`` reaches the real environment.

This is deliberately *not* a Python exception. An exception is a harness bug:
the rollout either crashes or is converted into a schema_violation, and neither
is a deployable API behaviour. A silent empty payload is the failure mode a
pinned client actually ships against.
"""

from world.tools import CloudEnv, TOOL_SCHEMAS


class D5SilentEmptyEnv(CloudEnv):
    """Bucket tools return `{}` for the stale argument and real data for the new one."""

    def __init__(self, env: CloudEnv):
        self._base_env = env

    # -- shared state ----------------------------------------------------
    @property
    def snapshot(self):
        return self._base_env.snapshot

    @property
    def finished(self):
        return self._base_env.finished

    @finished.setter
    def finished(self, value):
        self._base_env.finished = value

    @property
    def ignored_calls(self):
        return self._base_env.ignored_calls

    @ignored_calls.setter
    def ignored_calls(self, value):
        self._base_env.ignored_calls = value

    def _ignore(self):
        return self._base_env._ignore()

    # -- the drift -------------------------------------------------------
    def get_bucket_acl(self, bucket_name=None, **kwargs):
        if self.finished:
            return self._ignore()
        if "bucket" in kwargs:
            return {}
        if bucket_name is None:
            return {}
        return self._base_env.get_bucket_acl(bucket=bucket_name)

    def get_bucket_policy(self, bucket_name=None, **kwargs):
        if self.finished:
            return self._ignore()
        if "bucket" in kwargs:
            return {}
        if bucket_name is None:
            return {}
        return self._base_env.get_bucket_policy(bucket=bucket_name)

    def get_public_access_block(self, bucket_name=None, **kwargs):
        if self.finished:
            return self._ignore()
        if "bucket" in kwargs:
            return {}
        if bucket_name is None:
            return {}
        return self._base_env.get_public_access_block(bucket=bucket_name)


# The stale client sees the clean contract (`bucket`). `bucket_name` is an
# undocumented alias that returns real data; the expert knows it, the model must
# discover it from the training pairs.
TOOL_SCHEMAS_D5 = [dict(schema) for schema in TOOL_SCHEMAS]


def apply_d5(env: CloudEnv, enabled: bool = True):
    """Enable D5 or return the original environment unchanged."""
    return D5SilentEmptyEnv(env) if enabled else env

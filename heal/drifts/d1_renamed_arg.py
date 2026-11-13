"""D1: rename bucket arguments to bucket_name without changing world facts."""

from world.tools import CloudEnv, TOOL_SCHEMAS


class D1DriftEnv(CloudEnv):
    """Expose the D1 API while delegating state to the original environment."""

    def __init__(self, env: CloudEnv):
        self._base_env = env
        self.snapshot = env.snapshot

    @property
    def finished(self):
        return self._base_env.finished

    @finished.setter
    def finished(self, value):
        self._base_env.finished = value

    @staticmethod
    def _validate_bucket_name(bucket_name, extra_args):
        if extra_args:
            # D1-Harsh: do not name the replacement. The caller must already
            # know the rename; the error is deliberately uninformative.
            return {"error": "schema_violation: invalid arguments"}
        if not isinstance(bucket_name, str) or not bucket_name:
            return {"error": (
                "schema_violation: bucket_name must be a non-empty string"
            )}
        return None

    def get_bucket_acl(self, bucket_name=None, **kwargs):
        error = self._validate_bucket_name(bucket_name, kwargs)
        if error:
            return error
        return self._base_env.get_bucket_acl(bucket=bucket_name)

    def get_bucket_policy(self, bucket_name=None, **kwargs):
        error = self._validate_bucket_name(bucket_name, kwargs)
        if error:
            return error
        return self._base_env.get_bucket_policy(bucket=bucket_name)

    def get_public_access_block(self, bucket_name=None, **kwargs):
        error = self._validate_bucket_name(bucket_name, kwargs)
        if error:
            return error
        return self._base_env.get_public_access_block(bucket=bucket_name)


TOOL_SCHEMAS_D1 = []
for _schema in TOOL_SCHEMAS:
    _updated = dict(_schema)
    if _updated["name"] in {
        "get_bucket_acl",
        "get_bucket_policy",
        "get_public_access_block",
    }:
        _updated["description"] = _updated["description"].replace(
            "bucket (str)", "bucket_name (str)"
        )
    TOOL_SCHEMAS_D1.append(_updated)


def apply_d1(env: CloudEnv, enabled: bool = True):
    """Enable D1 or return the original environment unchanged."""
    return D1DriftEnv(env) if enabled else env

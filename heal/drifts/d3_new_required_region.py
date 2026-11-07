"""D3: require region on list_buckets and describe_security_groups."""

from world.tools import CloudEnv, TOOL_SCHEMAS


class D3DriftEnv(CloudEnv):
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
    def _region_error(region):
        if not isinstance(region, str) or not region.strip():
            return {
                "error": "schema_violation: region is required and must be a non-empty string"
            }
        return None

    def list_buckets(self, region=None, limit=10, next_token=None):
        error = self._region_error(region)
        if error:
            return error
        return self._base_env.list_buckets(limit=limit, next_token=next_token)

    def describe_security_groups(self, region=None, group_ids=None):
        error = self._region_error(region)
        if error:
            return error
        return self._base_env.describe_security_groups(group_ids=group_ids)


TOOL_SCHEMAS_D3 = []
for _schema in TOOL_SCHEMAS:
    _updated = dict(_schema)
    if _updated["name"] == "list_buckets":
        _updated["description"] += ", region (str, required)"
    elif _updated["name"] == "describe_security_groups":
        _updated["description"] += ", region (str, required)"
    TOOL_SCHEMAS_D3.append(_updated)


def apply_d3(env: CloudEnv, enabled: bool = True):
    return D3DriftEnv(env) if enabled else env

"""Central selection of clean and drifted CloudEnv interfaces.

D2 is a pagination drift: the response envelope stays clean
(``items`` / ``next_token``) and the page size is capped at one. The old
return-shape drift is no longer registered.
"""

from world.tools import TOOL_SCHEMAS

from heal.drifts.d1_renamed_arg import apply_d1, TOOL_SCHEMAS_D1
from heal.drifts.d2_pagination import apply_d2, TOOL_SCHEMAS_D2
from heal.drifts.d3_new_required_region import apply_d3, TOOL_SCHEMAS_D3
from heal.drifts.d4_renamed_tool import apply_d4, TOOL_SCHEMAS_D4


DRIFT_REGISTRY = {
    "D1": (apply_d1, TOOL_SCHEMAS_D1),
    "D2": (apply_d2, TOOL_SCHEMAS_D2),
    "D3": (apply_d3, TOOL_SCHEMAS_D3),
    "D4": (apply_d4, TOOL_SCHEMAS_D4),
}


def get_tool_schemas(drift=None):
    """Return clean or drift-specific tool descriptions."""
    if drift is None or str(drift).lower() == "clean":
        schemas = TOOL_SCHEMAS
    else:
        name = str(drift).upper()
        if name not in DRIFT_REGISTRY:
            raise ValueError(f"Unknown drift: {drift}")
        schemas = DRIFT_REGISTRY[name][1]

    return [dict(schema) for schema in schemas]


def apply_drift(env, drift=None, enabled=True):
    """Apply exactly one drift, or return the clean environment."""
    if not enabled or drift is None or str(drift).lower() == "clean":
        return env

    name = str(drift).upper()
    if name not in DRIFT_REGISTRY:
        raise ValueError(f"Unknown drift: {drift}")

    apply_function, _ = DRIFT_REGISTRY[name]
    return apply_function(env)


def build_runtime(env, drift=None, enabled=True):
    """Return the selected environment and matching tool schemas."""
    if not enabled:
        return env, get_tool_schemas()

    return apply_drift(env, drift), get_tool_schemas(drift)

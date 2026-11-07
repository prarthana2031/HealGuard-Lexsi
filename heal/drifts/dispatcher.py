"""Execute tool calls against the selected clean or drifted environment."""

from heal.drifts.registry import build_runtime


class DriftToolDispatcher:
    def __init__(self, env, drift=None, enabled=True):
        self.env, self.schemas = build_runtime(
            env, drift=drift, enabled=enabled
        )
        self.drift = str(drift).upper() if drift else "CLEAN"

        self.functions = {}
        for schema in self.schemas:
            name = schema["name"]
            function = getattr(self.env, name, None)
            if not callable(function):
                raise RuntimeError(
                    f"Schema tool {name!r} has no callable implementation"
                )
            self.functions[name] = function

    def dispatch(self, tool_call):
        if not isinstance(tool_call, dict):
            return {"error": "schema_violation: tool call must be an object"}

        name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})

        if not isinstance(name, str) or not name:
            return {"error": "schema_violation: tool name must be a string"}

        if not isinstance(arguments, dict):
            return {"error": "schema_violation: arguments must be an object"}

        if name not in self.functions:
            # D4 deliberately keeps the old method only to return
            # a helpful migration error. It is not advertised in the schema.
            if self.drift == "D4" and name == "get_bucket_acl":
                return self.env.get_bucket_acl(**arguments)

            return {"error": f"unknown_tool: Tool {name!r} not found"}

        try:
            return self.functions[name](**arguments)
        except TypeError as exc:
            return {"error": f"schema_violation: {exc}"}


def create_dispatcher(env, drift=None, enabled=True):
    """Create a dispatcher with matching environment and tool schemas."""
    return DriftToolDispatcher(env, drift=drift, enabled=enabled)

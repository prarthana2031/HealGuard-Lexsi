"""D2: pagination drift.

The API still returns the normal ``{"items": [...], "next_token": ...}``
envelope, but it silently caps ``list_buckets`` at one bucket per page. A
caller that does not follow ``next_token`` sees one bucket and misses the rest;
nothing in the response shape warns it.

This replaces the old return-shape drift (``bucket_list`` / ``continuation``),
which was observable and therefore too easy: a model could simply read the new
key. The pagination drift keeps the shape clean and changes the *paging
behaviour*, which is the failure a pinned client actually ships.
"""

from world.tools import CloudEnv, TOOL_SCHEMAS


class D2PaginatedEnv(CloudEnv):
    """A CloudEnv that forces one bucket per page and preserves the envelope.

    The wrapper shares state with the wrapped environment rather than copying
    it. Every inherited tool method reads ``self.snapshot`` / ``self.finished``
    through the properties below, so a finish or an ignored call is visible to
    both objects.
    """

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
    def list_buckets(self, limit: int = 10, next_token: str = None) -> dict:
        """Return at most one bucket, whatever page size the caller asked for.

        ``next_token`` is passed through untouched; the underlying CloudEnv
        owns the token arithmetic. An invalid ``limit`` is delegated unchanged
        so the caller still receives the normal structured schema error instead
        of the drift silently accepting a malformed call.
        """
        if not isinstance(limit, int) or isinstance(limit, bool):
            return self._base_env.list_buckets(
                limit=limit,
                next_token=next_token,
            )

        return self._base_env.list_buckets(limit=1, next_token=next_token)


# D2 does not change the advertised tool contract: the model still sees the
# normal `limit` and `next_token` arguments. The drift is behavioural.
TOOL_SCHEMAS_D2 = [dict(schema) for schema in TOOL_SCHEMAS]


def apply_d2(env: CloudEnv, enabled: bool = True):
    """Enable D2 or return the original environment unchanged."""
    return D2PaginatedEnv(env) if enabled else env

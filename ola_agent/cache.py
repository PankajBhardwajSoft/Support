"""Response cache for the grounded-generation step (Part 4, Task 16).

In memory, keyed by the normalised question text. The question is already
phone-masked when it gets here, so no key holds a raw phone number. The
cached value is the crew's result (draft, passages, ticket result); the
groundedness check and the Autogen review still run on every hit.

Chat follow-ups ("and its status?") are rewritten with their ticket ID before
they reach the pipeline, so one session's answer is never served to another
for a different ticket. `/add-document` clears the cache.
"""

from __future__ import annotations

import re
import threading
import unicodedata
from collections import OrderedDict
from typing import Any

from ola_agent.mock_llm import RECORD_ID_PATTERN

MAX_ENTRIES = 256

_TRAILING = re.compile(r"[\s?.!]+$")


def normalize_query(text: str) -> str:
    """Lower-case, Unicode-normalised, single-spaced, without trailing punctuation."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")  # drop zero-width characters
    return _TRAILING.sub("", " ".join(text.split()))


def cache_key(text: str, client=None, mock: bool = True) -> tuple:
    """The key for one question.

    Retrieval ignores case (the embedding model lower-cases its input), but the
    ticket ID match does not: "tkt-0023" is not looked up. So the exact ticket
    IDs in the question are part of the key: "tkt-0023" never gets the cached
    answer for "TKT-0023".
    """
    return (id(client), mock, normalize_query(text), tuple(RECORD_ID_PATTERN.findall(text or "")))


class ResponseCache:
    def __init__(self, max_entries: int = MAX_ENTRIES):
        self.max_entries = max_entries
        self._data: OrderedDict[tuple, Any] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple) -> Any | None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
            self.misses += 1
            return None

    def put(self, key: tuple, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)  # drop the oldest

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = self.misses = 0

    def __len__(self) -> int:
        return len(self._data)

    def keys(self) -> list[tuple]:
        with self._lock:
            return list(self._data)


CACHE = ResponseCache()

"""Deterministic mock model client for Autogen (Part 4, Task 14).

Built on Autogen's own extension point `autogen_core.models.ChatCompletionClient`,
the same idea as `MockLLM` for CrewAI: no call leaves the process, no key,
no network. Each agent gets its own client with a `responder` that turns the
messages it can see into its reply text.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from autogen_core.models import ChatCompletionClient, CreateResult, LLMMessage, ModelFamily, ModelInfo, RequestUsage

Responder = Callable[[Sequence[LLMMessage]], str]

_NO_USAGE = RequestUsage(prompt_tokens=0, completion_tokens=0)


class MockChatClient(ChatCompletionClient):
    def __init__(self, responder: Responder):
        self.responder = responder
        self.call_count = 0

    async def create(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = (), tool_choice: Any = "auto",
                     json_output: Any = None, extra_create_args: Mapping[str, Any] = {},
                     cancellation_token: Any = None) -> CreateResult:
        self.call_count += 1
        return CreateResult(finish_reason="stop", content=self.responder(messages), usage=_NO_USAGE, cached=False)

    def create_stream(self, *args, **kwargs):
        raise NotImplementedError("the mock client does not stream")

    async def close(self) -> None:
        pass

    def actual_usage(self) -> RequestUsage:
        return _NO_USAGE

    def total_usage(self) -> RequestUsage:
        return _NO_USAGE

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = ()) -> int:
        return 0

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = ()) -> int:
        return 1_000_000

    @property
    def model_info(self) -> ModelInfo:
        return {"vision": False, "function_calling": False, "json_output": True,
                "family": ModelFamily.UNKNOWN, "structured_output": True}

    @property
    def capabilities(self) -> ModelInfo:  # older name for model_info
        return self.model_info

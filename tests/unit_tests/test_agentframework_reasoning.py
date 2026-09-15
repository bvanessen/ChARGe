import asyncio
from types import SimpleNamespace
from typing import Any, Optional

from agent_framework import Content

from charge.clients.agentframework import AgentFrameworkAgent
from charge.tasks.task import Task


class FakeStream:
    def __init__(self, updates: list[Any], usage_details: Any = None) -> None:
        self.updates = updates
        self.usage_details = usage_details

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for update in self.updates:
            yield update

    async def get_final_response(self) -> Any:
        return SimpleNamespace(
            messages=[SimpleNamespace(text="final answer")],
            usage_details=self.usage_details,
        )


class FakeAgent:
    def __init__(self, stream: FakeStream) -> None:
        self.stream = stream

    async def run(self, *args: Any, **kwargs: Any) -> FakeStream:
        return self.stream


class LegacyCallback:
    def __init__(self) -> None:
        self.reasoning_updates: list[tuple[str, Optional[str]]] = []

    async def on_task_start(self) -> None:
        pass

    async def on_task_finish(self) -> None:
        pass

    async def on_reasoning_update(
        self, text: str, *, source: Optional[str] = None
    ) -> None:
        self.reasoning_updates.append((text, source))

    async def on_tool_call(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def on_tool_result(self, *args: Any, **kwargs: Any) -> None:
        pass


class StreamingCallback(LegacyCallback):
    def __init__(self) -> None:
        super().__init__()
        self.reasoning_deltas: list[tuple[str, Optional[str]]] = []
        self.reasoning_completions: list[tuple[str, Optional[str]]] = []

    async def on_reasoning_delta(
        self, text: str, *, source: Optional[str] = None
    ) -> None:
        self.reasoning_deltas.append((text, source))

    async def on_reasoning_complete(
        self, text: str, *, source: Optional[str] = None
    ) -> None:
        self.reasoning_completions.append((text, source))


def make_reasoning_stream() -> FakeStream:
    def update(*contents: Content) -> Any:
        return SimpleNamespace(contents=list(contents))

    return FakeStream(
        [
            update(
                Content.from_text_reasoning(
                    text="Compared ",
                    raw_representation=SimpleNamespace(
                        type="response.reasoning_summary_text.delta"
                    ),
                )
            ),
            update(
                Content.from_text_reasoning(
                    text="the options.",
                    raw_representation=SimpleNamespace(
                        type="response.reasoning_summary_text.delta"
                    ),
                )
            ),
            update(Content.from_text(text="final answer")),
        ]
    )


def make_client(callback: Any) -> AgentFrameworkAgent:
    return AgentFrameworkAgent(
        task=Task(user_prompt="Test reasoning callbacks"),
        client=object(),
        agent_key="test-agent",
        model="test-model",
        callback=callback,
    )


def test_reasoning_deltas_are_merged_for_legacy_callbacks() -> None:
    async def run() -> None:
        callback = LegacyCallback()
        client = make_client(callback)

        result = await client._execute_with_retries(
            FakeAgent(make_reasoning_stream()), "prompt", object()
        )

        assert result == "final answer"
        assert callback.reasoning_updates == [("Compared the options.", "test-agent")]

    asyncio.run(run())


def test_reasoning_deltas_use_streaming_callback_extension() -> None:
    async def run() -> None:
        callback = StreamingCallback()
        client = make_client(callback)

        await client._execute_with_retries(
            FakeAgent(make_reasoning_stream()), "prompt", object()
        )

        assert callback.reasoning_deltas == [
            ("Compared ", "test-agent"),
            ("the options.", "test-agent"),
        ]
        assert callback.reasoning_completions == [
            ("Compared the options.", "test-agent")
        ]
        assert callback.reasoning_updates == []

    asyncio.run(run())


def test_latest_response_usage_is_kept_separate_from_aggregate_usage() -> None:
    async def run() -> None:
        callback = LegacyCallback()
        stream = FakeStream(
            [
                SimpleNamespace(
                    contents=[
                        Content.from_usage(
                            usage_details={
                                "input_token_count": 100,
                                "output_token_count": 10,
                                "total_token_count": 110,
                            }
                        )
                    ]
                ),
                SimpleNamespace(
                    contents=[
                        Content.from_usage(
                            usage_details={
                                "input_token_count": 150,
                                "output_token_count": 20,
                                "openai.reasoning_tokens": 5,
                                "total_token_count": 170,
                            }
                        )
                    ]
                ),
            ],
            usage_details={
                "input_token_count": 250,
                "output_token_count": 30,
                "openai.reasoning_tokens": 5,
                "total_token_count": 280,
            },
        )
        client = make_client(callback)

        await client._execute_with_retries(FakeAgent(stream), "prompt", object())

        assert client.get_model_info()["lastUsage"] == {
            "inputTokens": 250,
            "outputTokens": 30,
            "reasoningTokens": 5,
            "totalTokens": 280,
            "contextInputTokens": 150,
            "contextOutputTokens": 20,
            "contextReasoningTokens": 5,
            "contextTotalTokens": 170,
        }

    asyncio.run(run())

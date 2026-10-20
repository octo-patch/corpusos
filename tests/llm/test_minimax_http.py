# SPDX-License-Identifier: Apache-2.0
"""LLM §8.3/§8.4: exercise public calls through the real urllib transport."""

from __future__ import annotations

import io
import json
import urllib.request

import pytest

from corpus_sdk.llm.llm_base import BadRequest
from corpus_sdk.llm.providers.minimax import MiniMaxAdapter, MiniMaxChunk, MiniMaxCompletion

pytestmark = pytest.mark.asyncio

FLASH = "MiniMax-M3.1-Flash"
TOOL_CALL = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "lookup", "arguments": '{"query":"hello"}'},
}
ASSISTANT = {
    "role": "assistant",
    "content": None,
    "reasoning_content": "Preserve this reasoning.",
    "tool_calls": [TOOL_CALL],
}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        },
    }
]


@pytest.fixture
def http_calls(monkeypatch):
    calls = []

    def urlopen(request, data):
        assert data is None
        body = json.loads(request.data)
        assert request.get_method() == "POST"
        assert request.get_header("Authorization") == "Bearer offline-test"
        assert request.get_header("Content-type") == "application/json"
        calls.append((request.full_url, body))
        if body["stream"]:
            delta = {**ASSISTANT, "tool_calls": [{"index": 0, **TOOL_CALL}]}
            events = [
                {"choices": [{"delta": delta, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            ]
            return io.BytesIO(
                b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
                + b"data: [DONE]\n\n"
            )
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [{"message": ASSISTANT, "finish_reason": "tool_calls"}],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
                }
            ).encode()
        )

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return calls


@pytest.mark.parametrize(
    "region,host", [("global_en", "api.minimax.io"), ("cn_zh", "api.minimax.cn")]
)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "saved,override,expected",
    [
        (None, None, FLASH),
        ("MiniMax-M3", None, "MiniMax-M3"),
        ("MiniMax-M2.7", None, "MiniMax-M2.7"),
        (None, "MiniMax-M3", "MiniMax-M3"),
        (None, "MiniMax-M2.7", "MiniMax-M2.7"),
    ],
)
async def test_default_and_saved_models_at_http_boundary(
    http_calls,
    region,
    host,
    stream,
    saved,
    override,
    expected,
):
    """§8.3/§8.4: catalog changes never replace explicit saved model IDs."""
    options = {"model": saved} if saved else {}
    adapter = MiniMaxAdapter(api_key="offline-test", region=region, **options)
    request = {"messages": [{"role": "user", "content": "hello"}], "model": override}
    if stream:
        chunks = [chunk async for chunk in adapter.stream(**request)]
        assert chunks[-1].is_final
        assert chunks[0].model == expected
    else:
        result = await adapter.complete(**request)
        assert result.model == expected
    url, body = http_calls[-1]
    assert url == f"https://{host}/v1/chat/completions"
    assert body["model"] == expected
    assert body["stream"] is stream
    assert "thinking" not in body  # API default: adaptive.
    assert "reasoning_effort" not in body  # Flash API default: max.
    assert "service_tier" not in body
    assert "reasoning_split" not in body
    assert (await adapter.capabilities()).supported_models == (FLASH, "MiniMax-M3")


@pytest.mark.parametrize("region", ["global_en", "cn_zh"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
async def test_flash_effort_at_http_boundary(http_calls, region, stream, effort):
    """§8.3/§8.4: valid Flash effort and adaptive thinking reach both transports."""
    adapter = MiniMaxAdapter(
        api_key="offline-test", region=region, reasoning_effort=effort, thinking="adaptive"
    )
    request = {"messages": [{"role": "user", "content": "hello"}]}
    if stream:
        _ = [chunk async for chunk in adapter.stream(**request)]
    else:
        await adapter.complete(**request)
    body = http_calls[-1][1]
    assert body["reasoning_effort"] == effort
    assert body["thinking"] == {"type": "adaptive"}


@pytest.mark.parametrize("stream", [False, True])
async def test_m3_disabled_and_flash_override_validation(http_calls, stream):
    """§8.3/§8.4: M3 accepts disabled thinking; a Flash override rejects it."""
    adapter = MiniMaxAdapter(
        api_key="offline-test", model="MiniMax-M3", reasoning_effort="low", thinking="disabled"
    )
    request = {"messages": [{"role": "user", "content": "hello"}]}
    if stream:
        _ = [chunk async for chunk in adapter.stream(**request)]
    else:
        await adapter.complete(**request)
    assert http_calls[-1][1]["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in http_calls[-1][1]
    with pytest.raises(BadRequest, match="requires adaptive"):
        if stream:
            _ = [chunk async for chunk in adapter.stream(model=FLASH, **request)]
        else:
            await adapter.complete(model=FLASH, **request)
    assert len(http_calls) == 1


@pytest.mark.parametrize("effort", ["none", "invalid", ""])
async def test_invalid_effort_rejected_before_http(http_calls, effort):
    """§6: reject unsupported depth values before making an HTTP request."""
    with pytest.raises(BadRequest, match="reasoning_effort"):
        MiniMaxAdapter(api_key="offline-test", reasoning_effort=effort)
    assert not http_calls


@pytest.mark.parametrize("region", ["global_en", "cn_zh"])
@pytest.mark.parametrize("stream", [False, True])
async def test_multimodal_tool_reasoning_history_roundtrip(http_calls, region, stream):
    """§8.3/§8.4: preserve full client history, content parts and response deltas."""
    adapter = MiniMaxAdapter(api_key="offline-test", region=region)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,aGVsbG8=",
                        "detail": "high",
                    },
                },
                {"type": "video_url", "video_url": {"url": "mm_file://offline-file"}},
            ],
        }
    ]
    first = await adapter.complete(messages=messages, tools=TOOLS, tool_choice="auto")
    assert isinstance(first, MiniMaxCompletion)
    assert first.message == ASSISTANT
    history = messages + [
        first.message,
        {
            "role": "tool",
            "content": "result",
            "tool_call_id": "call_1",
        },
    ]
    if stream:
        chunks = [
            chunk
            async for chunk in adapter.stream(
                messages=history,
                tools=TOOLS,
                tool_choice="auto",
            )
        ]
        assert isinstance(chunks[0], MiniMaxChunk)
        assert chunks[0].delta["reasoning_content"] == ASSISTANT["reasoning_content"]
        assert chunks[0].delta["tool_calls"] == [{"index": 0, **TOOL_CALL}]
        assert chunks[0].tool_calls[0].function.name == "lookup"
    else:
        await adapter.complete(messages=history, tools=TOOLS, tool_choice="auto")
    body = http_calls[-1][1]
    assert body["messages"] == history
    assert body["tools"] == TOOLS
    assert body["tool_choice"] == "auto"
    assert len(messages) == 1


async def test_legacy_token_count_remains_usable(http_calls):
    """§8.3: legacy token estimates remain available outside the current catalog."""
    adapter = MiniMaxAdapter(api_key="offline-test")
    assert await adapter.count_tokens("hello world", model="MiniMax-M2.7") > 0
    assert not http_calls


@pytest.mark.parametrize("content", [123, {"type": "text"}, ["invalid"], [{"type": 1}]])
async def test_invalid_message_content_rejected(http_calls, content):
    """§6: multimodal support does not bypass message-shape validation."""
    adapter = MiniMaxAdapter(api_key="offline-test")
    with pytest.raises(BadRequest):
        await adapter.complete(messages=[{"role": "user", "content": content}])
    assert not http_calls


async def test_unknown_custom_model_keeps_existing_rejection(http_calls):
    """§6: unsupported custom IDs are rejected, never silently reset to Flash."""
    adapter = MiniMaxAdapter(api_key="offline-test")
    with pytest.raises(BadRequest):
        await adapter.complete(
            messages=[{"role": "user", "content": "hello"}], model="custom-model"
        )
    assert not http_calls

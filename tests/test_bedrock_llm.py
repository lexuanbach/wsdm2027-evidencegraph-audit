import os

import pytest

from evidencegraph.bedrock_llm import (
    BEDROCK_TOKEN_ENV,
    STRUCTURED_TOOL,
    BedrockConverseClient,
    BedrockInferenceError,
)


class FakeBedrock:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def converse(self, **kwargs):
        self.requests.append(kwargs)
        return self.response


def successful_response(payload=None):
    return {
        "ResponseMetadata": {"HTTPStatusCode": 200},
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "name": STRUCTURED_TOOL,
                            "input": payload or {"label": "SUPPORTED"},
                        }
                    }
                ]
            }
        },
        "usage": {"inputTokens": 11, "outputTokens": 7},
    }


def test_structured_converse_maps_messages_usage_and_schema():
    fake = FakeBedrock(successful_response())
    client = BedrockConverseClient(client=fake)
    schema = {
        "type": "object",
        "required": ["label"],
        "properties": {"label": {"type": "string"}},
    }
    payload, response = client.chat_json(
        "us.anthropic.verified-profile",
        [
            {"role": "system", "content": "Follow evidence."},
            {"role": "user", "content": "Classify."},
        ],
        temperature=0,
        seed=103,
        max_tokens=32,
        json_schema=schema,
    )
    assert payload == {"label": "SUPPORTED"}
    assert response.model == "us.anthropic.verified-profile"
    assert response.prompt_tokens == 11
    assert response.completion_tokens == 7
    request = fake.requests[0]
    assert request["system"] == [{"text": "Follow evidence."}]
    assert request["messages"] == [
        {"role": "user", "content": [{"text": "Classify."}]}
    ]
    assert request["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"][
        "json"
    ] == schema


def test_structured_converse_rejects_missing_or_ambiguous_tool_output():
    response = successful_response()
    response["output"]["message"]["content"] = [{"text": "{}"}]
    client = BedrockConverseClient(client=FakeBedrock(response))
    with pytest.raises(BedrockInferenceError, match="exactly one"):
        client.chat_json(
            "model-id",
            [{"role": "user", "content": "Return it."}],
            json_schema={"type": "object"},
        )


def test_client_requires_bearer_token_before_constructing_sdk_client(monkeypatch):
    monkeypatch.delenv(BEDROCK_TOKEN_ENV, raising=False)
    with pytest.raises(BedrockInferenceError, match="is missing"):
        BedrockConverseClient()
    assert BEDROCK_TOKEN_ENV not in os.environ


def test_bedrock_requires_explicit_schema_and_verified_model_id():
    client = BedrockConverseClient(client=FakeBedrock(successful_response()))
    with pytest.raises(ValueError, match="require json_schema"):
        client.chat_json("model-id", [{"role": "user", "content": "x"}])
    with pytest.raises(ValueError, match="exact verified"):
        client.chat_json(
            "REPLACE_WITH_MODEL",
            [{"role": "user", "content": "x"}],
            json_schema={"type": "object"},
        )

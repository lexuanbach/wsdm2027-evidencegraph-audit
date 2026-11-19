"""Fail-closed Amazon Bedrock Converse adapter for structured benchmark calls."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from .local_llm import LocalResponse


BEDROCK_TOKEN_ENV = "AWS_BEARER_TOKEN_BEDROCK"
STRUCTURED_TOOL = "emit_structured_output"


class BedrockInferenceError(RuntimeError):
    """Safe resumable error that never includes credential material."""


def _safe_bedrock_error(error: Exception) -> BedrockInferenceError:
    if isinstance(error, ClientError):
        code = str(error.response.get("Error", {}).get("Code", "Unknown"))
        status = error.response.get("ResponseMetadata", {}).get(
            "HTTPStatusCode", "unknown"
        )
        return BedrockInferenceError(f"Bedrock request failed: {code} (HTTP {status})")
    return BedrockInferenceError(f"Bedrock SDK failure: {type(error).__name__}")


def _convert_messages(
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    system = []
    converted = []
    for message in messages:
        if set(message) != {"role", "content"}:
            raise ValueError("Bedrock message has the wrong schema")
        role = str(message["role"])
        content = str(message["content"])
        if not content:
            raise ValueError("Bedrock message content is empty")
        if role == "system":
            system.append({"text": content})
        elif role in {"user", "assistant"}:
            converted.append({"role": role, "content": [{"text": content}]})
        else:
            raise ValueError(f"unsupported Bedrock message role: {role}")
    if not converted or converted[-1]["role"] != "user":
        raise ValueError("Bedrock conversation must end with a user message")
    return system, converted


class BedrockConverseClient:
    """Expose the subset of ``OllamaClient`` used by benchmark protocols.

    Bedrock Converse does not expose a sampling seed. Callers retain the
    preregistered run seed as an experiment identifier and must separately
    record ``seed_control=temperature-zero-no-api-seed``.
    """

    def __init__(
        self,
        *,
        region: str = "us-east-1",
        timeout: int = 300,
        client: Any | None = None,
    ) -> None:
        if not region.strip():
            raise ValueError("Bedrock region is empty")
        self.region = region
        if client is None:
            if not os.environ.get(BEDROCK_TOKEN_ENV):
                raise BedrockInferenceError(
                    f"{BEDROCK_TOKEN_ENV} is missing; load it with ../load_bedrock_key.sh"
                )
            client = boto3.client(
                "bedrock-runtime",
                region_name=region,
                config=Config(
                    connect_timeout=20,
                    read_timeout=timeout,
                    retries={"max_attempts": 3, "mode": "standard"},
                ),
            )
        self.client = client

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        seed: int = 0,
        max_tokens: int = 512,
        json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
        **_: Any,
    ) -> LocalResponse:
        del seed  # Converse has no seed parameter; see class contract.
        if not model.strip() or model.startswith("REPLACE_WITH"):
            raise ValueError("Bedrock requires an exact verified model or profile ID")
        if max_tokens < 1:
            raise ValueError("Bedrock max_tokens must be positive")
        if json_mode and json_schema is not None:
            raise ValueError("json_mode and json_schema are mutually exclusive")
        if json_mode:
            raise ValueError("Bedrock benchmark calls require an explicit JSON schema")
        system, converted = _convert_messages(messages)
        request: dict[str, Any] = {
            "modelId": model,
            "messages": converted,
            "inferenceConfig": {
                "maxTokens": int(max_tokens),
                "temperature": float(temperature),
            },
        }
        if system:
            request["system"] = system
        if json_schema is not None:
            request["toolConfig"] = {
                "tools": [
                    {
                        "toolSpec": {
                            "name": STRUCTURED_TOOL,
                            "description": "Return the requested structured result.",
                            "inputSchema": {"json": json_schema},
                        }
                    }
                ],
                "toolChoice": {"tool": {"name": STRUCTURED_TOOL}},
            }
        started = time.monotonic()
        try:
            record = self.client.converse(**request)
        except (BotoCoreError, ClientError) as error:
            raise _safe_bedrock_error(error) from error
        wall_seconds = time.monotonic() - started
        status = record.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status != 200:
            raise BedrockInferenceError(f"Bedrock returned non-success HTTP {status}")
        blocks = record.get("output", {}).get("message", {}).get("content", [])
        if json_schema is not None:
            inputs = [
                block["toolUse"]["input"]
                for block in blocks
                if block.get("toolUse", {}).get("name") == STRUCTURED_TOOL
            ]
            if len(inputs) != 1 or not isinstance(inputs[0], Mapping):
                raise BedrockInferenceError(
                    "Bedrock did not return exactly one structured-output tool call"
                )
            content = json.dumps(dict(inputs[0]), ensure_ascii=False)
        else:
            text_blocks = [str(block["text"]) for block in blocks if "text" in block]
            if not text_blocks:
                raise BedrockInferenceError("Bedrock returned no text content")
            content = "\n".join(text_blocks)
        usage = record.get("usage", {})
        return LocalResponse(
            content=content,
            model=model,
            wall_seconds=wall_seconds,
            prompt_tokens=int(usage.get("inputTokens", 0)),
            completion_tokens=int(usage.get("outputTokens", 0)),
        )

    def chat_json(
        self,
        *args: Any,
        json_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Any, LocalResponse]:
        if json_schema is None:
            raise ValueError("Bedrock structured calls require json_schema")
        response = self.chat(*args, json_schema=json_schema, **kwargs)
        return json.loads(response.content), response

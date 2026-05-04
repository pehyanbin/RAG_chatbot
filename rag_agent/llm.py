from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Protocol
import requests


class LLM(Protocol):
    def generate(self, system: str, user: str) -> str:
        ...


@dataclass
class GenerationConfig:
    temperature: float = 0.2
    max_tokens: int = 900


class OpenAICompatibleLLM:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        config: GenerationConfig,
        extra_headers: dict[str, str] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.config = config
        self.extra_headers = extra_headers or {}

    def generate(self, system: str, user: str) -> str:
        if not self.api_key and "localhost" not in self.base_url and "127.0.0.1" not in self.base_url:
            raise RuntimeError(f"Missing API key for model {self.model}.")

        headers = {
            "Authorization": f"Bearer {self.api_key or 'ollama'}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"]


class GeminiLLM:
    def __init__(self, api_key: str, model: str, config: GenerationConfig):
        self.api_key = api_key
        self.model = model
        self.config = config

    def generate(self, system: str, user: str) -> str:
        if not self.api_key:
            raise RuntimeError("Missing GEMINI_API_KEY.")

        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self.model}:generateContent?key={self.api_key}"
        )
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {
                    "temperature": self.config.temperature,
                    "maxOutputTokens": self.config.max_tokens,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        candidates = payload.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "\n".join(part.get("text", "") for part in parts).strip()


class AnthropicLLM:
    def __init__(self, api_key: str, model: str, config: GenerationConfig):
        self.api_key = api_key
        self.model = model
        self.config = config

    def generate(self, system: str, user: str) -> str:
        if not self.api_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY.")

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "system": system,
                "messages": [{"role": "user", "content": user}],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        return "\n".join(
            block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text"
        ).strip()


class LocalCliLLM:
    """
    Bridge for OAuth/subscription-based local CLIs.

    The command is read from a JSON array env var and the final RAG prompt is passed via STDIN.
    Example:
      CODEX_COMMAND_JSON=["codex","exec"]
    """

    def __init__(self, command_json_env: str):
        self.command_json_env = command_json_env

    def generate(self, system: str, user: str) -> str:
        raw = os.getenv(self.command_json_env, "")
        if not raw:
            raise RuntimeError(f"Missing {self.command_json_env}. Set it to a JSON array command.")

        try:
            command = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{self.command_json_env} must be a JSON array, for example [\"gemini\",\"-p\"].") from exc

        if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
            raise RuntimeError(f"{self.command_json_env} must be a JSON array of strings.")

        prompt = f"{system}\n\n{user}"
        result = subprocess.run(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=180,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"CLI provider failed with exit code {result.returncode}.\nSTDERR:\n{result.stderr}"
            )
        return result.stdout.strip()


def build_llm(provider: str, config: GenerationConfig) -> LLM:
    provider = provider.lower().strip()

    if provider == "openai_api_key":
        return OpenAICompatibleLLM(
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            config=config,
        )

    if provider == "openai_compatible_api":
        return OpenAICompatibleLLM(
            base_url=os.getenv("OPENAI_COMPATIBLE_BASE_URL", ""),
            api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY", ""),
            model=os.getenv("OPENAI_COMPATIBLE_MODEL", ""),
            config=config,
        )

    if provider == "nvidia_nim_api":
        return OpenAICompatibleLLM(
            base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            api_key=os.getenv("NVIDIA_API_KEY", ""),
            model=os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct"),
            config=config,
        )

    if provider == "ollama_api":
        return OpenAICompatibleLLM(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            config=config,
        )

    if provider == "openrouter_api":
        return OpenAICompatibleLLM(
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini"),
            config=config,
            extra_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
                "X-Title": os.getenv("OPENROUTER_APP_NAME", "RAG Agent Starter"),
            },
        )

    if provider == "gemini_api_key":
        return GeminiLLM(
            api_key=os.getenv("GEMINI_API_KEY", ""),
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            config=config,
        )

    if provider == "anthropic_api_key":
        return AnthropicLLM(
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
            config=config,
        )

    if provider == "deepseek_api":
        return OpenAICompatibleLLM(
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            config=config,
        )

    if provider == "xai_api":
        return OpenAICompatibleLLM(
            base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
            api_key=os.getenv("XAI_API_KEY", ""),
            model=os.getenv("XAI_MODEL", "grok-4"),
            config=config,
        )

    if provider == "moonshot_ai":
        return OpenAICompatibleLLM(
            base_url=os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1"),
            api_key=os.getenv("MOONSHOT_API_KEY", ""),
            model=os.getenv("MOONSHOT_MODEL", "kimi-k2-0905-preview"),
            config=config,
        )

    if provider == "codex_oauth":
        return LocalCliLLM("CODEX_COMMAND_JSON")

    if provider == "openai_plus_pro_subscription":
        return LocalCliLLM("CODEX_COMMAND_JSON")

    if provider == "gemini_cli_oauth":
        return LocalCliLLM("GEMINI_CLI_COMMAND_JSON")

    if provider == "google_antigravity_oauth":
        return LocalCliLLM("ANTIGRAVITY_COMMAND_JSON")

    if provider == "opencode":
        return LocalCliLLM("OPENCODE_COMMAND_JSON")

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

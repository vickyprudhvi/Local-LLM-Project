"""ask_local, ask_claude, load_system_prompt, trim_history."""

import base64
import os

import requests
from anthropic import Anthropic
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()

OLLAMA_URL = "http://localhost:11434"
LOCAL_MODEL = os.environ["LOCAL_MODEL"]
CLAUDE_MODEL = os.environ["CLAUDE_MODEL"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

_anthropic_client = None

CLAUDE_TOOLS = [{"type": "web_search_20260209", "name": "web_search", "max_uses": 3}]


def load_system_prompt(path="system_prompt.txt"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "You are a helpful, concise assistant."


def trim_history(history, turns):
    """Keep the last `turns` user/assistant exchanges (2 messages each)."""
    max_messages = turns * 2
    if len(history) <= max_messages:
        return history
    return history[-max_messages:]


def ask_local(prompt, history, system_prompt):
    """POST to Ollama /api/chat. Returns (text, metrics dict)."""
    trimmed = trim_history(history, 12)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(trimmed)
    messages.append({"role": "user", "content": prompt})

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": LOCAL_MODEL,
                "messages": messages,
                "stream": False,
                "keep_alive": "10m",
            },
            timeout=120,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        console.print(f"[red]Local model call failed: {e}[/red]")
        return "Sorry, I couldn't reach the local model just now.", {}

    data = resp.json()
    text = data.get("message", {}).get("content", "").strip()
    metrics = {
        "prompt_tokens": data.get("prompt_eval_count"),
        "completion_tokens": data.get("eval_count"),
        "eval_duration": data.get("eval_duration"),
    }
    console.print(f"[dim]local metrics: {metrics}[/dim]")
    return text, metrics


def ask_claude(prompt, history, system_prompt, image_path=None):
    """Uses the Anthropic SDK. Returns (text, metrics dict). Attaches an image before the text block if given."""
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY, timeout=90.0)

    trimmed = trim_history(history, 6)

    content = []
    if image_path:
        with open(image_path, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode("utf-8")
        content.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            }
        )
    content.append({"type": "text", "text": prompt})

    messages = list(trimmed)
    messages.append({"role": "user", "content": content})

    try:
        resp = _anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2000,
            system=system_prompt,
            messages=messages,
            tools=CLAUDE_TOOLS,
        )
    except Exception as e:
        console.print(f"[red]Claude call failed: {e}[/red]")
        return "Sorry, I couldn't reach Claude just now.", {}

    parts = [block.text for block in resp.content if getattr(block, "type", None) == "text"]
    metrics = {
        "prompt_tokens": resp.usage.input_tokens,
        "completion_tokens": resp.usage.output_tokens,
    }
    console.print(f"[dim]claude metrics: {metrics}[/dim]")
    return "\n".join(parts).strip(), metrics

"""Router-dispatched built-ins: memory.remember and memory.recall.

Thin wrappers over the existing memory_store (ChromaDB) — the storage format,
embeddings, and remember/recall semantics are unchanged. These tools do the
data operation only and return a render directive; the conversational phrasing of
a recall (the ask_local summarization step) stays in the orchestrator, so the
tool never touches conversation history or system prompts. llm_callable=False:
selected by the router, never offered to the local tool-calling loop.
"""

import memory_store
from tools.base import BaseTool, ToolValidationError

# Moved verbatim from the former assistant.dispatch recall branch.
RECALL_SUMMARY_PROMPT = (
    "You're telling someone what you remember about them, out loud. Summarize the "
    "remembered facts in a natural, conversational sentence or two — no bullet points, "
    "no markdown, no raw list formatting. Be concise."
)


class RememberTool(BaseTool):
    name = "memory.remember"
    description = "Save a fact the user wants remembered for later."
    input_schema = {
        "type": "object",
        "properties": {"fact": {"type": "string", "description": "The fact to remember."}},
        "required": ["fact"],
    }
    timeout_seconds = 30.0
    llm_callable = False

    def validate_arguments(self, arguments: dict) -> dict:
        arguments = super().validate_arguments(arguments)
        if not isinstance(arguments.get("fact"), str):
            raise ToolValidationError("'fact' must be a string.")
        return arguments

    def execute(self, arguments: dict) -> dict:
        fact = arguments["fact"]
        fact_id = memory_store.remember(fact)
        if fact_id is None:
            return {"render": "speak", "text": "Sorry, I couldn't save that."}
        return {"render": "speak", "text": f"Got it, I'll remember: {fact}"}


class RecallTool(BaseTool):
    name = "memory.recall"
    description = "Recall stored memories about the user."
    input_schema = {
        "type": "object",
        "properties": {"topic": {"type": "string", "description": "Optional subject to recall."}},
    }
    timeout_seconds = 30.0
    llm_callable = False

    def execute(self, arguments: dict) -> dict:
        topic = arguments.get("topic")
        if topic:
            facts = memory_store.recall(topic, n_results=3)
            if not facts:
                return {"render": "speak", "text": "I don't have a relevant memory for that."}
        else:
            facts = memory_store.list_all(n_results=10)
            if not facts:
                return {"render": "speak", "text": "I don't have anything remembered yet."}
        content = "Here's what I remember: " + "; ".join(f["text"] for f in facts)
        return {"render": "summarize", "content": content, "instructions": RECALL_SUMMARY_PROMPT}

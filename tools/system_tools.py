"""Router-dispatched built-in: system.time.

Registered in the shared ToolRegistry but marked llm_callable=False, so it is
executed only via the router's tool route (through ToolExecutor) and never
offered to the local tool-calling loop. Its execute() returns a render directive
(see tool_dispatch._render) rather than final wording, keeping the tool free of
any knowledge of how the reply is spoken.
"""

from datetime import datetime

from tools.base import BaseTool


class TimeTool(BaseTool):
    name = "system.time"
    description = "Get the current date and time."
    input_schema = {"type": "object", "properties": {}}
    timeout_seconds = 5.0
    llm_callable = False

    def execute(self, arguments: dict) -> dict:
        # Same format string the former assistant.dispatch time branch used.
        return {"render": "speak", "text": datetime.now().strftime("%A, %B %d %Y, %I:%M %p")}

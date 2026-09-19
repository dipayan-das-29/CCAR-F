"""
Claude never runs your tools. When it wants one, it stops and returns stop_reason: "tool_use" with a tool_use block. Your code runs the function and sends the result back, and Claude continues. The exam tests that you get this handoff right:

Append Claude's entire response as an assistant message (the tool_use blocks must stay in history).
Run each requested tool.
Append one user message containing a tool_result block per call, each with a matching tool_use_id.

"""

import json
from anthropic import Anthropic
import dotenv
import os

dotenv.load_dotenv()

MODEL = "claude-haiku-4-5"

 # Access the value of the ANTHROPIC_API_KEY variable
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
if not anthropic_api_key:
        raise SystemExit("Set the ANTHROPIC_API_KEY environment variable first.")
    

# Set the Anthropic API key
client = Anthropic(api_key=anthropic_api_key)


# ---------- Logger helpers ----------
DIM, CYAN, GREEN = "\033[2m", "\033[36m", "\033[32m"
YELLOW, RED, MAGENTA = "\033[33m", "\033[31m", "\033[35m"
BOLD, RESET = "\033[1m", "\033[0m"


def indent(data):
    text = data if isinstance(data, str) else json.dumps(data, indent=2, default=str)
    return "\n".join("    " + line for line in text.split("\n"))


def log(color, label, data=None):
    print(f"{color}{label}{RESET}")
    if data is not None:
        print(indent(data))


def banner(title):
    line = "=" * 60
    print(f"\n{BOLD}{MAGENTA}{line}\n{title}\n{line}{RESET}")


def field(block, key, default=None):
    """Read a field from either a dict (our messages) or an SDK object (Claude's response)."""
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def summarize(msg, i):
    """One-line summary of a message so history is easy to scan."""
    role = msg["role"].ljust(9)
    content = msg["content"]
    if isinstance(content, str):
        return f'[{i}] {role} text: "{content}"'

    parts = []
    for b in content:
        btype = field(b, "type")
        if btype == "text":
            parts.append(f'text("{field(b, "text")[:40]}...")')
        elif btype == "tool_use":
            parts.append(f'tool_use({field(b, "name")}, id={field(b, "id")[-6:]})')
        elif btype == "tool_result":
            flag = ", ERROR" if field(b, "is_error") else ""
            parts.append(f'tool_result(for={field(b, "tool_use_id")[-6:]}{flag})')
        else:
            parts.append(btype)
    return f"[{i}] {role} " + " + ".join(parts)


def print_history(messages, title):
    log(DIM, f"{title} ({len(messages)} messages)")
    for i, m in enumerate(messages):
        print("    " + summarize(m, i))


# ---------- Tools ----------
TOOLS = [
    {
        "name": "get_order_status",
        "description": "Look up the status of a customer order by its ID.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "e.g. ORD-1001"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "get_delivery_eta",
        "description": "Get the estimated delivery time in minutes for an order.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "e.g. ORD-1001"}},
            "required": ["order_id"],
        },
    },
]

ORDERS = {
    "ORD-1001": {"status": "out_for_delivery", "eta_min": 25},
    "ORD-1002": {"status": "preparing", "eta_min": 70},
}


def execute_tool(name, tool_input):
    order = ORDERS.get(tool_input["order_id"])
    if order is None:
        raise ValueError(f"Order {tool_input['order_id']} not found")
    if name == "get_order_status":
        return json.dumps({"status": order["status"]})
    if name == "get_delivery_eta":
        return json.dumps({"eta_min": order["eta_min"]})
    raise ValueError(f"Unknown tool: {name}")


# ---------- Agentic loop (with logging) ----------
def run(user_message, test_name):
    banner(f"TEST: {test_name}")
    messages = [{"role": "user", "content": user_message}]
    log(CYAN, "USER INPUT", user_message)

    max_turns = 10  # safety net against runaway loops

    for turn in range(1, max_turns + 1):
        print(f"\n{BOLD}--- LOOP ITERATION {turn} ---{RESET}")
        print_history(messages, "Sending to Claude, history so far")

        response = client.messages.create(
            model=MODEL,
            max_tokens=500,
            tools=TOOLS,
            messages=messages,
        )

        # What came back
        log(YELLOW, f'CLAUDE RESPONSE  stop_reason = "{response.stop_reason}"')
        log(DIM, f"tokens: in={response.usage.input_tokens} out={response.usage.output_tokens}")
        for i, b in enumerate(response.content):
            if b.type == "text":
                log(YELLOW, f"  block[{i}] type=text", b.text)
            elif b.type == "tool_use":
                log(YELLOW, f"  block[{i}] type=tool_use  name={b.name}  id={b.id}", b.input)

        # Exit condition: Claude is done
        if response.stop_reason != "tool_use":
            final_text = next((b.text for b in response.content if b.type == "text"), "")
            log(GREEN, "\nLOOP ENDS (not tool_use). FINAL ANSWER:", final_text)
            print_history(messages, "Final history")
            return final_text

        # Append Claude's full response as the assistant message
        messages.append({"role": "assistant", "content": response.content})
        log(CYAN, f"APPENDED -> {summarize(messages[-1], len(messages) - 1)}")

        # Execute every requested tool
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            log(GREEN, f"EXECUTING {block.name}({json.dumps(block.input)})")
            try:
                result = execute_tool(block.name, block.input)
                log(GREEN, "  result:", result)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result}
                )
            except Exception as err:
                log(RED, f"  FAILED: {err}  (sending is_error: True)")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(err),
                        "is_error": True,
                    }
                )

        # Append ONE user message carrying all tool results
        messages.append({"role": "user", "content": tool_results})
        log(CYAN, f"APPENDED -> {summarize(messages[-1], len(messages) - 1)}")
        log(DIM, "  raw tool_result message:", messages[-1])

    raise RuntimeError(f"Stopped after {max_turns} iterations without a final answer")


# ---------- Test scenarios ----------
if __name__ == "__main__":
    run("Where is ORD-1001 and when will it arrive?", "Multi-tool (status + ETA)")
    run("Status of ORD-9999?", "Tool error path (is_error)")
    run("What is your return policy in one line?", "No tool needed (end_turn immediately)")
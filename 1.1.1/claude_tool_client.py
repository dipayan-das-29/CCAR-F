"""
Claude API client with two tools:
  1. calculator     — evaluates a math expression safely (no eval() of arbitrary code)
  2. web_search_stub — returns mock search results (swap in a real search API later)
 
Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY="your-key-here"
 
Run:
    python claude_tool_client.py
"""


import ast
import operator
import json
import os
from dotenv import load_dotenv
from anthropic import Anthropic

# Load variables from .env file
load_dotenv()


MODEL = "claude-haiku-4-5"

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
 
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
 
 
def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")
def calculator(expression: str) -> dict:
    """Safely evaluate an arithmetic expression (+ - * / // % ** and parens)."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return {"expression": expression, "result": result}
    except Exception as exc:
        return {"expression": expression, "error": str(exc)}
 
 
def web_search_stub(query: str) -> dict:
    """Mock web search — replace with a real search API call when ready."""
    mock_results = [
        {
            "title": f"Result 1 for '{query}'",
            "url": "https://example.com/result-1",
            "snippet": f"This is a mock snippet describing content related to '{query}'.",
        },
        {
            "title": f"Result 2 for '{query}'",
            "url": "https://example.com/result-2",
            "snippet": f"Another mock snippet with different info about '{query}'.",
        },
    ]
    return {"query": query, "results": mock_results}
 
 
TOOL_FUNCTIONS = {
    "calculator": lambda inp: calculator(inp["expression"]),
    "web_search_stub": lambda inp: web_search_stub(inp["query"]),
}
 
# ---------------------------------------------------------------------------
# Tool schemas (sent to the API)
# ---------------------------------------------------------------------------
 
TOOLS = [
    {
        "name": "calculator",
        "description": (
            "Evaluate a basic arithmetic expression and return the numeric result. "
            "Supports +, -, *, /, //, %, **, parentheses, and unary minus."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate, e.g. '(3 + 4) * 2'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "web_search_stub",
        "description": (
            "Return mock web search results for a query. This is a stub for testing — "
            "it does not hit a real search engine."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string",
                }
            },
            "required": ["query"],
        },
    },
]
 
# ---------------------------------------------------------------------------
# Conversation loop
# ---------------------------------------------------------------------------
 

def run_conversation(user_message: str, client: Anthropic, verbose: bool = True) -> str:
    messages = [{"role": "user", "content": user_message}]

    while True:
        # 1. Send the full conversation so far, with tools available
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        # 2. Always record what Claude said/did — including tool_use blocks —
        #    before deciding what to do next.
        messages.append({"role": "assistant", "content": response.content})

        # 3. THE key branch: stop_reason, not content inspection.
        if response.stop_reason != "tool_use":
            # Claude ended the turn on its own — done, no more tools requested.
            return "".join(b.text for b in response.content if b.type == "text")

        # 4. stop_reason == "tool_use": run every tool_use block in this turn.
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            fn = TOOL_FUNCTIONS.get(block.name)
            output = fn(block.input) if fn else {"error": f"Unknown tool: {block.name}"}
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(output),
            })

        # 5. Feed results back as a new user turn, then loop again.
        messages.append({"role": "user", "content": tool_results}) 
 
if __name__ == "__main__":
    # Access the value of the ANTHROPIC_API_KEY variable
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
            raise SystemExit("Set the ANTHROPIC_API_KEY environment variable first.")
    

    # Set the Anthropic API key
    client = Anthropic(api_key=anthropic_api_key)
 
    demo_prompt = (
        "What is (245 * 3) - 17? Also, search the web for 'best cut vegetable "
        "delivery practices' and summarize what you find."
    )
    answer = run_conversation(demo_prompt, client)
    print("\n--- Final answer ---")
    print(answer)
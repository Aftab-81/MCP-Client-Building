import asyncio
import os
import json

from dotenv import load_dotenv
from langchain.mcp import MCPAdapter
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openrouter import ChatOpenRouter

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError(
        "OpenRouter API key not found. Check your .env file."
    )

CONFIG = {
    "mcpServers": {
        "ArithmeticServer": {
            "command": "uv",
            "args": [
                "--directory",
                "D:/Multiple Projects/mcp-maths-server",
                "run",
                "fastmcp",
                "run",
                "main.py",
            ],
        },
        "ExpenseTracker": {
            "transport": "streamable-http",
            "url": "https://aftab-expense-tracker.fastmcp.app/mcp",
        },
    }
}


async def main():
    async with MCPAdapter(CONFIG) as adapter:

        # 1. Discover tools from both MCP servers
        tools = await adapter.list_tools()

        named_tool = {
            tool.name: tool
            for tool in tools
        }

        print("Available tools:", list(named_tool.keys()))

        # 2. Initialize the OpenRouter model
        model = ChatOpenRouter(
            model="nvidia/nemotron-3.5-lightning:free",
            api_key=api_key,
            temperature=0,
            max_tokens=1024,
        )

        # 3. Bind MCP tools to the model
        llm_with_tools = model.bind_tools(tools)

        messages = [
            HumanMessage(
                content="Add an expense of $50 for groceries today."
            )
        ]

        # 4. Ask the model to decide which tool to call
        response = await llm_with_tools.ainvoke(messages)

        if not response.tool_calls:
            print(f"\nLLM Response: {response.content}")
            return

        # 5. Execute every requested tool call
        tool_messages = []

        for tc in response.tool_calls:
            selected_tool = tc["name"]
            selected_tool_args = tc.get("args", {})
            selected_tool_id = tc["id"]

            print(
                f"\nSelected tool: {selected_tool}"
                f"\nArguments: {selected_tool_args}"
            )

            if selected_tool not in named_tool:
                raise ValueError(
                    f"Unknown tool requested: {selected_tool}"
                )

            result = await named_tool[selected_tool].ainvoke(
                selected_tool_args
            )

            tool_messages.append(
                ToolMessage(
                    content=json.dumps(
                        result,
                        default=str
                    ),
                    tool_call_id=selected_tool_id,
                )
            )

            print(f"Tool result: {result}")

        # 6. Send the assistant's tool calls AND all results
        # back to the model so it can generate a final answer.
        final_messages = messages + [response] + tool_messages

        final_response = await llm_with_tools.ainvoke(
            final_messages
        )

        print(f"\nFinal response: {final_response.content}")


if __name__ == "__main__":
    asyncio.run(main())


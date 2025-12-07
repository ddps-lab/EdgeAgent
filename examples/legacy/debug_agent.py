"""
Debug agent response format
"""

import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

load_dotenv()

async def main():
    # 간단한 MCP 서버
    servers = {
        "filesystem": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp/edgeagent_device"]
        }
    }

    client = MultiServerMCPClient(servers)

    async with client.session("filesystem") as session:
        tools = await load_mcp_tools(session)
        print(f"Loaded {len(tools)} tools")

        llm = ChatOpenAI(model="gpt-4", temperature=0)
        agent_executor = create_agent(llm, tools, system_prompt="You are a helpful assistant.")

        print("\n테스트 1: 간단한 요청")
        try:
            result = await agent_executor.ainvoke({
                "messages": [("user", "List files in /tmp/edgeagent_device")]
            })

            print(f"\nResult type: {type(result)}")
            print(f"Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
            print(f"\nFull result:")
            print(result)

            if isinstance(result, dict) and "messages" in result:
                print(f"\nMessages type: {type(result['messages'])}")
                print(f"Messages length: {len(result['messages'])}")
                print(f"\nLast message:")
                print(result['messages'][-1])
                print(f"\nLast message content:")
                print(result['messages'][-1].content)

        except Exception as e:
            print(f"\nException type: {type(e)}")
            print(f"Exception: {repr(e)}")
            print(f"String: '{str(e)}'")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

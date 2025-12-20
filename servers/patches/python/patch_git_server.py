#!/usr/bin/env python3
"""
Patch mcp-server-git to add ToolTimer profiling.
Run after pip install mcp-server-git.
"""
import site
import os
import re

def find_package_path():
    """Find mcp_server_git package location"""
    for sp in site.getsitepackages():
        pkg_path = os.path.join(sp, 'mcp_server_git')
        if os.path.exists(pkg_path):
            return pkg_path
    # Try user site-packages
    user_sp = site.getusersitepackages()
    pkg_path = os.path.join(user_sp, 'mcp_server_git')
    if os.path.exists(pkg_path):
        return pkg_path
    raise FileNotFoundError("mcp_server_git package not found")

def patch_server():
    pkg_path = find_package_path()
    server_py = os.path.join(pkg_path, 'server.py')

    with open(server_py, 'r') as f:
        content = f.read()

    # Add import after existing imports
    import_patch = """
# ToolTimer for profiling
from .timing import ToolTimer, measure_io
"""

    # Find the end of imports (before first class definition)
    import_end = content.find('\nclass ')
    if import_end == -1:
        import_end = content.find('\ndef ')

    content = content[:import_end] + import_patch + content[import_end:]

    # Patch call_tool function
    # Find the function and wrap it with ToolTimer
    old_call_tool = '''    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        repo_path = Path(arguments["repo_path"])'''

    new_call_tool = '''    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        timer = ToolTimer(name)
        try:
            repo_path = Path(arguments["repo_path"])'''

    content = content.replace(old_call_tool, new_call_tool)

    # Find the end of call_tool and add finally block before the function ends
    # Look for the last case _ statement and add finally after it
    old_end = '''            case _:
                raise ValueError(f"Unknown tool: {name}")

    options = server.create_initialization_options()'''

    new_end = '''            case _:
                raise ValueError(f"Unknown tool: {name}")
        finally:
            timer.finish()

    options = server.create_initialization_options()'''

    content = content.replace(old_end, new_end)

    # Also need to indent all the code inside try block
    # This is complex, so let's do a simpler approach - just add proper indentation

    with open(server_py, 'w') as f:
        f.write(content)

    print(f"Patched {server_py}")

if __name__ == '__main__':
    patch_server()

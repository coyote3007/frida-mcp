#!/usr/bin/env python3
"""
Frida MCP Server — Mobile security testing via Model Context Protocol.

Exposes Frida dynamic instrumentation tools over SSE transport so the
malware-lab MCP gateway can aggregate them under the ``frida__`` prefix.

Environment variables:
    FRIDA_HOST           Bind address (default: 0.0.0.0)
    FRIDA_PORT           Listen port  (default: 8772)
    ANDROID_DEVICE_ID    Remote ADB device (e.g. 192.168.2.220:5555)
    LOG_LEVEL            Logging verbosity (default: INFO)
"""

import json
import asyncio
import logging
import os
from typing import Any

import uvicorn
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from .tools import TOOLS
from . import device, adb, memory, android, files, hooks

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("frida-mcp")

FRIDA_HOST = os.getenv("FRIDA_HOST", "0.0.0.0")
FRIDA_PORT = int(os.getenv("FRIDA_PORT", "8772"))


def call_tool(name: str, arguments: dict) -> Any:
    """Dispatch tool call to implementation."""

    # ADB remote connection management
    if name == "adb_connect":
        return adb.adb_connect_remote(
            arguments["device_address"],
        )
    elif name == "adb_disconnect":
        return adb.adb_disconnect_remote(
            arguments.get("device_address"),
        )

    # Device & connection
    elif name == "list_devices":
        return device.list_devices()
    elif name == "list_processes":
        return device.list_processes(arguments.get("device_id"))
    elif name == "list_apps":
        return device.list_apps(arguments.get("device_id"))
    elif name == "connect":
        return device.connect(
            arguments["target"],
            arguments.get("device_id"),
            arguments.get("spawn", False),
            arguments.get("timeout_ms", 15000),
        )
    elif name == "disconnect":
        return device.disconnect()
    elif name == "is_connected":
        return device.is_connected()
    elif name == "list_sessions":
        return device.list_sessions()
    elif name == "switch_session":
        return device.switch_session(arguments["session_id"])

    # ADB app lifecycle
    elif name == "get_pid":
        return adb.get_pid(arguments["package"], arguments.get("device_id"))
    elif name == "launch_app":
        return adb.launch_app(
            arguments["package"],
            arguments.get("activity"),
            arguments.get("device_id"),
            arguments.get("timeout_ms", 10000),
        )
    elif name == "stop_app":
        return adb.stop_app(arguments["package"], arguments.get("device_id"))
    elif name == "spawn_and_attach":
        return device.spawn_and_attach(
            arguments["package"],
            arguments.get("device_id"),
            arguments.get("wait_ms", 3000),
        )

    # Memory
    elif name == "memory_list_modules":
        return memory.memory_list_modules()
    elif name == "memory_list_exports":
        return memory.memory_list_exports(arguments["module_name"])
    elif name == "memory_search":
        return memory.memory_search(arguments["pattern"], arguments.get("is_string", False))
    elif name == "memory_read":
        return memory.memory_read(arguments["address"], arguments["size"])
    elif name == "memory_write":
        return memory.memory_write(arguments["address"], arguments["hex_bytes"])
    elif name == "get_module_base":
        return memory.get_module_base(arguments["name"])

    # Android
    elif name == "android_list_classes":
        return android.android_list_classes(arguments.get("pattern"))
    elif name == "android_list_methods":
        return android.android_list_methods(arguments["class_name"])
    elif name == "android_hook_method":
        return android.android_hook_method(
            arguments["class_name"],
            arguments["method_name"],
            arguments.get("dump_args", True),
            arguments.get("dump_return", True),
            arguments.get("dump_backtrace", False),
        )
    elif name == "android_search_classes":
        return android.android_search_classes(arguments["pattern"])
    elif name == "android_ssl_pinning_disable":
        return android.android_ssl_pinning_disable()
    elif name == "android_get_current_activity":
        return android.android_get_current_activity()
    elif name == "dump_class":
        return android.dump_class(arguments["class_name"])
    elif name == "heap_search":
        return android.heap_search(arguments["class_name"], arguments.get("max_results", 10))
    elif name == "run_java":
        return android.run_java(arguments["code"])

    # Files
    elif name == "file_ls":
        return files.file_ls(arguments.get("path", "."))
    elif name == "file_read":
        return files.file_read(arguments["path"])
    elif name == "file_download":
        return files.file_download(arguments["remote_path"], arguments["local_path"])

    # Hooks
    elif name == "run_script":
        return hooks.run_script(arguments["js_code"])
    elif name == "install_hook":
        return hooks.install_hook(arguments["js_code"], arguments.get("name"))
    elif name == "get_hook_messages":
        return hooks.get_hook_messages(arguments.get("clear", False))
    elif name == "clear_hook_messages":
        return hooks.clear_hook_messages()
    elif name == "uninstall_hooks":
        return hooks.uninstall_hooks()
    elif name == "list_hooks":
        return hooks.list_hooks()
    elif name == "hook_native":
        return hooks.hook_native(arguments["module"], arguments["offset"], arguments.get("name"))

    else:
        raise ValueError(f"Unknown tool: {name}")


# ── MCP server (SSE transport) ────────────────────────────────────────────────

mcp_server = Server("frida-mcp")


@mcp_server.list_tools()
async def _list_tools():
    return TOOLS


@mcp_server.call_tool()
async def _handle_call_tool(name: str, arguments: dict):
    try:
        result = call_tool(name, arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


# ── HTTP + SSE app ─────────────────────────────────────────────────────────────

sse_transport = SseServerTransport("/messages/")


async def handle_sse(request: Request):
    """SSE endpoint — AI agents connect here."""
    try:
        send = request._send  # noqa: SLF001 — private Starlette API
    except AttributeError:
        async def _fallback_send(message):
            raise RuntimeError("request._send not available; ASGI send interface changed")
        send = _fallback_send

    async with sse_transport.connect_sse(
        request.scope, request.receive, send,
    ) as streams:
        await mcp_server.run(
            streams[0], streams[1],
            mcp_server.create_initialization_options(),
            stateless=True,
        )
    from starlette.responses import Response as StarletteResponse
    return StarletteResponse()


async def health(request: Request) -> JSONResponse:
    """Health check endpoint."""
    android_device = os.getenv("ANDROID_DEVICE_ID", "")
    return JSONResponse({
        "status": "ok",
        "service": "frida-mcp",
        "tools": len(TOOLS),
        "android_device_id": android_device or None,
    })


app = Starlette(
    debug=False,
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/sse", handle_sse, methods=["GET"]),
        Mount("/messages", app=sse_transport.handle_post_message),
    ],
)


def main():
    """Entry point."""
    log.info("Starting Frida MCP server on %s:%d", FRIDA_HOST, FRIDA_PORT)

    # Auto-connect to remote Android device if configured
    android_device = os.getenv("ANDROID_DEVICE_ID", "")
    if android_device:
        log.info("Auto-connecting ADB to remote device: %s", android_device)
        try:
            result = adb.adb_connect_remote(android_device)
            log.info("ADB connect result: %s", result)
        except Exception as exc:
            log.warning("ADB auto-connect failed (device may not be ready): %s", exc)

    uvicorn.run(app, host=FRIDA_HOST, port=FRIDA_PORT, log_level="warning")


if __name__ == "__main__":
    main()

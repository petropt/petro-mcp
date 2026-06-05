"""petro-mcp: MCP server for petroleum engineering calculations."""

__version__ = "1.1.0"
__all__ = ["create_server"]


def __getattr__(name):
    if name == "create_server":
        from petro_mcp.server import create_server
        return create_server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

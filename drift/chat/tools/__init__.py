from drift.chat.tools.common import ToolResult
from drift.chat.tools.search import tool_search
from drift.chat.tools.duplicates import tool_duplicates
from drift.chat.tools.structure import tool_suggest_structure
from drift.chat.tools.moves import tool_preview_moves, tool_move
from drift.chat.tools.delete import tool_delete
from drift.chat.tools.undo import tool_undo_last_action


def tool_definitions() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "search_files",
                "description": "Semantic search over indexed text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_duplicates",
                "description": "Find duplicate files by exact content hash",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "suggest_structure",
                "description": "Suggest a folder structure based on clustered embeddings",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "min_cluster_size": {"type": "integer", "default": 3},
                        "min_samples": {"type": "integer", "default": 2},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "preview_moves",
                "description": "Preview file move/rename plan before execution",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "plan": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "src": {"type": "string"},
                                    "dest": {"type": "string"},
                                },
                                "required": ["src", "dest"],
                            },
                        }
                    },
                    "required": ["plan"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "move_files",
                "description": "Move specific files to a destination (uses the same preview/approval flow as preview_moves)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "plan": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "src": {"type": "string"},
                                    "dest": {"type": "string"},
                                },
                                "required": ["src", "dest"],
                            },
                        }
                    },
                    "required": ["plan"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_items",
                "description": "Delete specific files or folders (requires approval)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["items"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "undo_last_action",
                "description": "Undo the most recent applied move plan (single-level undo).",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


__all__ = [
    "ToolResult",
    "tool_definitions",
    "tool_duplicates",
    "tool_search",
    "tool_suggest_structure",
    "tool_preview_moves",
    "tool_move",
    "tool_delete",
    "tool_undo_last_action",
]

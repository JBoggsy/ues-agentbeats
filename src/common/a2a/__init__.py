"""A2A Protocol utilities for Green and Purple agents.

This module provides reusable A2A protocol helpers including:
- Agent card building utilities
- Server setup helpers
- Client communication helpers
- Message creation and parsing utilities
- Task state management
- Artifact creation helpers
"""

from src.common.a2a.agent_card import AgentCardBuilder, create_skill
from src.common.a2a.artifacts import (
    create_artifact,
    create_json_artifact,
    extract_json_from_artifact,
)
from src.common.a2a.client import A2AClientWrapper
from src.common.a2a.messages import (
    create_data_message,
    create_text_message,
    data_part,
    file_part,
    get_all_text_parts,
    get_data_content,
    get_text_content,
    text_part,
)
from src.common.a2a.server import A2AServer
from src.common.a2a.tasks import (
    create_artifact_update,
    create_status_update,
    is_completed,
    is_failed,
    is_input_required,
    is_terminal_state,
)

__all__ = [
    # Agent card
    "AgentCardBuilder",
    "create_skill",
    # Server
    "A2AServer",
    # Client
    "A2AClientWrapper",
    # Messages
    "create_text_message",
    "create_data_message",
    "text_part",
    "data_part",
    "file_part",
    "get_text_content",
    "get_data_content",
    "get_all_text_parts",
    # Tasks
    "is_terminal_state",
    "is_completed",
    "is_failed",
    "is_input_required",
    "create_status_update",
    "create_artifact_update",
    # Artifacts
    "create_artifact",
    "create_json_artifact",
    "extract_json_from_artifact",
]

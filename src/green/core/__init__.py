"""Core infrastructure for the Green agent.

This package contains foundational utilities shared across Green agent
components, including LLM configuration, action logging, and message collection.

Modules:
    llm_config: Factory for creating LLM instances (Phase 3.3)
    action_log: Action log builder for assessments (Phase 3.4)
    message_collector: New message collection from UES (Phase 3.5)
"""

from src.green.core.action_log import (
    ActionLogBuilder,
    ActionLogBuilderError,
    InvalidTurnNumberError,
    InvalidTurnStateError,
)
from src.green.core.llm_config import (
    LLMConfig,
    LLMFactory,
    LLMProvider,
    UnsupportedModelError,
)
from src.green.core.message_collector import (
    CollectorNotInitializedError,
    MessageCollectorError,
    NewMessageCollector,
    NewMessages,
)

__all__ = [
    # LLM Configuration (Phase 3.3)
    "LLMConfig",
    "LLMFactory",
    "LLMProvider",
    "UnsupportedModelError",
    # Action Log (Phase 3.4)
    "ActionLogBuilder",
    "ActionLogBuilderError",
    "InvalidTurnNumberError",
    "InvalidTurnStateError",
    # Message Collector (Phase 3.5)
    "NewMessageCollector",
    "NewMessages",
    "MessageCollectorError",
    "CollectorNotInitializedError",
]

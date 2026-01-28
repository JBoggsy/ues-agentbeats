"""Agent Card builder utilities for A2A protocol.

This module provides a fluent builder pattern for creating A2A-compliant
AgentCard objects, along with helper functions for creating AgentSkill objects.
"""

from __future__ import annotations

from typing import Self

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentProvider,
    AgentSkill,
)


def create_skill(
    id: str,
    name: str,
    description: str,
    tags: list[str] | None = None,
    examples: list[str] | None = None,
    input_modes: list[str] | None = None,
    output_modes: list[str] | None = None,
) -> AgentSkill:
    """Create an AgentSkill with sensible defaults.

    Args:
        id: Unique identifier for the skill.
        name: Human-readable name for the skill.
        description: Detailed explanation of what the skill does.
        tags: Keywords for categorization and discovery.
        examples: Sample prompts or use cases.
        input_modes: Supported input media types (defaults to ["text"]).
        output_modes: Supported output media types (defaults to ["text"]).

    Returns:
        An AgentSkill object ready to be added to an AgentCard.

    Example:
        >>> skill = create_skill(
        ...     id="email_triage",
        ...     name="Email Triage",
        ...     description="Triages and prioritizes incoming emails",
        ...     tags=["email", "productivity"],
        ...     examples=["Triage my unread emails", "Check for urgent messages"],
        ... )
    """
    return AgentSkill(
        id=id,
        name=name,
        description=description,
        tags=tags or [],
        examples=examples or [],
        inputModes=input_modes or ["text"],
        outputModes=output_modes or ["text"],
    )


class AgentCardBuilder:
    """Fluent builder for creating AgentCard objects.

    This class provides a convenient way to construct A2A AgentCard objects
    with sensible defaults and a fluent interface.

    Example:
        >>> card = (
        ...     AgentCardBuilder()
        ...     .with_name("My Agent")
        ...     .with_description("An example agent")
        ...     .with_url("http://localhost:8000")
        ...     .with_version("1.0.0")
        ...     .with_skill(create_skill("greet", "Greeting", "Says hello"))
        ...     .with_capabilities(streaming=True)
        ...     .build()
        ... )
    """

    def __init__(self) -> None:
        """Initialize the builder with default values."""
        self._name: str | None = None
        self._description: str | None = None
        self._url: str | None = None
        self._version: str = "1.0.0"
        self._skills: list[AgentSkill] = []
        self._capabilities: AgentCapabilities | None = None
        self._default_input_modes: list[str] = ["text"]
        self._default_output_modes: list[str] = ["text"]
        self._provider: AgentProvider | None = None
        self._documentation_url: str | None = None
        self._icon_url: str | None = None
        self._supports_authenticated_extended_card: bool = False

    def with_name(self, name: str) -> Self:
        """Set the agent name.

        Args:
            name: Human-readable name for the agent.

        Returns:
            Self for method chaining.
        """
        self._name = name
        return self

    def with_description(self, description: str) -> Self:
        """Set the agent description.

        Args:
            description: Detailed description of what the agent does.

        Returns:
            Self for method chaining.
        """
        self._description = description
        return self

    def with_url(self, url: str) -> Self:
        """Set the agent's service URL.

        Args:
            url: The endpoint where the A2A service can be reached.

        Returns:
            Self for method chaining.
        """
        self._url = url
        return self

    def with_version(self, version: str) -> Self:
        """Set the agent version.

        Args:
            version: Version string (e.g., "1.0.0").

        Returns:
            Self for method chaining.
        """
        self._version = version
        return self

    def with_skill(self, skill: AgentSkill) -> Self:
        """Add a skill to the agent.

        Args:
            skill: An AgentSkill object to add.

        Returns:
            Self for method chaining.
        """
        self._skills.append(skill)
        return self

    def with_skills(self, skills: list[AgentSkill]) -> Self:
        """Add multiple skills to the agent.

        Args:
            skills: List of AgentSkill objects to add.

        Returns:
            Self for method chaining.
        """
        self._skills.extend(skills)
        return self

    def with_capabilities(
        self,
        streaming: bool = False,
        push_notifications: bool = False,
        state_transition_history: bool = False,
    ) -> Self:
        """Set the agent capabilities.

        Args:
            streaming: Whether the agent supports streaming responses.
            push_notifications: Whether the agent supports push notifications.
            state_transition_history: Whether the agent tracks state history.

        Returns:
            Self for method chaining.
        """
        self._capabilities = AgentCapabilities(
            streaming=streaming,
            pushNotifications=push_notifications,
            stateTransitionHistory=state_transition_history,
        )
        return self

    def with_default_input_modes(self, modes: list[str]) -> Self:
        """Set default input media types.

        Args:
            modes: List of supported input media types.

        Returns:
            Self for method chaining.
        """
        self._default_input_modes = modes
        return self

    def with_default_output_modes(self, modes: list[str]) -> Self:
        """Set default output media types.

        Args:
            modes: List of supported output media types.

        Returns:
            Self for method chaining.
        """
        self._default_output_modes = modes
        return self

    def with_provider(self, organization: str, url: str) -> Self:
        """Set the agent provider information.

        Args:
            organization: Name of the organization providing the agent.
            url: URL for the organization.

        Returns:
            Self for method chaining.
        """
        self._provider = AgentProvider(organization=organization, url=url)
        return self

    def with_documentation_url(self, url: str) -> Self:
        """Set the documentation URL.

        Args:
            url: URL to the agent's documentation.

        Returns:
            Self for method chaining.
        """
        self._documentation_url = url
        return self

    def with_icon_url(self, url: str) -> Self:
        """Set the agent's icon URL.

        Args:
            url: URL to the agent's icon image.

        Returns:
            Self for method chaining.
        """
        self._icon_url = url
        return self

    def with_authenticated_extended_card(self, supported: bool = True) -> Self:
        """Set whether the agent supports authenticated extended cards.

        Args:
            supported: Whether authenticated extended cards are supported.

        Returns:
            Self for method chaining.
        """
        self._supports_authenticated_extended_card = supported
        return self

    def build(self) -> AgentCard:
        """Build the AgentCard.

        Returns:
            The constructed AgentCard object.

        Raises:
            ValueError: If required fields (name, description, url) are not set.
        """
        if not self._name:
            raise ValueError("Agent name is required")
        if not self._description:
            raise ValueError("Agent description is required")
        if not self._url:
            raise ValueError("Agent URL is required")

        # Provide default capabilities if none set
        capabilities = self._capabilities or AgentCapabilities()

        return AgentCard(
            name=self._name,
            description=self._description,
            url=self._url,
            version=self._version,
            skills=self._skills,
            capabilities=capabilities,
            defaultInputModes=self._default_input_modes,
            defaultOutputModes=self._default_output_modes,
            provider=self._provider,
            documentationUrl=self._documentation_url,
            iconUrl=self._icon_url,
            supportsAuthenticatedExtendedCard=self._supports_authenticated_extended_card,
        )

    def copy(self) -> AgentCardBuilder:
        """Create a copy of this builder with the same settings.

        Returns:
            A new AgentCardBuilder with copied settings.
        """
        builder = AgentCardBuilder()
        builder._name = self._name
        builder._description = self._description
        builder._url = self._url
        builder._version = self._version
        builder._skills = list(self._skills)
        builder._capabilities = self._capabilities
        builder._default_input_modes = list(self._default_input_modes)
        builder._default_output_modes = list(self._default_output_modes)
        builder._provider = self._provider
        builder._documentation_url = self._documentation_url
        builder._icon_url = self._icon_url
        builder._supports_authenticated_extended_card = (
            self._supports_authenticated_extended_card
        )
        return builder

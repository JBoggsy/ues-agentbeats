"""Tests for agent_card.py."""

import pytest
from a2a.types import AgentCard, AgentSkill

from src.common.a2a.agent_card import AgentCardBuilder, create_skill


class TestCreateSkill:
    """Tests for the create_skill function."""

    def test_create_skill_minimal(self):
        """Test creating a skill with minimal arguments."""
        skill = create_skill(
            id="test_skill",
            name="Test Skill",
            description="A test skill",
        )

        assert skill.id == "test_skill"
        assert skill.name == "Test Skill"
        assert skill.description == "A test skill"
        assert skill.tags == []
        assert skill.examples == []
        assert skill.input_modes == ["text"]
        assert skill.output_modes == ["text"]

    def test_create_skill_full(self):
        """Test creating a skill with all arguments."""
        skill = create_skill(
            id="email_triage",
            name="Email Triage",
            description="Triages and prioritizes incoming emails",
            tags=["email", "productivity"],
            examples=["Triage my unread emails", "Check for urgent messages"],
            input_modes=["text", "application/json"],
            output_modes=["text"],
        )

        assert skill.id == "email_triage"
        assert skill.name == "Email Triage"
        assert skill.description == "Triages and prioritizes incoming emails"
        assert skill.tags == ["email", "productivity"]
        assert skill.examples == ["Triage my unread emails", "Check for urgent messages"]
        assert skill.input_modes == ["text", "application/json"]
        assert skill.output_modes == ["text"]

    def test_create_skill_returns_agent_skill(self):
        """Test that create_skill returns an AgentSkill instance."""
        skill = create_skill(id="test", name="Test", description="Desc")
        assert isinstance(skill, AgentSkill)


class TestAgentCardBuilder:
    """Tests for the AgentCardBuilder class."""

    def test_build_minimal_card(self):
        """Test building a card with minimal required fields."""
        card = (
            AgentCardBuilder()
            .with_name("Test Agent")
            .with_description("A test agent")
            .with_url("http://localhost:8000")
            .build()
        )

        assert card.name == "Test Agent"
        assert card.description == "A test agent"
        assert card.url == "http://localhost:8000"
        assert card.version == "1.0.0"  # default
        assert card.skills == []
        assert card.default_input_modes == ["text"]
        assert card.default_output_modes == ["text"]

    def test_build_full_card(self):
        """Test building a card with all fields."""
        skill = create_skill(id="greet", name="Greeting", description="Says hello")

        card = (
            AgentCardBuilder()
            .with_name("Full Agent")
            .with_description("A fully configured agent")
            .with_url("http://localhost:9000")
            .with_version("2.0.0")
            .with_skill(skill)
            .with_capabilities(streaming=True, push_notifications=True)
            .with_default_input_modes(["text", "application/json"])
            .with_default_output_modes(["text"])
            .with_provider("Test Org", "https://test.org")
            .with_documentation_url("https://docs.test.org")
            .with_icon_url("https://test.org/icon.png")
            .with_authenticated_extended_card(True)
            .build()
        )

        assert card.name == "Full Agent"
        assert card.description == "A fully configured agent"
        assert card.url == "http://localhost:9000"
        assert card.version == "2.0.0"
        assert len(card.skills) == 1
        assert card.skills[0].id == "greet"
        assert card.capabilities.streaming is True
        assert card.capabilities.push_notifications is True
        assert card.default_input_modes == ["text", "application/json"]
        assert card.default_output_modes == ["text"]
        assert card.provider.organization == "Test Org"
        assert card.provider.url == "https://test.org"
        assert card.documentation_url == "https://docs.test.org"
        assert card.icon_url == "https://test.org/icon.png"
        assert card.supports_authenticated_extended_card is True

    def test_build_with_multiple_skills(self):
        """Test adding multiple skills to a card."""
        skill1 = create_skill(id="skill1", name="Skill 1", description="First skill")
        skill2 = create_skill(id="skill2", name="Skill 2", description="Second skill")
        skill3 = create_skill(id="skill3", name="Skill 3", description="Third skill")

        card = (
            AgentCardBuilder()
            .with_name("Multi-Skill Agent")
            .with_description("Agent with multiple skills")
            .with_url("http://localhost:8000")
            .with_skill(skill1)
            .with_skills([skill2, skill3])
            .build()
        )

        assert len(card.skills) == 3
        assert card.skills[0].id == "skill1"
        assert card.skills[1].id == "skill2"
        assert card.skills[2].id == "skill3"

    def test_build_missing_name_raises(self):
        """Test that building without a name raises ValueError."""
        with pytest.raises(ValueError, match="Agent name is required"):
            AgentCardBuilder().with_description("Desc").with_url("http://x").build()

    def test_build_missing_description_raises(self):
        """Test that building without a description raises ValueError."""
        with pytest.raises(ValueError, match="Agent description is required"):
            AgentCardBuilder().with_name("Name").with_url("http://x").build()

    def test_build_missing_url_raises(self):
        """Test that building without a URL raises ValueError."""
        with pytest.raises(ValueError, match="Agent URL is required"):
            AgentCardBuilder().with_name("Name").with_description("Desc").build()

    def test_copy_creates_independent_builder(self):
        """Test that copy creates an independent builder."""
        original = (
            AgentCardBuilder()
            .with_name("Original")
            .with_description("Original desc")
            .with_url("http://original")
        )

        copied = original.copy()
        copied.with_name("Copied").with_url("http://copied")

        original_card = original.build()
        copied_card = copied.build()

        assert original_card.name == "Original"
        assert original_card.url == "http://original"
        assert copied_card.name == "Copied"
        assert copied_card.url == "http://copied"

    def test_builder_returns_self(self):
        """Test that builder methods return self for chaining."""
        builder = AgentCardBuilder()

        assert builder.with_name("Name") is builder
        assert builder.with_description("Desc") is builder
        assert builder.with_url("http://x") is builder
        assert builder.with_version("1.0.0") is builder
        assert builder.with_capabilities() is builder
        assert builder.with_default_input_modes([]) is builder
        assert builder.with_default_output_modes([]) is builder

    def test_build_returns_agent_card(self):
        """Test that build returns an AgentCard instance."""
        card = (
            AgentCardBuilder()
            .with_name("Test")
            .with_description("Test")
            .with_url("http://test")
            .build()
        )
        assert isinstance(card, AgentCard)

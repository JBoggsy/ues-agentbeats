"""Response generator for creating character responses during assessments.

This module provides the ResponseGenerator class which creates in-character
responses from simulated contacts (characters) during assessments. It processes
new messages from the NewMessageCollector and generates appropriate responses
using LLM-based decision making.

The ResponseGenerator transforms the UES environment from a static simulation
into a dynamic, interactive world where characters respond realistically to
the Purple agent's actions.

Classes:
    ResponseGenerator: Generates character responses to new messages.
    ResponseGeneratorError: Base exception for response generator errors.
    ResponseGenerationError: Error during response content generation.

Example:
    >>> from src.green.response_generator import ResponseGenerator
    >>> from src.green.llm_config import LLMFactory
    >>>
    >>> llm = LLMFactory.create("gpt-4o-mini")
    >>> generator = ResponseGenerator(
    ...     client=ues_client,
    ...     scenario_config=scenario,
    ...     response_llm=llm,
    ...     summarization_llm=llm,
    ... )
    >>>
    >>> responses = await generator.process_new_messages(
    ...     new_messages=messages,
    ...     current_time=datetime.now(timezone.utc),
    ... )
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.common.agentbeats.updates import TaskUpdateEmitter
from src.green.message_collector import NewMessages
from src.green.prompts.response_prompts import (
    CALENDAR_RSVP_SYSTEM_PROMPT,
    CALENDAR_RSVP_USER_PROMPT,
    GENERATE_RESPONSE_SYSTEM_PROMPT,
    GENERATE_RESPONSE_USER_PROMPT,
    SHOULD_RESPOND_SYSTEM_PROMPT,
    SHOULD_RESPOND_USER_PROMPT,
    SUMMARIZE_THREAD_PROMPT,
    build_config_section,
    build_special_instructions_section,
    build_thread_summary_section,
    format_participant_list,
)
from src.green.response_models import (
    CalendarEventContext,
    CalendarRSVPResult,
    MessageContext,
    ScheduledResponse,
    ShouldRespondResult,
    ThreadContext,
)
from src.green.scenarios.schema import CharacterProfile, ScenarioConfig


if TYPE_CHECKING:
    from ues.client import AsyncUESClient, CalendarEvent, Email, SMSMessage


logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Maximum number of messages to include in thread context before summarizing
MAX_THREAD_MESSAGES = 10


# =============================================================================
# Exceptions
# =============================================================================


class ResponseGeneratorError(Exception):
    """Base exception for response generator errors.

    Attributes:
        message: Human-readable error description.
    """

    pass


class ResponseGenerationError(ResponseGeneratorError):
    """Error during response content generation.

    Raised when LLM-based response generation fails.

    Attributes:
        message: Human-readable error description.
        character_name: Name of the character that failed to generate response.
        modality: The modality of the message being responded to.
    """

    def __init__(
        self,
        message: str,
        character_name: str | None = None,
        modality: str | None = None,
    ) -> None:
        """Initialize the error.

        Args:
            message: Human-readable error description.
            character_name: Name of the character that failed.
            modality: The modality of the failed response.
        """
        super().__init__(message)
        self.character_name = character_name
        self.modality = modality


# =============================================================================
# ResponseGenerator Class
# =============================================================================


class ResponseGenerator:
    """Generates character responses to new messages during assessment.

    This class processes new messages (emails, SMS, calendar events) and
    generates in-character responses from simulated contacts. It uses LLMs
    for decision-making (should a character respond?) and content generation
    (what should they say?).

    The generator:
    1. Processes new messages from NewMessageCollector
    2. Identifies which characters are involved
    3. Determines if each character should respond (LLM decision)
    4. Generates response content if they should respond (LLM generation)
    5. Calculates response timing based on character configuration
    6. Returns ScheduledResponse objects for GreenAgent to inject into UES

    Attributes:
        scenario_config: The scenario configuration with character profiles.
        user_email: Email address of the user being assisted.
        user_phone: Phone number of the user being assisted (if available).

    Example:
        >>> generator = ResponseGenerator(
        ...     client=ues_client,
        ...     scenario_config=scenario,
        ...     response_llm=response_model,
        ...     summarization_llm=summarization_model,
        ... )
        >>>
        >>> responses = await generator.process_new_messages(
        ...     new_messages=new_messages,
        ...     current_time=current_sim_time,
        ... )
        >>> for response in responses:
        ...     print(f"{response.character_name} will respond at {response.scheduled_time}")
    """

    def __init__(
        self,
        client: AsyncUESClient,
        scenario_config: ScenarioConfig,
        response_llm: BaseChatModel,
        summarization_llm: BaseChatModel,
    ) -> None:
        """Initialize the response generator.

        Args:
            client: Async UES client for querying thread history.
            scenario_config: Scenario configuration with character profiles.
            response_llm: LLM for should-respond checks and response generation.
            summarization_llm: LLM for summarizing long thread histories
                (can be a cheaper/faster model).
        """
        self._client = client
        self._scenario_config = scenario_config
        self._response_llm = response_llm
        self._summarization_llm = summarization_llm

        # Cache user character info for quick lookups
        user_character = scenario_config.get_user_character_profile()
        self._user_email = user_character.email
        self._user_phone = user_character.phone

        # Build lookup tables for characters by contact method
        self._characters_by_email: dict[str, CharacterProfile] = {}
        self._characters_by_phone: dict[str, CharacterProfile] = {}
        for character in scenario_config.characters.values():
            if character.email:
                self._characters_by_email[character.email.lower()] = character
            if character.phone:
                self._characters_by_phone[character.phone] = character

    @property
    def scenario_config(self) -> ScenarioConfig:
        """Return the scenario configuration."""
        return self._scenario_config

    @property
    def user_email(self) -> str | None:
        """Return the user's email address."""
        return self._user_email

    @property
    def user_phone(self) -> str | None:
        """Return the user's phone number."""
        return self._user_phone

    # =========================================================================
    # Main Public Method
    # =========================================================================

    async def process_new_messages(
        self,
        new_messages: NewMessages,
        current_time: datetime,
        task_updater: TaskUpdateEmitter | None = None,
    ) -> list[ScheduledResponse]:
        """Process new messages and generate character responses.

        This is the main entry point for response generation. It processes
        all new emails, SMS messages, and calendar events, determining which
        characters should respond and generating appropriate responses.

        Args:
            new_messages: Container with new emails, SMS, and calendar events.
            current_time: Current simulation time (used for scheduling responses).
            task_updater: Optional task updater for emitting progress updates.

        Returns:
            List of ScheduledResponse objects to be injected into UES.
            May be empty if no responses are needed.

        Note:
            This method handles errors gracefully. If generation fails for
            a specific message or character, it logs a warning and continues
            processing other messages. The returned list will contain only
            the responses that were successfully generated.
        """
        responses: list[ScheduledResponse] = []

        # Process emails
        for email in new_messages.emails:
            try:
                email_responses = await self._process_email(
                    email, current_time, task_updater
                )
                responses.extend(email_responses)
            except Exception as e:
                logger.warning(
                    f"Failed to process email {email.message_id}: {e}",
                    exc_info=True,
                )
                if task_updater:
                    await self._emit_warning(
                        task_updater,
                        f"Failed to process email: {e}",
                        "email_processing_error",
                        {"message_id": email.message_id},
                    )

        # Process SMS messages
        for sms in new_messages.sms_messages:
            try:
                sms_responses = await self._process_sms(sms, current_time, task_updater)
                responses.extend(sms_responses)
            except Exception as e:
                logger.warning(
                    f"Failed to process SMS {sms.message_id}: {e}",
                    exc_info=True,
                )
                if task_updater:
                    await self._emit_warning(
                        task_updater,
                        f"Failed to process SMS: {e}",
                        "sms_processing_error",
                        {"message_id": sms.message_id},
                    )

        # Process calendar events
        for event in new_messages.calendar_events:
            try:
                calendar_responses = await self._process_calendar_event(
                    event, current_time, task_updater
                )
                responses.extend(calendar_responses)
            except Exception as e:
                logger.warning(
                    f"Failed to process calendar event {event.event_id}: {e}",
                    exc_info=True,
                )
                if task_updater:
                    await self._emit_warning(
                        task_updater,
                        f"Failed to process calendar event: {e}",
                        "calendar_processing_error",
                        {"event_id": event.event_id},
                    )

        logger.info(
            f"Generated {len(responses)} responses from {new_messages.total_count} new messages"
        )
        return responses

    # =========================================================================
    # Message Processing Methods
    # =========================================================================

    async def _process_email(
        self,
        email: Email,
        current_time: datetime,
        task_updater: TaskUpdateEmitter | None,
    ) -> list[ScheduledResponse]:
        """Process a single email and generate responses from characters.

        Args:
            email: The email to process.
            current_time: Current simulation time.
            task_updater: Optional task updater for progress updates.

        Returns:
            List of responses from characters who should respond.
        """
        responses: list[ScheduledResponse] = []

        # Build message context
        sender_address = email.from_address
        all_recipients = set(email.to_addresses)
        if hasattr(email, "cc_addresses") and email.cc_addresses:
            all_recipients.update(email.cc_addresses)

        # Get thread history
        thread_context = await self._prepare_email_thread_context(
            email, task_updater
        )

        message_context = MessageContext(
            message=email,
            modality="email",
            sender_address=sender_address,
            all_recipients=all_recipients,
            thread_context=thread_context,
        )

        # Find characters who might respond (recipients, excluding sender and user)
        potential_responders = self._find_email_responders(
            sender_address, all_recipients
        )

        # For each potential responder, check if they should respond
        for character in potential_responders:
            try:
                response = await self._generate_email_response_if_needed(
                    character, message_context, current_time, task_updater
                )
                if response:
                    responses.append(response)
            except Exception as e:
                logger.warning(
                    f"Failed to generate response from {character.name}: {e}",
                    exc_info=True,
                )

        return responses

    async def _process_sms(
        self,
        sms: SMSMessage,
        current_time: datetime,
        task_updater: TaskUpdateEmitter | None,
    ) -> list[ScheduledResponse]:
        """Process a single SMS message and generate responses from characters.

        Args:
            sms: The SMS message to process.
            current_time: Current simulation time.
            task_updater: Optional task updater for progress updates.

        Returns:
            List of responses from characters who should respond.
        """
        responses: list[ScheduledResponse] = []

        # Build message context
        sender_phone = sms.from_number
        all_recipients = set(sms.to_numbers)

        # Get thread history
        thread_context = await self._prepare_sms_thread_context(sms, task_updater)

        message_context = MessageContext(
            message=sms,
            modality="sms",
            sender_address=sender_phone,
            all_recipients=all_recipients,
            thread_context=thread_context,
        )

        # Find characters who might respond
        potential_responders = self._find_sms_responders(sender_phone, all_recipients)

        # For each potential responder, check if they should respond
        for character in potential_responders:
            try:
                response = await self._generate_sms_response_if_needed(
                    character, message_context, current_time, task_updater
                )
                if response:
                    responses.append(response)
            except Exception as e:
                logger.warning(
                    f"Failed to generate SMS response from {character.name}: {e}",
                    exc_info=True,
                )

        return responses

    async def _process_calendar_event(
        self,
        event: CalendarEvent,
        current_time: datetime,
        task_updater: TaskUpdateEmitter | None,
    ) -> list[ScheduledResponse]:
        """Process a calendar event and generate RSVP responses from attendees.

        Args:
            event: The calendar event to process.
            current_time: Current simulation time.
            task_updater: Optional task updater for progress updates.

        Returns:
            List of RSVP responses from character attendees.
        """
        responses: list[ScheduledResponse] = []

        # Get organizer and attendees
        organizer_email = event.organizer or ""
        all_attendees = {a.email for a in event.attendees if hasattr(a, "email")}

        event_context = CalendarEventContext(
            event=event,
            organizer_email=organizer_email,
            all_attendees=all_attendees,
        )

        # Find characters who are attendees (excluding organizer)
        potential_responders = self._find_calendar_responders(
            organizer_email, all_attendees
        )

        # For each character attendee, generate RSVP
        for character in potential_responders:
            try:
                response = await self._generate_calendar_rsvp_if_needed(
                    character, event_context, current_time, task_updater
                )
                if response:
                    responses.append(response)
            except Exception as e:
                logger.warning(
                    f"Failed to generate calendar RSVP from {character.name}: {e}",
                    exc_info=True,
                )

        return responses

    # =========================================================================
    # Character Lookup Methods
    # =========================================================================

    def _get_character_by_email(self, email: str) -> CharacterProfile | None:
        """Find a character by their email address.

        Args:
            email: Email address to look up (case-insensitive).

        Returns:
            CharacterProfile if found, None otherwise.
        """
        return self._characters_by_email.get(email.lower())

    def _get_character_by_phone(self, phone: str) -> CharacterProfile | None:
        """Find a character by their phone number.

        Args:
            phone: Phone number to look up.

        Returns:
            CharacterProfile if found, None otherwise.
        """
        return self._characters_by_phone.get(phone)

    def _is_user_email(self, email: str) -> bool:
        """Check if an email address belongs to the user being assisted.

        Args:
            email: Email address to check.

        Returns:
            True if this is the user's email, False otherwise.
        """
        if not self._user_email:
            return False
        return email.lower() == self._user_email.lower()

    def _is_user_phone(self, phone: str) -> bool:
        """Check if a phone number belongs to the user being assisted.

        Args:
            phone: Phone number to check.

        Returns:
            True if this is the user's phone, False otherwise.
        """
        if not self._user_phone:
            return False
        return phone == self._user_phone

    def _find_email_responders(
        self,
        sender_address: str,
        all_recipients: set[str],
    ) -> list[CharacterProfile]:
        """Find characters who might respond to an email.

        Returns characters who:
        - Are in the recipient list (to or cc)
        - Are not the sender
        - Are not the user being assisted
        - Don't have response generation disabled

        Args:
            sender_address: Email address of the sender.
            all_recipients: Set of all recipient email addresses.

        Returns:
            List of CharacterProfile objects for potential responders.
        """
        responders = []
        for email_addr in all_recipients:
            # Skip sender
            if email_addr.lower() == sender_address.lower():
                continue
            # Skip user
            if self._is_user_email(email_addr):
                continue
            # Look up character
            character = self._get_character_by_email(email_addr)
            if character and not self._should_skip_character(character):
                responders.append(character)
        return responders

    def _find_sms_responders(
        self,
        sender_phone: str,
        all_recipients: set[str],
    ) -> list[CharacterProfile]:
        """Find characters who might respond to an SMS.

        Returns characters who:
        - Are in the recipient list
        - Are not the sender
        - Are not the user being assisted
        - Don't have response generation disabled

        Args:
            sender_phone: Phone number of the sender.
            all_recipients: Set of all recipient phone numbers.

        Returns:
            List of CharacterProfile objects for potential responders.
        """
        responders = []
        for phone in all_recipients:
            # Skip sender
            if phone == sender_phone:
                continue
            # Skip user
            if self._is_user_phone(phone):
                continue
            # Look up character
            character = self._get_character_by_phone(phone)
            if character and not self._should_skip_character(character):
                responders.append(character)
        return responders

    def _find_calendar_responders(
        self,
        organizer_email: str,
        all_attendees: set[str],
    ) -> list[CharacterProfile]:
        """Find characters who might respond to a calendar invitation.

        Returns characters who:
        - Are in the attendee list
        - Are not the organizer
        - Are not the user being assisted
        - Don't have response generation disabled

        Args:
            organizer_email: Email address of the event organizer.
            all_attendees: Set of all attendee email addresses.

        Returns:
            List of CharacterProfile objects for potential responders.
        """
        responders = []
        for email_addr in all_attendees:
            # Skip organizer
            if email_addr.lower() == organizer_email.lower():
                continue
            # Skip user
            if self._is_user_email(email_addr):
                continue
            # Look up character
            character = self._get_character_by_email(email_addr)
            if character and not self._should_skip_character(character):
                responders.append(character)
        return responders

    def _should_skip_character(self, character: CharacterProfile) -> bool:
        """Check if a character should be skipped for response generation.

        Characters can be skipped if their special_instructions indicate
        they should not respond (e.g., "never responds", "unavailable").

        Args:
            character: The character to check.

        Returns:
            True if the character should be skipped, False otherwise.
        """
        if not character.special_instructions:
            return False

        # Check for keywords that indicate no response
        skip_keywords = [
            "never respond",
            "do not respond",
            "don't respond",
            "unavailable",
            "out of office",
            "on vacation",
            "no responses",
        ]
        instructions_lower = character.special_instructions.lower()
        return any(keyword in instructions_lower for keyword in skip_keywords)

    # =========================================================================
    # Response Generation Methods
    # =========================================================================

    async def _generate_email_response_if_needed(
        self,
        character: CharacterProfile,
        context: MessageContext,
        current_time: datetime,
        task_updater: TaskUpdateEmitter | None,
    ) -> ScheduledResponse | None:
        """Generate an email response if the character should respond.

        Args:
            character: The character who might respond.
            context: Message context with email and thread history.
            current_time: Current simulation time.
            task_updater: Optional task updater for progress updates.

        Returns:
            ScheduledResponse if the character should respond, None otherwise.
        """
        email: Email = context.message  # type: ignore

        # Check if character should respond
        should_respond = await self._should_respond(
            character, context, task_updater
        )
        if not should_respond:
            logger.debug(f"{character.name} decided not to respond to email")
            return None

        # Generate response content
        content = await self._generate_response(character, context, task_updater)

        # Calculate response time
        scheduled_time = self._calculate_response_time(character, current_time)

        # Build recipients: reply to sender, preserve other recipients
        recipients = [email.from_address]
        cc_recipients = []
        for addr in context.all_recipients:
            if addr.lower() != character.email.lower():  # type: ignore
                if addr not in recipients:
                    cc_recipients.append(addr)

        # Build the response
        return ScheduledResponse(
            modality="email",
            character_name=character.name,
            character_email=character.email,
            scheduled_time=scheduled_time,
            content=content,
            original_message_id=email.message_id,
            thread_id=email.thread_id,
            subject=self._derive_email_subject(email.subject),
            recipients=recipients,
            cc_recipients=cc_recipients,
            in_reply_to=email.message_id,
            references=self._build_email_references(email),
        )

    async def _generate_sms_response_if_needed(
        self,
        character: CharacterProfile,
        context: MessageContext,
        current_time: datetime,
        task_updater: TaskUpdateEmitter | None,
    ) -> ScheduledResponse | None:
        """Generate an SMS response if the character should respond.

        Args:
            character: The character who might respond.
            context: Message context with SMS and thread history.
            current_time: Current simulation time.
            task_updater: Optional task updater for progress updates.

        Returns:
            ScheduledResponse if the character should respond, None otherwise.
        """
        sms: SMSMessage = context.message  # type: ignore

        # Check if character should respond
        should_respond = await self._should_respond(
            character, context, task_updater
        )
        if not should_respond:
            logger.debug(f"{character.name} decided not to respond to SMS")
            return None

        # Generate response content
        content = await self._generate_response(character, context, task_updater)

        # Calculate response time
        scheduled_time = self._calculate_response_time(character, current_time)

        # Build recipients: reply to sender
        recipients = [sms.from_number]

        # Build the response
        return ScheduledResponse(
            modality="sms",
            character_name=character.name,
            character_phone=character.phone,
            scheduled_time=scheduled_time,
            content=content,
            original_message_id=sms.message_id,
            thread_id=sms.thread_id,
            recipients=recipients,
        )

    async def _generate_calendar_rsvp_if_needed(
        self,
        character: CharacterProfile,
        context: CalendarEventContext,
        current_time: datetime,
        task_updater: TaskUpdateEmitter | None,
    ) -> ScheduledResponse | None:
        """Generate a calendar RSVP if the character should respond.

        Args:
            character: The character who might respond.
            context: Calendar event context.
            current_time: Current simulation time.
            task_updater: Optional task updater for progress updates.

        Returns:
            ScheduledResponse with RSVP if needed, None otherwise.
        """
        event = context.event

        # Decide RSVP status
        rsvp_result = await self._decide_calendar_rsvp(
            character, context, task_updater
        )

        # Calculate response time
        scheduled_time = self._calculate_response_time(character, current_time)

        # Build the response
        return ScheduledResponse(
            modality="calendar",
            character_name=character.name,
            character_email=character.email,
            scheduled_time=scheduled_time,
            event_id=event.event_id,
            rsvp_status=rsvp_result.status,
            rsvp_comment=rsvp_result.comment,
        )

    # =========================================================================
    # LLM Integration Methods
    # =========================================================================

    async def _should_respond(
        self,
        character: CharacterProfile,
        context: MessageContext,
        task_updater: TaskUpdateEmitter | None,
    ) -> bool:
        """Use LLM to decide if a character should respond.

        Args:
            character: The character making the decision.
            context: Message context with message and thread history.
            task_updater: Optional task updater for progress updates.

        Returns:
            True if the character should respond, False otherwise.
        """
        # Build prompts
        system_prompt = SHOULD_RESPOND_SYSTEM_PROMPT.format(
            modality=context.modality,
        )

        # Format the latest message
        formatted_message = self._format_message_for_prompt(
            context.message, context.modality
        )

        # Get sender name (try to find character, fall back to address)
        sender_name = self._get_sender_name(context)

        # Build thread history section
        thread_summary_section = ""
        formatted_thread_history = "(No previous messages in thread)"
        if context.thread_context:
            thread_summary_section = build_thread_summary_section(
                context.thread_context.summary
            )
            formatted_thread_history = (
                context.thread_context.formatted_history
                or "(No previous messages in thread)"
            )

        user_prompt = SHOULD_RESPOND_USER_PROMPT.format(
            character_name=character.name,
            character_personality=character.personality,
            special_instructions_section=build_special_instructions_section(
                character.special_instructions
            ),
            config_section=build_config_section(character.config),
            modality=context.modality,
            thread_summary_section=thread_summary_section,
            formatted_thread_history=formatted_thread_history,
            sender_name=sender_name,
            formatted_latest_message=formatted_message,
        )

        # Call LLM with structured output
        try:
            structured_llm = self._response_llm.with_structured_output(
                ShouldRespondResult
            )
            result: ShouldRespondResult = await structured_llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            logger.debug(
                f"Should-respond for {character.name}: {result.should_respond} "
                f"({result.reasoning})"
            )
            return result.should_respond
        except Exception as e:
            logger.warning(
                f"LLM should-respond check failed for {character.name}: {e}"
            )
            # Default to not responding on error
            return False

    async def _generate_response(
        self,
        character: CharacterProfile,
        context: MessageContext,
        task_updater: TaskUpdateEmitter | None,
    ) -> str:
        """Use LLM to generate in-character response content.

        Args:
            character: The character generating the response.
            context: Message context with message and thread history.
            task_updater: Optional task updater for progress updates.

        Returns:
            The generated response content.

        Raises:
            ResponseGenerationError: If LLM generation fails.
        """
        # Build prompts
        system_prompt = GENERATE_RESPONSE_SYSTEM_PROMPT.format(
            character_name=character.name,
            character_personality=character.personality,
            special_instructions_section=build_special_instructions_section(
                character.special_instructions
            ),
            config_section=build_config_section(character.config),
            modality=context.modality,
        )

        # Format the latest message
        formatted_message = self._format_message_for_prompt(
            context.message, context.modality
        )

        # Get sender and participant names
        sender_name = self._get_sender_name(context)
        participant_names = self._get_participant_names(context, character)

        # Build thread history section
        thread_summary_section = ""
        formatted_thread_history = "(No previous messages in thread)"
        if context.thread_context:
            thread_summary_section = build_thread_summary_section(
                context.thread_context.summary
            )
            formatted_thread_history = (
                context.thread_context.formatted_history
                or "(No previous messages in thread)"
            )

        user_prompt = GENERATE_RESPONSE_USER_PROMPT.format(
            modality=context.modality,
            participant_names=format_participant_list(participant_names),
            thread_summary_section=thread_summary_section,
            formatted_thread_history=formatted_thread_history,
            sender_name=sender_name,
            formatted_latest_message=formatted_message,
            character_name=character.name,
        )

        # Call LLM
        try:
            result = await self._response_llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            content = result.content
            if isinstance(content, str):
                return content.strip()
            # Handle case where content might be a list (shouldn't happen normally)
            return str(content).strip()
        except Exception as e:
            raise ResponseGenerationError(
                f"Failed to generate response: {e}",
                character_name=character.name,
                modality=context.modality,
            ) from e

    async def _decide_calendar_rsvp(
        self,
        character: CharacterProfile,
        context: CalendarEventContext,
        task_updater: TaskUpdateEmitter | None,
    ) -> CalendarRSVPResult:
        """Use LLM to decide calendar RSVP status.

        Args:
            character: The character making the RSVP decision.
            context: Calendar event context.
            task_updater: Optional task updater for progress updates.

        Returns:
            CalendarRSVPResult with status, optional comment, and reasoning.
        """
        event = context.event

        # Build prompts
        system_prompt = CALENDAR_RSVP_SYSTEM_PROMPT.format(
            character_name=character.name,
            character_personality=character.personality,
            special_instructions_section=build_special_instructions_section(
                character.special_instructions
            ),
            config_section=build_config_section(character.config),
        )

        # Format attendee list
        attendee_list = ", ".join(context.all_attendees) or "(no other attendees)"

        user_prompt = CALENDAR_RSVP_USER_PROMPT.format(
            event_title=event.title,
            organizer=context.organizer_email or "(unknown)",
            start_time=event.start.isoformat() if event.start else "(unknown)",
            end_time=event.end.isoformat() if event.end else "(unknown)",
            location=event.location or "(no location)",
            description=event.description or "(no description)",
            attendee_list=attendee_list,
            character_name=character.name,
        )

        # Call LLM with structured output
        try:
            structured_llm = self._response_llm.with_structured_output(
                CalendarRSVPResult
            )
            result: CalendarRSVPResult = await structured_llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            logger.debug(
                f"Calendar RSVP for {character.name}: {result.status} "
                f"({result.reasoning})"
            )
            return result
        except Exception as e:
            logger.warning(
                f"LLM calendar RSVP failed for {character.name}: {e}"
            )
            # Default to tentative on error
            return CalendarRSVPResult(
                status="tentative",
                comment=None,
                reasoning=f"Default response due to error: {e}",
            )

    # =========================================================================
    # Thread History Methods
    # =========================================================================

    async def _prepare_email_thread_context(
        self,
        email: Email,
        task_updater: TaskUpdateEmitter | None,
    ) -> ThreadContext | None:
        """Prepare thread context for an email.

        Retrieves the email thread history and formats it for LLM prompts.
        If the thread has many messages, older ones are summarized.

        Args:
            email: The email to get thread context for.
            task_updater: Optional task updater for progress updates.

        Returns:
            ThreadContext with formatted history, or None if retrieval fails.
        """
        if not email.thread_id:
            return None

        try:
            # Query for all emails in this thread
            result = await self._client.email.query(
                thread_id=email.thread_id,
                sort_order="asc",  # Oldest first
            )
            messages = list(result.emails)

            if not messages:
                return None

            return await self._prepare_thread_context(
                messages, "email", task_updater
            )
        except Exception as e:
            logger.warning(f"Failed to retrieve email thread history: {e}")
            return None

    async def _prepare_sms_thread_context(
        self,
        sms: SMSMessage,
        task_updater: TaskUpdateEmitter | None,
    ) -> ThreadContext | None:
        """Prepare thread context for an SMS message.

        Retrieves the SMS thread history and formats it for LLM prompts.
        If the thread has many messages, older ones are summarized.

        Args:
            sms: The SMS to get thread context for.
            task_updater: Optional task updater for progress updates.

        Returns:
            ThreadContext with formatted history, or None if retrieval fails.
        """
        if not sms.thread_id:
            return None

        try:
            # Query for all messages in this thread
            result = await self._client.sms.query(
                thread_id=sms.thread_id,
                sort_order="asc",  # Oldest first
            )
            messages = list(result.messages)

            if not messages:
                return None

            return await self._prepare_thread_context(
                messages, "sms", task_updater
            )
        except Exception as e:
            logger.warning(f"Failed to retrieve SMS thread history: {e}")
            return None

    async def _prepare_thread_context(
        self,
        messages: list[Email] | list[SMSMessage],
        modality: Literal["email", "sms"],
        task_updater: TaskUpdateEmitter | None,
    ) -> ThreadContext:
        """Prepare thread context from a list of messages.

        If the thread has more than MAX_THREAD_MESSAGES, older messages
        are summarized and only the most recent ones are included in full.

        Args:
            messages: List of messages in chronological order.
            modality: The modality type ("email" or "sms").
            task_updater: Optional task updater for progress updates.

        Returns:
            ThreadContext with formatted history and optional summary.
        """
        total_count = len(messages)

        if total_count <= MAX_THREAD_MESSAGES:
            # Include all messages
            formatted = self._format_messages_for_prompt(messages, modality)
            return ThreadContext(
                messages=messages,  # type: ignore
                formatted_history=formatted,
                total_message_count=total_count,
                included_message_count=total_count,
                summary=None,
            )

        # Summarize older messages and include recent ones
        older_messages = messages[:-MAX_THREAD_MESSAGES]
        recent_messages = messages[-MAX_THREAD_MESSAGES:]

        # Summarize older messages
        summary = await self._summarize_messages(older_messages, modality)

        # Format recent messages
        formatted = self._format_messages_for_prompt(recent_messages, modality)

        return ThreadContext(
            messages=recent_messages,  # type: ignore
            formatted_history=formatted,
            total_message_count=total_count,
            included_message_count=len(recent_messages),
            summary=summary,
        )

    async def _summarize_messages(
        self,
        messages: list[Email] | list[SMSMessage],
        modality: Literal["email", "sms"],
    ) -> str:
        """Summarize a list of messages using the summarization LLM.

        Args:
            messages: List of messages to summarize.
            modality: The modality type ("email" or "sms").

        Returns:
            A 2-3 sentence summary of the messages.
        """
        formatted = self._format_messages_for_prompt(messages, modality)

        prompt = SUMMARIZE_THREAD_PROMPT.format(
            modality=modality,
            formatted_messages=formatted,
        )

        try:
            result = await self._summarization_llm.ainvoke(
                [HumanMessage(content=prompt)]
            )
            content = result.content
            if isinstance(content, str):
                return content.strip()
            return str(content).strip()
        except Exception as e:
            logger.warning(f"Failed to summarize messages: {e}")
            return f"(Earlier conversation with {len(messages)} messages)"

    # =========================================================================
    # Message Formatting Methods
    # =========================================================================

    def _format_message_for_prompt(
        self,
        message: Email | SMSMessage,
        modality: Literal["email", "sms"],
    ) -> str:
        """Format a single message for inclusion in a prompt.

        Args:
            message: The message to format.
            modality: The modality type.

        Returns:
            Formatted string representation of the message.
        """
        if modality == "email":
            return self._format_email_for_prompt(message)  # type: ignore
        else:
            return self._format_sms_for_prompt(message)  # type: ignore

    def _format_email_for_prompt(self, email: Email) -> str:
        """Format an email for inclusion in a prompt.

        Args:
            email: The email to format.

        Returns:
            Formatted string with subject and body.
        """
        timestamp = ""
        if hasattr(email, "received_at") and email.received_at:
            timestamp = f"[{email.received_at.strftime('%Y-%m-%d %H:%M')}] "

        sender_name = self._get_name_for_email(email.from_address)
        lines = [
            f"{timestamp}{sender_name}:",
            f"Subject: {email.subject}",
            "",
            email.body_text or "(no body)",
        ]
        return "\n".join(lines)

    def _format_sms_for_prompt(self, sms: SMSMessage) -> str:
        """Format an SMS message for inclusion in a prompt.

        Args:
            sms: The SMS message to format.

        Returns:
            Formatted string with sender and body.
        """
        timestamp = ""
        if hasattr(sms, "sent_at") and sms.sent_at:
            timestamp = f"[{sms.sent_at.strftime('%Y-%m-%d %H:%M')}] "

        sender_name = self._get_name_for_phone(sms.from_number)
        return f"{timestamp}{sender_name}: {sms.body}"

    def _format_messages_for_prompt(
        self,
        messages: list[Email] | list[SMSMessage],
        modality: Literal["email", "sms"],
    ) -> str:
        """Format multiple messages for inclusion in a prompt.

        Args:
            messages: List of messages to format.
            modality: The modality type.

        Returns:
            Formatted string with all messages, separated by blank lines.
        """
        formatted = []
        for msg in messages:
            formatted.append(self._format_message_for_prompt(msg, modality))
        return "\n\n".join(formatted)

    def _get_name_for_email(self, email: str) -> str:
        """Get a display name for an email address.

        Args:
            email: The email address.

        Returns:
            Character name if found, otherwise the email address.
        """
        character = self._get_character_by_email(email)
        if character:
            return character.name
        return email

    def _get_name_for_phone(self, phone: str) -> str:
        """Get a display name for a phone number.

        Args:
            phone: The phone number.

        Returns:
            Character name if found, otherwise the phone number.
        """
        character = self._get_character_by_phone(phone)
        if character:
            return character.name
        return phone

    def _get_sender_name(self, context: MessageContext) -> str:
        """Get the sender's name for display in prompts.

        Args:
            context: Message context.

        Returns:
            Sender's name or address.
        """
        if context.modality == "email":
            return self._get_name_for_email(context.sender_address)
        else:
            return self._get_name_for_phone(context.sender_address)

    def _get_participant_names(
        self,
        context: MessageContext,
        excluding_character: CharacterProfile,
    ) -> list[str]:
        """Get names of all participants except the responding character.

        Args:
            context: Message context.
            excluding_character: Character to exclude from the list.

        Returns:
            List of participant names.
        """
        names = []
        # Add sender
        names.append(self._get_sender_name(context))

        # Add recipients (excluding the character who is responding)
        for addr in context.all_recipients:
            if context.modality == "email":
                if excluding_character.email and addr.lower() == excluding_character.email.lower():
                    continue
                names.append(self._get_name_for_email(addr))
            else:
                if excluding_character.phone and addr == excluding_character.phone:
                    continue
                names.append(self._get_name_for_phone(addr))

        # Deduplicate while preserving order
        seen = set()
        unique_names = []
        for name in names:
            if name not in seen:
                seen.add(name)
                unique_names.append(name)

        return unique_names

    # =========================================================================
    # Response Building Methods
    # =========================================================================

    def _calculate_response_time(
        self,
        character: CharacterProfile,
        reference_time: datetime,
    ) -> datetime:
        """Calculate when a response should be scheduled.

        Uses the character's response_timing configuration to calculate
        a realistic response delay with variance.

        Args:
            character: The character responding.
            reference_time: Reference time (usually current simulation time).

        Returns:
            The scheduled response time.
        """
        timing = character.response_timing

        # Get base delay and variance as seconds
        base_seconds = timing.base_delay_timedelta.total_seconds()
        variance_seconds = timing.variance_timedelta.total_seconds()

        # Calculate random delay within variance range
        min_delay = max(0, base_seconds - variance_seconds)
        max_delay = base_seconds + variance_seconds
        delay_seconds = random.uniform(min_delay, max_delay)

        return reference_time + timedelta(seconds=delay_seconds)

    def _derive_email_subject(self, original_subject: str) -> str:
        """Derive reply subject line following email conventions.

        Args:
            original_subject: The original email subject.

        Returns:
            Subject with "Re: " prefix if not already present.
        """
        subject = original_subject.strip()
        if subject.lower().startswith("re:"):
            return subject
        return f"Re: {subject}"

    def _build_email_references(self, email: Email) -> list[str]:
        """Build the references header for an email reply.

        Args:
            email: The email being replied to.

        Returns:
            List of message IDs for the References header.
        """
        references = []
        # Add existing references
        if hasattr(email, "references") and email.references:
            references.extend(email.references)
        # Add the message being replied to
        if email.message_id and email.message_id not in references:
            references.append(email.message_id)
        return references

    # =========================================================================
    # Warning Emission
    # =========================================================================

    async def _emit_warning(
        self,
        task_updater: TaskUpdateEmitter,
        message: str,
        warning_type: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Emit a warning via task updater.

        Args:
            task_updater: The task updater to use.
            message: Warning message.
            warning_type: Type of warning for categorization.
            details: Optional additional details.
        """
        # For now, just log - task updates for warnings could be added later
        logger.warning(f"[{warning_type}] {message}: {details}")

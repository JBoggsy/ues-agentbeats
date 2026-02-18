"""Ground truth data for the email_triage_basic scenario.

This module defines the canonical email classifications, urgency labels,
thread membership, and hourly expectations used by programmatic evaluators
in evaluators.py.

All data structures are module-level constants, typed with dataclasses and
enums for safety and IDE support. Evaluators import what they need:

    from scenarios.email_triage_basic.ground_truth import (
        EMAIL_CLASSIFICATIONS,
        NOISE_EMAIL_IDS,
        SUBSTANTIVE_EMAIL_IDS,
        THREAD_MEMBERSHIP,
        HOURLY_EXPECTED_EMAILS,
    )

Design Decision:
    Ground truth lives here (not duplicated in scenario.json params) because
    evaluators are already scenario-specific. The params dict in scenario.json
    is reserved for evaluator-specific configuration (scoring weights,
    thresholds), not raw ground truth tables.

Data Sources:
    All data is derived from the README.md email timeline tables and the
    build_email_triage_state.py email definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# Enums
# =============================================================================


class EmailCategory(str, Enum):
    """Category assigned to each email for triage classification.

    Noise categories should be EXCLUDED from summaries.
    Substantive categories should be INCLUDED in summaries.
    """

    # Noise categories (exclude from summaries)
    SPAM = "spam"
    NEWSLETTER = "newsletter"
    GITHUB_NOTIFICATION = "github_notification"
    CALENDAR_NOTIFICATION = "calendar_notification"
    LINKEDIN_NOTIFICATION = "linkedin_notification"

    # Substantive categories (include in summaries)
    URGENT_WORK = "urgent_work"
    CLIENT_REQUEST = "client_request"
    CLIENT_FOLLOWUP = "client_followup"
    CLIENT_ESCALATION = "client_escalation"
    CLIENT_INTERNAL = "client_internal"
    WORK_ACTION_NEEDED = "work_action_needed"
    ROUTINE_WORK = "routine_work"
    CONFERENCE_CAREER = "conference_career"
    VENDOR_NOTICE = "vendor_notice"
    PERSONAL = "personal"
    IT_HR_FACILITIES = "it_hr_facilities"
    WORK_POSITIVE = "work_positive"

    @property
    def is_noise(self) -> bool:
        """Whether this category is noise (should be excluded)."""
        return self in _NOISE_CATEGORIES

    @property
    def is_substantive(self) -> bool:
        """Whether this category is substantive (should be included)."""
        return not self.is_noise


_NOISE_CATEGORIES = frozenset({
    EmailCategory.SPAM,
    EmailCategory.NEWSLETTER,
    EmailCategory.GITHUB_NOTIFICATION,
    EmailCategory.CALENDAR_NOTIFICATION,
    EmailCategory.LINKEDIN_NOTIFICATION,
})


class Urgency(str, Enum):
    """Urgency label for substantive emails.

    Only three levels: low, medium, high.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(frozen=True)
class EmailClassification:
    """Canonical classification for a single email.

    Attributes:
        email_id: The email's message_id (e.g., "email_001").
        email_number: The email's sequence number (1-49).
        category: The triage category for this email.
        urgency: Expected urgency label (None for noise emails).
        sender_name: Human-readable sender name.
        subject: Email subject line.
        summary_hint: Brief description of what a correct summary should
            mention. Used by summary_accuracy evaluator as reference.
            None for noise emails.
        thread_id: Thread ID if part of a multi-email thread, else None.
        hour: The hour window (1-12) when this email is first available.
            Hour 1 includes all 7 pre-existing emails plus arrivals in
            06:00-07:00.
    """

    email_id: str
    email_number: int
    category: EmailCategory
    urgency: Urgency | None
    sender_name: str
    subject: str
    summary_hint: str | None
    thread_id: str | None
    hour: int


@dataclass(frozen=True)
class ThreadInfo:
    """Information about a multi-email thread.

    Attributes:
        thread_id: The UES thread ID.
        name: Human-readable thread name.
        email_ids: Ordered list of email_ids in this thread.
        arc_description: Description of the narrative arc for evaluators.
        initial_urgency: Urgency at thread start.
        final_urgency: Urgency at thread end (for de-/escalation tracking).
        related_thread_ids: Other thread IDs contextually related to
            this one (e.g., David's internal Acme thread relates to
            Karen's Acme Feature thread).
    """

    thread_id: str
    name: str
    email_ids: list[str] = field(default_factory=list)
    arc_description: str = ""
    initial_urgency: Urgency = Urgency.MEDIUM
    final_urgency: Urgency = Urgency.MEDIUM
    related_thread_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HourlyExpectation:
    """What substantive emails should appear in each hour's summary.

    Attributes:
        hour: Hour number (1-12). Hour 1 = 06:00-07:00, Hour 12 = 17:00-18:00.
        window_start: UTC hour of the window start (e.g., 6 for 06:00).
        substantive_email_ids: Email IDs that should appear in this hour's
            summary (noise emails excluded).
        noise_email_ids: Email IDs that are noise and should NOT appear.
        is_quiet_hour: Whether this is a quiet hour (no substantive emails).
        key_events: Human-readable description of what's notable this hour.
    """

    hour: int
    window_start: int
    substantive_email_ids: list[str] = field(default_factory=list)
    noise_email_ids: list[str] = field(default_factory=list)
    is_quiet_hour: bool = False
    key_events: str = ""


# =============================================================================
# Constants: Thread IDs (must match initial_state.json)
# =============================================================================

THREAD_PROD_INCIDENT = "thread_prod_incident"
THREAD_ACME_FEATURE = "thread_acme_feature"
THREAD_ACME_INTERNAL = "thread_acme_internal"
THREAD_WEEKEND_PLANS = "thread_weekend_plans"


# =============================================================================
# Helper
# =============================================================================


def _eid(n: int) -> str:
    """Generate canonical email ID for email number n."""
    return f"email_{n:03d}"


# =============================================================================
# Email Classifications (49 emails)
# =============================================================================

EMAIL_CLASSIFICATIONS: dict[str, EmailClassification] = {}

_classifications_list: list[EmailClassification] = [
    # -------------------------------------------------------------------------
    # Pre-Existing Emails (7) — available at Hour 1
    # -------------------------------------------------------------------------
    EmailClassification(
        email_id=_eid(1),
        email_number=1,
        category=EmailCategory.PERSONAL,
        urgency=Urgency.LOW,
        sender_name="Sam Rivera",
        subject="Weekend plans? 🍕",
        summary_hint=(
            "Personal email from friend Sam Rivera asking about weekend "
            "dinner plans, suggesting pizza."
        ),
        thread_id=THREAD_WEEKEND_PLANS,
        hour=1,
    ),
    EmailClassification(
        email_id=_eid(2),
        email_number=2,
        category=EmailCategory.GITHUB_NOTIFICATION,
        urgency=None,
        sender_name="GitHub",
        subject="[meridian/api-gateway] Review requested: PR #892",
        summary_hint=None,
        thread_id=None,
        hour=1,
    ),
    EmailClassification(
        email_id=_eid(3),
        email_number=3,
        category=EmailCategory.URGENT_WORK,
        urgency=Urgency.HIGH,
        sender_name="Jordan Lee",
        subject="🔴 ALERT: Production API latency spike — need eyes on this",
        summary_hint=(
            "Urgent production alert from manager Jordan Lee. API gateway "
            "P99 latency spiked to 4.2s (normal <200ms), hitting "
            "/v2/accounts and /v2/transactions endpoints. Started ~03:30 UTC."
        ),
        thread_id=THREAD_PROD_INCIDENT,
        hour=1,
    ),
    EmailClassification(
        email_id=_eid(4),
        email_number=4,
        category=EmailCategory.NEWSLETTER,
        urgency=None,
        sender_name="TechCrunch Daily",
        subject="TechCrunch Daily — AI agents reshape enterprise software landscape",
        summary_hint=None,
        thread_id=None,
        hour=1,
    ),
    EmailClassification(
        email_id=_eid(5),
        email_number=5,
        category=EmailCategory.SPAM,
        urgency=None,
        sender_name="Amazing Prize Center",
        subject="🎉 Congratulations! You've won a $500 gift card!",
        summary_hint=None,
        thread_id=None,
        hour=1,
    ),
    EmailClassification(
        email_id=_eid(6),
        email_number=6,
        category=EmailCategory.IT_HR_FACILITIES,
        urgency=Urgency.LOW,
        sender_name="IT Department",
        subject="Scheduled maintenance: Saturday 02:00-06:00 UTC",
        summary_hint=(
            "IT notice about scheduled maintenance Saturday 02:00-06:00 UTC. "
            "Internal Git repos will be read-only, CI/CD queued, "
            "VPN may have brief interruptions. Production unaffected."
        ),
        thread_id=None,
        hour=1,
    ),
    EmailClassification(
        email_id=_eid(7),
        email_number=7,
        category=EmailCategory.IT_HR_FACILITIES,
        urgency=Urgency.LOW,
        sender_name="HR Department",
        subject="Reminder: All-hands meeting Thursday 2pm",
        summary_hint=(
            "HR reminder about monthly all-hands meeting Thursday Jan 30 at "
            "2pm UTC. Agenda includes Q4 results, Q1 priorities, benefits "
            "enrollment deadline."
        ),
        thread_id=None,
        hour=1,
    ),

    # -------------------------------------------------------------------------
    # Hour 1: 06:00-07:00 (2 arriving emails)
    # -------------------------------------------------------------------------
    EmailClassification(
        email_id=_eid(8),
        email_number=8,
        category=EmailCategory.GITHUB_NOTIFICATION,
        urgency=None,
        sender_name="GitHub",
        subject="[meridian/api-gateway] CI failed: PR #892",
        summary_hint=None,
        thread_id=None,
        hour=1,
    ),
    EmailClassification(
        email_id=_eid(9),
        email_number=9,
        category=EmailCategory.SPAM,
        urgency=None,
        sender_name="Prince Abubakar",
        subject="URGENT: Inheritance transfer requires your help",
        summary_hint=None,
        thread_id=None,
        hour=1,
    ),

    # -------------------------------------------------------------------------
    # Hour 2: 07:00-08:00 (3 arriving emails)
    # -------------------------------------------------------------------------
    EmailClassification(
        email_id=_eid(10),
        email_number=10,
        category=EmailCategory.CLIENT_REQUEST,
        urgency=Urgency.MEDIUM,
        sender_name="Karen Mitchell",
        subject="Dashboard export feature — timeline needed",
        summary_hint=(
            "External client Karen Mitchell (VP Product, Acme Corp) "
            "requesting timeline for dashboard export feature (CSV + PDF). "
            "Their Q2 planning depends on it. CC'd David Chen."
        ),
        thread_id=THREAD_ACME_FEATURE,
        hour=2,
    ),
    EmailClassification(
        email_id=_eid(11),
        email_number=11,
        category=EmailCategory.NEWSLETTER,
        urgency=None,
        sender_name="Morning Brew",
        subject="Morning Brew ☕ — Markets rally on Fed signals, AI chip shortage deepens",
        summary_hint=None,
        thread_id=None,
        hour=2,
    ),
    EmailClassification(
        email_id=_eid(12),
        email_number=12,
        category=EmailCategory.ROUTINE_WORK,
        urgency=Urgency.LOW,
        sender_name="Marcus Williams",
        subject="Standup agenda — anything to add?",
        summary_hint=(
            "Marcus Williams running standup today, asking if anyone has "
            "items to add. Current agenda: rate limiter refactor, sprint "
            "planning prep, database migration status."
        ),
        thread_id=None,
        hour=2,
    ),

    # -------------------------------------------------------------------------
    # Hour 3: 08:00-09:00 (5 arriving emails)
    # -------------------------------------------------------------------------
    EmailClassification(
        email_id=_eid(13),
        email_number=13,
        category=EmailCategory.URGENT_WORK,
        urgency=Urgency.HIGH,
        sender_name="Priya Sharma",
        subject="Re: ALERT: Production API latency spike — initial analysis",
        summary_hint=(
            "Priya's initial analysis of the prod incident. CPU at 94% on "
            "primary DB, connection pool exhaustion, full table scan on "
            "transactions table correlated with v2.14.3 deployment that "
            "dropped an index. Investigating hotfix."
        ),
        thread_id=THREAD_PROD_INCIDENT,
        hour=3,
    ),
    EmailClassification(
        email_id=_eid(14),
        email_number=14,
        category=EmailCategory.CALENDAR_NOTIFICATION,
        urgency=None,
        sender_name="Google Calendar",
        subject="Reminder: Architecture Review @ 10:00 AM",
        summary_hint=None,
        thread_id=None,
        hour=3,
    ),
    EmailClassification(
        email_id=_eid(15),
        email_number=15,
        category=EmailCategory.CLIENT_FOLLOWUP,
        urgency=Urgency.MEDIUM,
        sender_name="Karen Mitchell",
        subject="Re: Dashboard export feature — following up",
        summary_hint=(
            "Karen following up on her earlier feature request. Her CEO "
            "wants status for a board meeting next week. Asking for even "
            "a ballpark estimate."
        ),
        thread_id=THREAD_ACME_FEATURE,
        hour=3,
    ),
    EmailClassification(
        email_id=_eid(16),
        email_number=16,
        category=EmailCategory.GITHUB_NOTIFICATION,
        urgency=None,
        sender_name="GitHub",
        subject="[meridian/core-lib] You were assigned: Issue #4521",
        summary_hint=None,
        thread_id=None,
        hour=3,
    ),
    EmailClassification(
        email_id=_eid(17),
        email_number=17,
        category=EmailCategory.CONFERENCE_CAREER,
        urgency=Urgency.LOW,
        sender_name="DevConf 2026 Committee",
        subject="Invitation: Speak at DevConf 2026?",
        summary_hint=(
            "Invitation to speak at DevConf 2026 (June 15-17, San Francisco) "
            "in the 'Building Resilient APIs' track. 30-minute talk, "
            "conference pass + $1,500 travel stipend. Reply by Feb 15."
        ),
        thread_id=None,
        hour=3,
    ),

    # -------------------------------------------------------------------------
    # Hour 4: 09:00-10:00 (6 arriving emails)
    # -------------------------------------------------------------------------
    EmailClassification(
        email_id=_eid(18),
        email_number=18,
        category=EmailCategory.URGENT_WORK,
        urgency=Urgency.HIGH,
        sender_name="Marcus Williams",
        subject="Re: ALERT: Production API latency spike — monitoring data",
        summary_hint=(
            "Marcus sharing monitoring data: connection pool at 98%, 187 "
            "active queries, 12s replication lag. Scaled API pods from "
            "3 to 5. Confirmed Priya's missing index finding from "
            "migration log."
        ),
        thread_id=THREAD_PROD_INCIDENT,
        hour=4,
    ),
    EmailClassification(
        email_id=_eid(19),
        email_number=19,
        category=EmailCategory.URGENT_WORK,
        urgency=Urgency.HIGH,
        sender_name="Jordan Lee",
        subject=(
            "Re: ALERT: Production API latency spike "
            "— status update needed ASAP"
        ),
        summary_hint=(
            "Jordan (manager) demanding a clear status update for VP of "
            "Engineering within 15 minutes. Needs: confirmed root cause, "
            "customer impact count, resolution ETA, rollback vs hotfix "
            "decision."
        ),
        thread_id=THREAD_PROD_INCIDENT,
        hour=4,
    ),
    EmailClassification(
        email_id=_eid(20),
        email_number=20,
        category=EmailCategory.CLIENT_INTERNAL,
        urgency=Urgency.MEDIUM,
        sender_name="David Chen",
        subject="Acme Corp situation — can you weigh in?",
        summary_hint=(
            "Account manager David Chen asking Alex for a realistic timeline "
            "on the dashboard export feature. Acme contract renewal in Q2, "
            "Karen's CEO is involved. $450K/year client at risk."
        ),
        thread_id=THREAD_ACME_INTERNAL,
        hour=4,
    ),
    EmailClassification(
        email_id=_eid(21),
        email_number=21,
        category=EmailCategory.GITHUB_NOTIFICATION,
        urgency=None,
        sender_name="GitHub",
        subject="[meridian/api-gateway] PR #892 merged",
        summary_hint=None,
        thread_id=None,
        hour=4,
    ),
    EmailClassification(
        email_id=_eid(22),
        email_number=22,
        category=EmailCategory.VENDOR_NOTICE,
        urgency=Urgency.LOW,
        sender_name="VendorSoft Renewals",
        subject="Action required: Your VendorSoft license expires in 30 days",
        summary_hint=(
            "VendorSoft Professional license (10 seats) expiring Feb 27. "
            "Renewal is $2,400/year. Action needed before expiration."
        ),
        thread_id=None,
        hour=4,
    ),
    EmailClassification(
        email_id=_eid(23),
        email_number=23,
        category=EmailCategory.SPAM,
        urgency=None,
        sender_name="MarketBoost SEO",
        subject="10x your website traffic with our SEO services!",
        summary_hint=None,
        thread_id=None,
        hour=4,
    ),

    # -------------------------------------------------------------------------
    # Hour 5: 10:00-11:00 (4 arriving emails)
    # -------------------------------------------------------------------------
    EmailClassification(
        email_id=_eid(24),
        email_number=24,
        category=EmailCategory.URGENT_WORK,
        urgency=Urgency.MEDIUM,
        sender_name="Priya Sharma",
        subject="Re: ALERT: Production API latency spike — root cause confirmed",
        summary_hint=(
            "Root cause confirmed: migration v2.14.3 dropped composite "
            "index. Hotfix PR ready (CREATE INDEX CONCURRENTLY). Latency "
            "recovering: P99 down from 5.1s to 1.8s. ~12K requests "
            "affected over 6 hours. Recommends deploying today."
        ),
        thread_id=THREAD_PROD_INCIDENT,
        hour=5,
    ),
    EmailClassification(
        email_id=_eid(25),
        email_number=25,
        category=EmailCategory.CLIENT_ESCALATION,
        urgency=Urgency.HIGH,
        sender_name="Karen Mitchell",
        subject="Re: Dashboard export feature — still waiting for a response",
        summary_hint=(
            "Karen escalating — 3+ hours without response. Q2 analytics "
            "strategy depends on dashboard export. Demands a response today."
        ),
        thread_id=THREAD_ACME_FEATURE,
        hour=5,
    ),
    EmailClassification(
        email_id=_eid(26),
        email_number=26,
        category=EmailCategory.LINKEDIN_NOTIFICATION,
        urgency=None,
        sender_name="LinkedIn",
        subject="5 people viewed your profile this week",
        summary_hint=None,
        thread_id=None,
        hour=5,
    ),
    EmailClassification(
        email_id=_eid(27),
        email_number=27,
        category=EmailCategory.WORK_ACTION_NEEDED,
        urgency=Urgency.MEDIUM,
        sender_name="Lisa Park",
        subject="Q1 budget review — need your input by Friday",
        summary_hint=(
            "Lisa Park (Finance) requesting Q1 budget input by Friday: "
            "cloud infra costs ($18K/month budget), tool licenses, "
            "training/conference budget for Alex's team."
        ),
        thread_id=None,
        hour=5,
    ),

    # -------------------------------------------------------------------------
    # Hour 6: 11:00-12:00 (4 arriving emails)
    # -------------------------------------------------------------------------
    EmailClassification(
        email_id=_eid(28),
        email_number=28,
        category=EmailCategory.CLIENT_ESCALATION,
        urgency=Urgency.HIGH,
        sender_name="Karen Mitchell",
        subject="Re: Dashboard export feature — escalation warning",
        summary_hint=(
            "Karen issuing escalation warning — three emails with zero "
            "response. Now CC'ing Jordan Lee and David Chen. Threatening "
            "to escalate to Meridian VP of Engineering and Acme executive "
            "team. Emphasizes Acme's revenue significance."
        ),
        thread_id=THREAD_ACME_FEATURE,
        hour=6,
    ),
    EmailClassification(
        email_id=_eid(29),
        email_number=29,
        category=EmailCategory.NEWSLETTER,
        urgency=None,
        sender_name="Engineering Weekly",
        subject=(
            "Engineering Weekly: Q1 roadmap updates, team kudos, tech talks"
        ),
        summary_hint=None,
        thread_id=None,
        hour=6,
    ),
    EmailClassification(
        email_id=_eid(30),
        email_number=30,
        category=EmailCategory.WORK_ACTION_NEEDED,
        urgency=Urgency.MEDIUM,
        sender_name="Priya Sharma",
        subject="Post-mortem draft — production incident 01/28",
        summary_hint=(
            "Priya sharing post-mortem draft for this morning's production "
            "incident. Requesting review from Alex and Jordan before wider "
            "distribution. Wants input on 'Preventive Measures' section."
        ),
        thread_id=None,
        hour=6,
    ),
    EmailClassification(
        email_id=_eid(31),
        email_number=31,
        category=EmailCategory.PERSONAL,
        urgency=Urgency.LOW,
        sender_name="Sam Rivera",
        subject="Re: Weekend plans? — restaurant reservation?",
        summary_hint=(
            "Sam following up on weekend plans — suggesting Bella Notte "
            "Italian restaurant on 5th Street for Saturday at 7pm. "
            "Asking about reservation preferences."
        ),
        thread_id=THREAD_WEEKEND_PLANS,
        hour=6,
    ),

    # -------------------------------------------------------------------------
    # Hour 7: 12:00-13:00 (2 arriving emails) — QUIET HOUR
    # -------------------------------------------------------------------------
    EmailClassification(
        email_id=_eid(32),
        email_number=32,
        category=EmailCategory.SPAM,
        urgency=None,
        sender_name="MediDiscount Pharmacy",
        subject="💊 Save 80% on prescription medications today!",
        summary_hint=None,
        thread_id=None,
        hour=7,
    ),
    EmailClassification(
        email_id=_eid(33),
        email_number=33,
        category=EmailCategory.CALENDAR_NOTIFICATION,
        urgency=None,
        sender_name="Google Calendar",
        subject="Reminder: 1:1 with Jordan @ 2:00 PM",
        summary_hint=None,
        thread_id=None,
        hour=7,
    ),

    # -------------------------------------------------------------------------
    # Hour 8: 13:00-14:00 (4 arriving emails)
    # -------------------------------------------------------------------------
    EmailClassification(
        email_id=_eid(34),
        email_number=34,
        category=EmailCategory.WORK_ACTION_NEEDED,
        urgency=Urgency.MEDIUM,
        sender_name="Jordan Lee",
        subject=(
            "Re: ALERT: Production API latency spike "
            "— post-mortem timeline?"
        ),
        summary_hint=(
            "Jordan asking Alex to coordinate finalizing the post-mortem "
            "by end of week. Wants concrete follow-up action items with "
            "owners and deadlines."
        ),
        thread_id=THREAD_PROD_INCIDENT,
        hour=8,
    ),
    EmailClassification(
        email_id=_eid(35),
        email_number=35,
        category=EmailCategory.GITHUB_NOTIFICATION,
        urgency=None,
        sender_name="GitHub",
        subject="[meridian/core-lib] Comment on Issue #4521",
        summary_hint=None,
        thread_id=None,
        hour=8,
    ),
    EmailClassification(
        email_id=_eid(36),
        email_number=36,
        category=EmailCategory.SPAM,
        urgency=None,
        sender_name="BlockProfit",
        subject="🚀 Turn $100 into $10,000 — limited time crypto deal!",
        summary_hint=None,
        thread_id=None,
        hour=8,
    ),
    EmailClassification(
        email_id=_eid(37),
        email_number=37,
        category=EmailCategory.CONFERENCE_CAREER,
        urgency=Urgency.LOW,
        sender_name="RecruitPro Talent",
        subject="Exciting opportunity: Staff Engineer at [Stealth Startup]",
        summary_hint=(
            "Recruiter outreach for Staff Engineer role at stealth startup "
            "(Series B, $80M raised). $280K-$340K base + equity, "
            "remote-first. Exploratory conversation requested."
        ),
        thread_id=None,
        hour=8,
    ),

    # -------------------------------------------------------------------------
    # Hour 9: 14:00-15:00 (5 arriving emails)
    # -------------------------------------------------------------------------
    EmailClassification(
        email_id=_eid(38),
        email_number=38,
        category=EmailCategory.CLIENT_ESCALATION,
        urgency=Urgency.HIGH,
        sender_name="Karen Mitchell",
        subject="Re: Dashboard export feature — this is unacceptable",
        summary_hint=(
            "Karen furious — four emails with no response. Threatening "
            "to recommend evaluating alternative platforms at her executive "
            "team meeting in one hour. Deadline: 3pm today for a concrete "
            "timeline."
        ),
        thread_id=THREAD_ACME_FEATURE,
        hour=9,
    ),
    EmailClassification(
        email_id=_eid(39),
        email_number=39,
        category=EmailCategory.CLIENT_INTERNAL,
        urgency=Urgency.HIGH,
        sender_name="David Chen",
        subject="Re: Acme Corp situation — getting urgent on their end",
        summary_hint=(
            "David reporting Karen is furious after a call — threatening "
            "to evaluate competitor platforms. Acme renewal worth $450K/year "
            "is at risk. Urgent request for Alex to respond to Karen today."
        ),
        thread_id=THREAD_ACME_INTERNAL,
        hour=9,
    ),
    EmailClassification(
        email_id=_eid(40),
        email_number=40,
        category=EmailCategory.WORK_ACTION_NEEDED,
        urgency=Urgency.MEDIUM,
        sender_name="Marcus Williams",
        subject="Architecture review follow-up — action items",
        summary_hint=(
            "Architecture review action items: Alex to draft RFC for "
            "payments microservice decomposition (due Feb 7). Team "
            "decided on strangler fig pattern, Go for new performance-"
            "critical services."
        ),
        thread_id=None,
        hour=9,
    ),
    EmailClassification(
        email_id=_eid(41),
        email_number=41,
        category=EmailCategory.GITHUB_NOTIFICATION,
        urgency=None,
        sender_name="GitHub",
        subject="[meridian/api-gateway] Security advisory: CVE-2026-1234",
        summary_hint=None,
        thread_id=None,
        hour=9,
    ),
    EmailClassification(
        email_id=_eid(42),
        email_number=42,
        category=EmailCategory.IT_HR_FACILITIES,
        urgency=Urgency.LOW,
        sender_name="Facilities",
        subject="Office snack preferences survey — vote by Friday!",
        summary_hint=(
            "Facilities team requesting snack preference survey responses "
            "by Friday. Also considering adding a cold brew tap."
        ),
        thread_id=None,
        hour=9,
    ),

    # -------------------------------------------------------------------------
    # Hour 10: 15:00-16:00 (3 arriving emails)
    # -------------------------------------------------------------------------
    EmailClassification(
        email_id=_eid(43),
        email_number=43,
        category=EmailCategory.WORK_ACTION_NEEDED,
        urgency=Urgency.MEDIUM,
        sender_name="Priya Sharma",
        subject="Hotfix deployment plan — review needed",
        summary_hint=(
            "Priya's hotfix deployment plan for the missing index. "
            "Timeline: staging at 16:00, verification at 16:30, "
            "production at 17:00. Needs review and approval, plus "
            "someone on-call during deployment."
        ),
        thread_id=None,
        hour=10,
    ),
    EmailClassification(
        email_id=_eid(44),
        email_number=44,
        category=EmailCategory.CALENDAR_NOTIFICATION,
        urgency=None,
        sender_name="Google Calendar",
        subject="Updated: Team retrospective moved to Friday 3pm",
        summary_hint=None,
        thread_id=None,
        hour=10,
    ),
    EmailClassification(
        email_id=_eid(45),
        email_number=45,
        category=EmailCategory.NEWSLETTER,
        urgency=None,
        sender_name="TechCrunch PM",
        subject=(
            "TechCrunch PM Edition — More layoffs in Big Tech, "
            "startup funding rebounds"
        ),
        summary_hint=None,
        thread_id=None,
        hour=10,
    ),

    # -------------------------------------------------------------------------
    # Hour 11: 16:00-17:00 (2 arriving emails)
    # -------------------------------------------------------------------------
    EmailClassification(
        email_id=_eid(46),
        email_number=46,
        category=EmailCategory.CLIENT_ESCALATION,
        urgency=Urgency.HIGH,
        sender_name="Karen Mitchell",
        subject="Re: Dashboard export feature — any update?",
        summary_hint=(
            "Karen's fifth email — met with exec team, formally "
            "requesting meeting with Meridian VP of Engineering. "
            "Giving one more chance to respond before further "
            "escalation."
        ),
        thread_id=THREAD_ACME_FEATURE,
        hour=11,
    ),
    EmailClassification(
        email_id=_eid(47),
        email_number=47,
        category=EmailCategory.WORK_POSITIVE,
        urgency=Urgency.LOW,
        sender_name="Jordan Lee",
        subject="Good work today, team 👏",
        summary_hint=(
            "Jordan sending kudos to the team for handling the production "
            "incident well. Praises Priya's post-mortem, Marcus's pod "
            "scaling, and Alex's coordination. Asks Alex to track "
            "follow-ups in Jira."
        ),
        thread_id=None,
        hour=11,
    ),

    # -------------------------------------------------------------------------
    # Hour 12: 17:00-18:00 (2 arriving emails)
    # -------------------------------------------------------------------------
    EmailClassification(
        email_id=_eid(48),
        email_number=48,
        category=EmailCategory.PERSONAL,
        urgency=Urgency.LOW,
        sender_name="Sam Rivera",
        subject="Re: Weekend plans? — dinner confirmed! 🎉",
        summary_hint=(
            "Sam confirming dinner reservation at Bella Notte for "
            "Saturday at 7pm, table for 2."
        ),
        thread_id=THREAD_WEEKEND_PLANS,
        hour=12,
    ),
    EmailClassification(
        email_id=_eid(49),
        email_number=49,
        category=EmailCategory.SPAM,
        urgency=None,
        sender_name="WatchKingdom",
        subject="Rolex at 90% OFF — today only!!!",
        summary_hint=None,
        thread_id=None,
        hour=12,
    ),
]

# Build the lookup dict
for _c in _classifications_list:
    EMAIL_CLASSIFICATIONS[_c.email_id] = _c


# =============================================================================
# Derived Sets
# =============================================================================

NOISE_EMAIL_IDS: frozenset[str] = frozenset(
    c.email_id for c in _classifications_list if c.category.is_noise
)
"""Set of all 20 noise email IDs that should be excluded from summaries."""

SUBSTANTIVE_EMAIL_IDS: frozenset[str] = frozenset(
    c.email_id for c in _classifications_list if c.category.is_substantive
)
"""Set of all 29 substantive email IDs that should be included in summaries."""

SPAM_EMAIL_IDS: frozenset[str] = frozenset(
    c.email_id
    for c in _classifications_list
    if c.category == EmailCategory.SPAM
)
"""Set of 6 spam email IDs."""

GITHUB_NOTIFICATION_IDS: frozenset[str] = frozenset(
    c.email_id
    for c in _classifications_list
    if c.category == EmailCategory.GITHUB_NOTIFICATION
)
"""Set of 6 GitHub notification email IDs."""

CALENDAR_NOTIFICATION_IDS: frozenset[str] = frozenset(
    c.email_id
    for c in _classifications_list
    if c.category == EmailCategory.CALENDAR_NOTIFICATION
)
"""Set of 3 calendar notification email IDs."""

NEWSLETTER_IDS: frozenset[str] = frozenset(
    c.email_id
    for c in _classifications_list
    if c.category == EmailCategory.NEWSLETTER
)
"""Set of 4 newsletter email IDs."""


# =============================================================================
# Thread Membership
# =============================================================================

THREAD_MEMBERSHIP: dict[str, ThreadInfo] = {
    THREAD_PROD_INCIDENT: ThreadInfo(
        thread_id=THREAD_PROD_INCIDENT,
        name="Production Incident",
        email_ids=[
            _eid(3), _eid(13), _eid(18), _eid(19), _eid(24), _eid(34),
        ],
        arc_description=(
            "Alert → Investigation (Priya's analysis) → Monitoring data "
            "(Marcus) → Manager escalation (Jordan wants status) → "
            "Root cause confirmed & resolution (Priya) → "
            "Post-mortem timeline request (Jordan)"
        ),
        initial_urgency=Urgency.HIGH,
        final_urgency=Urgency.MEDIUM,
    ),
    THREAD_ACME_FEATURE: ThreadInfo(
        thread_id=THREAD_ACME_FEATURE,
        name="Acme Corp Client — Dashboard Export",
        email_ids=[
            _eid(10), _eid(15), _eid(25), _eid(28), _eid(38), _eid(46),
        ],
        arc_description=(
            "Polite request → Follow-up (CEO involved) → Impatience "
            "(3+ hours no response) → Escalation warning (CC's manager) "
            "→ Anger (threatening to evaluate alternatives) → "
            "Final poke (exec meeting happened, VP meeting requested)"
        ),
        initial_urgency=Urgency.MEDIUM,
        final_urgency=Urgency.HIGH,
        related_thread_ids=[THREAD_ACME_INTERNAL],
    ),
    THREAD_ACME_INTERNAL: ThreadInfo(
        thread_id=THREAD_ACME_INTERNAL,
        name="Acme Corp Internal — David Chen",
        email_ids=[_eid(20), _eid(39)],
        arc_description=(
            "David asks for realistic timeline assessment → David reports "
            "Karen is furious, relationship at risk, $450K/year renewal "
            "threatened"
        ),
        initial_urgency=Urgency.MEDIUM,
        final_urgency=Urgency.HIGH,
        related_thread_ids=[THREAD_ACME_FEATURE],
    ),
    THREAD_WEEKEND_PLANS: ThreadInfo(
        thread_id=THREAD_WEEKEND_PLANS,
        name="Weekend Plans — Sam Rivera",
        email_ids=[_eid(1), _eid(31), _eid(48)],
        arc_description=(
            "Invitation (pizza, Saturday evening?) → Restaurant suggestion "
            "(Bella Notte, Saturday 7pm) → Confirmation (reservation made)"
        ),
        initial_urgency=Urgency.LOW,
        final_urgency=Urgency.LOW,
    ),
}

# Non-initial thread emails: emails whose summaries should show cumulative
# thread awareness (i.e., reference information from prior thread emails).
# This is all thread emails EXCEPT the first one in each thread.
NON_INITIAL_THREAD_EMAIL_IDS: frozenset[str] = frozenset(
    eid
    for info in THREAD_MEMBERSHIP.values()
    for eid in info.email_ids[1:]  # Skip first email in each thread
)
"""Set of 14 non-initial thread emails where thread tracking is evaluated.

Breakdown:
- Production Incident: 5 (#13, #18, #19, #24, #34)
- Acme Feature (Karen): 5 (#15, #25, #28, #38, #46)
- Acme Internal (David): 1 (#39) — first email (#20) is initial
- Weekend Plans: 2 (#31, #48)

Note: David's emails (#20, #39) are in a separate UES thread from Karen's,
but evaluators expect the agent to connect them contextually.

The thread_tracking evaluator also awards points for David's emails being
connected to Karen's Acme Feature thread, despite being in a different
UES thread_id. Specifically:
- email_020: First in its thread, but evaluator checks if agent connects
  it to Karen's thread contextually (counted separately, not in this set)
- email_039: Non-initial in its thread AND should be connected to Karen's
  thread contextually

Total non-initial: 5 + 5 + 1 + 2 = 13 from email_ids[1:], plus 1 for
email_020's cross-thread connection = 14 evaluated points.
"""

# Cross-thread contextual connections that evaluators should check.
# David's internal Acme emails should be connected to Karen's thread.
CROSS_THREAD_CONNECTIONS: dict[str, str] = {
    _eid(20): THREAD_ACME_FEATURE,
    _eid(39): THREAD_ACME_FEATURE,
}
"""Email IDs that belong to one UES thread but should be contextually
connected to another thread by a smart agent.

email_020 and email_039 are in thread_acme_internal, but reference Karen
Mitchell, Acme Corp, and the dashboard export feature — the same situation
tracked in thread_acme_feature.
"""


# =============================================================================
# Hourly Expected Emails
# =============================================================================

HOURLY_EXPECTED_EMAILS: dict[int, HourlyExpectation] = {
    1: HourlyExpectation(
        hour=1,
        window_start=6,
        substantive_email_ids=[
            _eid(3), _eid(1), _eid(6), _eid(7),
        ],
        noise_email_ids=[
            _eid(2), _eid(4), _eid(5), _eid(8), _eid(9),
        ],
        key_events=(
            "Pre-existing backlog: prod alert (high), weekend plans (low), "
            "IT maintenance (low), HR all-hands (low). Plus 2 noise arrivals."
        ),
    ),
    2: HourlyExpectation(
        hour=2,
        window_start=7,
        substantive_email_ids=[_eid(10), _eid(12)],
        noise_email_ids=[_eid(11)],
        key_events=(
            "Karen's first email (medium), standup agenda (low). "
            "Morning Brew newsletter arrives."
        ),
    ),
    3: HourlyExpectation(
        hour=3,
        window_start=8,
        substantive_email_ids=[_eid(13), _eid(15), _eid(17)],
        noise_email_ids=[_eid(14), _eid(16)],
        key_events=(
            "Priya's incident analysis (high), Karen follows up (medium), "
            "DevConf invite (low). Calendar + GitHub notifications."
        ),
    ),
    4: HourlyExpectation(
        hour=4,
        window_start=9,
        substantive_email_ids=[
            _eid(18), _eid(19), _eid(20), _eid(22),
        ],
        noise_email_ids=[_eid(21), _eid(23)],
        key_events=(
            "Incident thread heats up (Marcus monitoring data + Jordan "
            "wants status, both high). David flags Acme situation "
            "(medium). VendorSoft renewal (low)."
        ),
    ),
    5: HourlyExpectation(
        hour=5,
        window_start=10,
        substantive_email_ids=[_eid(24), _eid(25), _eid(27)],
        noise_email_ids=[_eid(26)],
        key_events=(
            "Prod incident root cause confirmed — resolving (medium, "
            "de-escalated). Karen escalates (high). Budget review "
            "request (medium)."
        ),
    ),
    6: HourlyExpectation(
        hour=6,
        window_start=11,
        substantive_email_ids=[_eid(28), _eid(30), _eid(31)],
        noise_email_ids=[_eid(29)],
        key_events=(
            "Karen escalation warning (high). Post-mortem draft "
            "(medium). Sam's restaurant follow-up (low)."
        ),
    ),
    7: HourlyExpectation(
        hour=7,
        window_start=12,
        substantive_email_ids=[],
        noise_email_ids=[_eid(32), _eid(33)],
        is_quiet_hour=True,
        key_events=(
            "Quiet hour — only spam and a calendar notification. "
            "Agent should send a 'quiet hour' message."
        ),
    ),
    8: HourlyExpectation(
        hour=8,
        window_start=13,
        substantive_email_ids=[_eid(34), _eid(37)],
        noise_email_ids=[_eid(35), _eid(36)],
        key_events=(
            "Jordan asks for post-mortem timeline (medium). "
            "Recruiter outreach (low). GitHub comment + crypto spam."
        ),
    ),
    9: HourlyExpectation(
        hour=9,
        window_start=14,
        substantive_email_ids=[
            _eid(38), _eid(39), _eid(40), _eid(42),
        ],
        noise_email_ids=[_eid(41)],
        key_events=(
            "Karen goes nuclear (high). David sounds alarm (high). "
            "Architecture review action items (medium). Facilities "
            "survey (low). GitHub security advisory."
        ),
    ),
    10: HourlyExpectation(
        hour=10,
        window_start=15,
        substantive_email_ids=[_eid(43)],
        noise_email_ids=[_eid(44), _eid(45)],
        key_events=(
            "Hotfix deployment plan (medium). Calendar update + "
            "TechCrunch PM newsletter."
        ),
    ),
    11: HourlyExpectation(
        hour=11,
        window_start=16,
        substantive_email_ids=[_eid(46), _eid(47)],
        noise_email_ids=[],
        key_events=(
            "Karen's final poke (high). Jordan's kudos (low)."
        ),
    ),
    12: HourlyExpectation(
        hour=12,
        window_start=17,
        substantive_email_ids=[_eid(48)],
        noise_email_ids=[_eid(49)],
        key_events=(
            "Sam's dinner confirmation (low). End-of-day spam."
        ),
    ),
}


# =============================================================================
# Validation (runs at import time)
# =============================================================================

def _validate() -> None:
    """Validate internal consistency of ground truth data.

    Raises:
        AssertionError: If any consistency check fails.
    """
    # Total email count
    assert len(EMAIL_CLASSIFICATIONS) == 49, (
        f"Expected 49 emails, got {len(EMAIL_CLASSIFICATIONS)}"
    )

    # Noise / substantive split
    assert len(NOISE_EMAIL_IDS) == 20, (
        f"Expected 20 noise emails, got {len(NOISE_EMAIL_IDS)}"
    )
    assert len(SUBSTANTIVE_EMAIL_IDS) == 29, (
        f"Expected 29 substantive emails, got {len(SUBSTANTIVE_EMAIL_IDS)}"
    )

    # No overlap
    assert not NOISE_EMAIL_IDS & SUBSTANTIVE_EMAIL_IDS, (
        "Overlap between noise and substantive email sets"
    )

    # Union equals full set
    assert NOISE_EMAIL_IDS | SUBSTANTIVE_EMAIL_IDS == set(
        EMAIL_CLASSIFICATIONS.keys()
    ), "Noise + substantive doesn't cover all emails"

    # Noise breakdown: 6 spam + 6 GitHub + 3 calendar + 4 newsletter + 1 LinkedIn = 20
    assert len(SPAM_EMAIL_IDS) == 6, (
        f"Expected 6 spam, got {len(SPAM_EMAIL_IDS)}"
    )
    assert len(GITHUB_NOTIFICATION_IDS) == 6, (
        f"Expected 6 GitHub notifications, got {len(GITHUB_NOTIFICATION_IDS)}"
    )
    assert len(CALENDAR_NOTIFICATION_IDS) == 3, (
        f"Expected 3 calendar notifications, "
        f"got {len(CALENDAR_NOTIFICATION_IDS)}"
    )
    assert len(NEWSLETTER_IDS) == 4, (
        f"Expected 4 newsletters, got {len(NEWSLETTER_IDS)}"
    )
    linkedin_count = sum(
        1 for c in _classifications_list
        if c.category == EmailCategory.LINKEDIN_NOTIFICATION
    )
    assert linkedin_count == 1, (
        f"Expected 1 LinkedIn notification, got {linkedin_count}"
    )

    # All substantive emails have urgency; all noise emails don't
    for c in _classifications_list:
        if c.category.is_substantive:
            assert c.urgency is not None, (
                f"{c.email_id} is substantive but has no urgency"
            )
            assert c.summary_hint is not None, (
                f"{c.email_id} is substantive but has no summary_hint"
            )
        else:
            assert c.urgency is None, (
                f"{c.email_id} is noise but has urgency={c.urgency}"
            )
            assert c.summary_hint is None, (
                f"{c.email_id} is noise but has a summary_hint"
            )

    # Thread membership consistency
    all_thread_email_ids: set[str] = set()
    for info in THREAD_MEMBERSHIP.values():
        for eid in info.email_ids:
            assert eid in EMAIL_CLASSIFICATIONS, (
                f"Thread {info.thread_id} references unknown email {eid}"
            )
            ec = EMAIL_CLASSIFICATIONS[eid]
            assert ec.thread_id == info.thread_id, (
                f"{eid} in thread {info.thread_id} but classification "
                f"says thread_id={ec.thread_id}"
            )
            all_thread_email_ids.add(eid)

    # All emails with thread_ids are in THREAD_MEMBERSHIP
    for c in _classifications_list:
        if c.thread_id:
            assert c.email_id in all_thread_email_ids, (
                f"{c.email_id} has thread_id={c.thread_id} but isn't in "
                f"THREAD_MEMBERSHIP"
            )

    # Thread sizes
    assert len(THREAD_MEMBERSHIP[THREAD_PROD_INCIDENT].email_ids) == 6
    assert len(THREAD_MEMBERSHIP[THREAD_ACME_FEATURE].email_ids) == 6
    assert len(THREAD_MEMBERSHIP[THREAD_ACME_INTERNAL].email_ids) == 2
    assert len(THREAD_MEMBERSHIP[THREAD_WEEKEND_PLANS].email_ids) == 3

    # Hourly expectations cover all 12 hours
    assert set(HOURLY_EXPECTED_EMAILS.keys()) == set(range(1, 13)), (
        "Hourly expectations must cover hours 1-12"
    )

    # All emails in hourly expectations map back to valid classifications
    all_hourly_substantive: set[str] = set()
    all_hourly_noise: set[str] = set()
    for he in HOURLY_EXPECTED_EMAILS.values():
        for eid in he.substantive_email_ids:
            assert eid in SUBSTANTIVE_EMAIL_IDS, (
                f"Hour {he.hour}: {eid} listed as substantive but isn't"
            )
            all_hourly_substantive.add(eid)
        for eid in he.noise_email_ids:
            assert eid in NOISE_EMAIL_IDS, (
                f"Hour {he.hour}: {eid} listed as noise but isn't"
            )
            all_hourly_noise.add(eid)

    # Every email should appear in exactly one hour's expectations
    assert all_hourly_substantive == SUBSTANTIVE_EMAIL_IDS, (
        f"Substantive email mismatch. "
        f"Missing from hourly: {SUBSTANTIVE_EMAIL_IDS - all_hourly_substantive}. "
        f"Extra in hourly: {all_hourly_substantive - SUBSTANTIVE_EMAIL_IDS}"
    )
    assert all_hourly_noise == NOISE_EMAIL_IDS, (
        f"Noise email mismatch. "
        f"Missing from hourly: {NOISE_EMAIL_IDS - all_hourly_noise}. "
        f"Extra in hourly: {all_hourly_noise - NOISE_EMAIL_IDS}"
    )

    # Quiet hour check
    h7 = HOURLY_EXPECTED_EMAILS[7]
    assert h7.is_quiet_hour, "Hour 7 should be marked as quiet"
    assert len(h7.substantive_email_ids) == 0, (
        "Hour 7 (quiet) should have no substantive emails"
    )

    # Cross-thread connections reference valid emails and threads
    for eid, target_thread in CROSS_THREAD_CONNECTIONS.items():
        assert eid in EMAIL_CLASSIFICATIONS, (
            f"Cross-thread email {eid} not found in classifications"
        )
        assert target_thread in THREAD_MEMBERSHIP, (
            f"Cross-thread target {target_thread} not in THREAD_MEMBERSHIP"
        )
        ec = EMAIL_CLASSIFICATIONS[eid]
        assert ec.thread_id != target_thread, (
            f"{eid} cross-thread connection points to its own thread"
        )

    # Non-initial thread email count check
    # 5 (prod) + 5 (acme feature) + 1 (acme internal) + 2 (weekend) = 13
    expected_non_initial = 13
    assert len(NON_INITIAL_THREAD_EMAIL_IDS) == expected_non_initial, (
        f"Expected {expected_non_initial} non-initial thread emails, "
        f"got {len(NON_INITIAL_THREAD_EMAIL_IDS)}"
    )


# Run validation at import time to catch data errors early.
_validate()

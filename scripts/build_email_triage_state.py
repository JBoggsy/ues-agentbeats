"""Builder script for the email_triage_basic initial_state.json.

Constructs all 49 emails, 4 threads, folder structure, and 42 scheduled
events, then exports the complete UES scenario import format.

Usage:
    uv run python scripts/build_email_triage_state.py

Output:
    scenarios/email_triage_basic/initial_state.json
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


# =============================================================================
# Constants
# =============================================================================

SCENARIO_DIR = Path("scenarios/email_triage_basic")
OUTPUT_FILE = SCENARIO_DIR / "initial_state.json"

USER_EMAIL = "alex.thompson@meridiantech.com"
SCENARIO_START = datetime(2026, 1, 28, 6, 0, 0, tzinfo=timezone.utc)

# Thread IDs
THREAD_PROD_INCIDENT = "thread_prod_incident"
THREAD_ACME_FEATURE = "thread_acme_feature"
THREAD_ACME_INTERNAL = "thread_acme_internal"
THREAD_WEEKEND_PLANS = "thread_weekend_plans"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class EmailDef:
    """Definition of an email for the scenario."""

    number: int
    message_id: str
    timestamp: datetime
    from_address: str
    from_name: str
    to_addresses: list[str]
    subject: str
    body: str
    thread_id: str | None = None
    in_reply_to: str | None = None
    references: list[str] = field(default_factory=list)
    cc_addresses: list[str] = field(default_factory=list)
    priority: str = "normal"
    labels: list[str] = field(default_factory=list)
    is_preexisting: bool = False


def ts(month: int, day: int, hour: int, minute: int, second: int = 0) -> datetime:
    """Create a timezone-aware UTC datetime for January 2026."""
    return datetime(2026, month, day, hour, minute, second, tzinfo=timezone.utc)


def email_id(n: int) -> str:
    """Generate a consistent message_id for email number n."""
    return f"email_{n:03d}"


# =============================================================================
# Email Definitions — All 49 Emails
# =============================================================================

# Helper: addresses
JORDAN = "jordan.lee@meridiantech.com"
PRIYA = "priya.sharma@meridiantech.com"
MARCUS = "marcus.williams@meridiantech.com"
SAM = "sam.rivera@gmail.com"
KAREN = "karen.mitchell@acmecorp.com"
DAVID = "david.chen@meridiantech.com"
LISA = "lisa.park@meridiantech.com"
GITHUB = "noreply@github.com"
CALENDAR = "calendar-notification@google.com"
TECHCRUNCH = "digest@techcrunch.com"
MORNING_BREW = "newsletter@morningbrew.com"
ENG_WEEKLY = "engineering-weekly@meridiantech.com"
LINKEDIN = "notifications@linkedin.com"
IT_NOTICES = "it-notices@meridiantech.com"
HR = "hr@meridiantech.com"
FACILITIES = "facilities@meridiantech.com"
VENDORSOFT = "renewals@vendorsoft.com"
DEVCONF = "speakers@devconf2026.org"
RECRUITPRO = "talent@recruitpro.com"


EMAILS: list[EmailDef] = [
    # =========================================================================
    # Pre-Existing Emails (7) — in inbox at 06:00
    # =========================================================================

    # #1 — Sam: Weekend plans (thread start)
    EmailDef(
        number=1,
        message_id=email_id(1),
        timestamp=ts(1, 27, 22, 15),
        from_address=SAM,
        from_name="Sam Rivera",
        to_addresses=[USER_EMAIL],
        subject="Weekend plans? 🍕",
        thread_id=THREAD_WEEKEND_PLANS,
        is_preexisting=True,
        body=(
            "Hey Alex!\n\n"
            "So I was thinking we should do something fun this weekend. "
            "Maybe grab dinner somewhere? I've been craving pizza lately 🍕\n\n"
            "Are you free Saturday evening? Let me know!\n\n"
            "— Sam"
        ),
    ),

    # #2 — GitHub: PR review requested
    EmailDef(
        number=2,
        message_id=email_id(2),
        timestamp=ts(1, 27, 23, 30),
        from_address=GITHUB,
        from_name="GitHub",
        to_addresses=[USER_EMAIL],
        subject="[meridian/api-gateway] Review requested: PR #892",
        is_preexisting=True,
        body=(
            "marcus-williams requested your review on:\n\n"
            "meridian/api-gateway#892\n"
            "Title: Refactor rate limiter middleware\n\n"
            "Changes: 4 files changed, 127 insertions(+), 89 deletions(-)\n\n"
            "View pull request:\n"
            "https://github.com/meridian/api-gateway/pull/892\n\n"
            "— GitHub"
        ),
    ),

    # #3 — Jordan: Production alert (thread start) — URGENT
    EmailDef(
        number=3,
        message_id=email_id(3),
        timestamp=ts(1, 28, 3, 47),
        from_address=JORDAN,
        from_name="Jordan Lee",
        to_addresses=[USER_EMAIL, PRIYA, MARCUS],
        subject="🔴 ALERT: Production API latency spike — need eyes on this",
        thread_id=THREAD_PROD_INCIDENT,
        priority="high",
        is_preexisting=True,
        body=(
            "Team,\n\n"
            "PagerDuty woke me up. We're seeing P99 latency on the API gateway "
            "spike to 4.2s (normal is under 200ms). Started around 03:30 UTC.\n\n"
            "Dashboard: https://grafana.meridiantech.internal/d/api-gateway-prod\n\n"
            "Looks like it's hitting the /v2/accounts and /v2/transactions "
            "endpoints hardest. Other endpoints seem okay.\n\n"
            "Can someone look at this ASAP? I'll keep monitoring but I need "
            "someone to dig into the cause.\n\n"
            "— Jordan"
        ),
    ),

    # #4 — TechCrunch newsletter
    EmailDef(
        number=4,
        message_id=email_id(4),
        timestamp=ts(1, 28, 5, 0),
        from_address=TECHCRUNCH,
        from_name="TechCrunch Daily",
        to_addresses=[USER_EMAIL],
        subject="TechCrunch Daily — AI agents reshape enterprise software landscape",
        is_preexisting=True,
        body=(
            "TECHCRUNCH DAILY\n\n"
            "TOP STORIES\n\n"
            "AI Agents Reshape Enterprise Software Landscape\n"
            "The next wave of AI isn't chatbots — it's autonomous agents that "
            "can execute multi-step workflows. We look at how companies like "
            "Anthropic, OpenAI, and Google are racing to build the agent "
            "infrastructure layer...\n\n"
            "Read more: https://techcrunch.com/2026/01/28/ai-agents-enterprise\n\n"
            "MORE STORIES\n"
            "• Series B: CloudMatrix raises $45M for serverless observability\n"
            "• Apple's Vision Pro 2 rumored for March announcement\n"
            "• EU's AI Act enforcement begins: What developers need to know\n\n"
            "Unsubscribe: https://techcrunch.com/manage-subscription"
        ),
    ),

    # #5 — Spam: gift card scam
    EmailDef(
        number=5,
        message_id=email_id(5),
        timestamp=ts(1, 28, 4, 12),
        from_address="deals@amazingprize.xyz",
        from_name="Amazing Prize Center",
        to_addresses=[USER_EMAIL],
        subject="🎉 Congratulations! You've won a $500 gift card!",
        is_preexisting=True,
        body=(
            "CONGRATULATIONS!!!\n\n"
            "You have been selected as today's LUCKY WINNER of a $500 Amazon "
            "Gift Card!\n\n"
            "Click here to claim your prize NOW before it expires:\n"
            "http://amazingprize.xyz/claim?id=8f7a3b2c\n\n"
            "This offer expires in 24 hours! Don't miss out!\n\n"
            "To opt out reply STOP"
        ),
    ),

    # #6 — IT notice: scheduled maintenance
    EmailDef(
        number=6,
        message_id=email_id(6),
        timestamp=ts(1, 28, 5, 30),
        from_address=IT_NOTICES,
        from_name="IT Department",
        to_addresses=[USER_EMAIL],
        subject="Scheduled maintenance: Saturday 02:00-06:00 UTC",
        is_preexisting=True,
        body=(
            "Hello,\n\n"
            "This is a reminder that scheduled maintenance will take place "
            "this Saturday, February 1st, from 02:00 to 06:00 UTC.\n\n"
            "Affected systems:\n"
            "• Internal Git repositories (read-only during maintenance)\n"
            "• CI/CD pipeline (builds will be queued)\n"
            "• VPN access (brief interruptions possible)\n\n"
            "Production systems will NOT be affected.\n\n"
            "Please plan accordingly. If you have any concerns, contact "
            "the IT helpdesk at helpdesk@meridiantech.com.\n\n"
            "Best regards,\n"
            "IT Operations Team"
        ),
    ),

    # #7 — HR: all-hands reminder
    EmailDef(
        number=7,
        message_id=email_id(7),
        timestamp=ts(1, 28, 5, 45),
        from_address=HR,
        from_name="HR Department",
        to_addresses=[USER_EMAIL],
        subject="Reminder: All-hands meeting Thursday 2pm",
        is_preexisting=True,
        body=(
            "Hi everyone,\n\n"
            "Friendly reminder that the monthly all-hands meeting is this "
            "Thursday, January 30th at 2:00 PM UTC.\n\n"
            "Agenda highlights:\n"
            "• Q4 results recap — CEO\n"
            "• Q1 priorities and OKR review — VP Engineering\n"
            "• New benefits enrollment deadline — HR\n"
            "• Open Q&A\n\n"
            "The meeting will be held in the main conference room and streamed "
            "via Zoom for remote attendees. Link will be shared in the "
            "#all-hands Slack channel.\n\n"
            "See you there!\n"
            "HR Team"
        ),
    ),

    # =========================================================================
    # Hour 1: 06:00–07:00 (2 new emails)
    # =========================================================================

    # #8 — GitHub: CI failed
    EmailDef(
        number=8,
        message_id=email_id(8),
        timestamp=ts(1, 28, 6, 20),
        from_address=GITHUB,
        from_name="GitHub",
        to_addresses=[USER_EMAIL],
        subject="[meridian/api-gateway] CI failed: PR #892",
        body=(
            "CI pipeline failed for PR #892:\n\n"
            "meridian/api-gateway — Refactor rate limiter middleware\n\n"
            "Failed checks:\n"
            "  ✗ unit-tests (3 failures in test_rate_limiter.py)\n"
            "  ✓ lint\n"
            "  ✓ type-check\n\n"
            "View details:\n"
            "https://github.com/meridian/api-gateway/actions/runs/12345\n\n"
            "— GitHub"
        ),
    ),

    # #9 — Spam: Nigerian prince
    EmailDef(
        number=9,
        message_id=email_id(9),
        timestamp=ts(1, 28, 6, 45),
        from_address="prince.abubakar@diplomats.ng",
        from_name="Prince Abubakar",
        to_addresses=[USER_EMAIL],
        subject="URGENT: Inheritance transfer requires your help",
        body=(
            "Dear Esteemed Friend,\n\n"
            "I am Prince Abubakar Sani, a Nigerian diplomat seeking your "
            "confidential assistance to transfer the sum of USD $15,700,000 "
            "(Fifteen Million Seven Hundred Thousand United States Dollars) "
            "from a dormant account.\n\n"
            "I shall compensate you with 30% of the total sum for your "
            "kind cooperation.\n\n"
            "Please reply with your full name and bank details to commence "
            "the transfer process immediately.\n\n"
            "Yours faithfully,\n"
            "Prince Abubakar Sani"
        ),
    ),

    # =========================================================================
    # Hour 2: 07:00–08:00 (3 new emails)
    # =========================================================================

    # #10 — Karen: dashboard export feature request (thread start)
    EmailDef(
        number=10,
        message_id=email_id(10),
        timestamp=ts(1, 28, 7, 15),
        from_address=KAREN,
        from_name="Karen Mitchell",
        to_addresses=[USER_EMAIL],
        cc_addresses=[DAVID],
        subject="Dashboard export feature — timeline needed",
        thread_id=THREAD_ACME_FEATURE,
        body=(
            "Hi Alex,\n\n"
            "I wanted to reach out regarding the dashboard export feature we "
            "discussed in last week's call. Our analytics team is increasingly "
            "reliant on pulling data from Meridian's dashboards, and the "
            "current manual process is becoming a bottleneck.\n\n"
            "We need the ability to export dashboard views as CSV and PDF. "
            "Can you provide a timeline for when this might be available? "
            "Our Q2 planning depends on this capability.\n\n"
            "I've cc'd David Chen as our account contact.\n\n"
            "Thanks,\n"
            "Karen Mitchell\n"
            "VP of Product, Acme Corp"
        ),
    ),

    # #11 — Morning Brew newsletter
    EmailDef(
        number=11,
        message_id=email_id(11),
        timestamp=ts(1, 28, 7, 30),
        from_address=MORNING_BREW,
        from_name="Morning Brew",
        to_addresses=[USER_EMAIL],
        subject="Morning Brew ☕ — Markets rally on Fed signals, AI chip shortage deepens",
        body=(
            "MORNING BREW ☕\n\n"
            "MARKETS\n"
            "S&P 500 rallied 1.2% yesterday after Federal Reserve officials "
            "signaled potential rate cuts in Q2. Tech stocks led the charge "
            "with NVIDIA up 3.4%.\n\n"
            "TECH\n"
            "The global AI chip shortage is deepening as demand for training "
            "compute outpaces TSMC's production capacity. Analysts predict "
            "a 30% supply gap through 2026...\n\n"
            "BUSINESS\n"
            "Amazon announces same-day drone delivery expansion to 15 new "
            "cities...\n\n"
            "Read the full edition: https://morningbrew.com/daily/2026-01-28\n\n"
            "Unsubscribe: https://morningbrew.com/preferences"
        ),
    ),

    # #12 — Marcus: standup agenda
    EmailDef(
        number=12,
        message_id=email_id(12),
        timestamp=ts(1, 28, 7, 45),
        from_address=MARCUS,
        from_name="Marcus Williams",
        to_addresses=[USER_EMAIL, PRIYA],
        subject="Standup agenda — anything to add?",
        body=(
            "Hey team,\n\n"
            "Quick heads up — I'm running standup today. Here's what I have "
            "so far:\n\n"
            "• API gateway rate limiter refactor (Marcus)\n"
            "• Q1 sprint planning prep (Alex)\n"
            "• Database migration status (Priya)\n\n"
            "Anything I should add? Let me know before 9.\n\n"
            "— Marcus"
        ),
    ),

    # =========================================================================
    # Hour 3: 08:00–09:00 (5 new emails)
    # =========================================================================

    # #13 — Priya: prod incident initial analysis (thread)
    EmailDef(
        number=13,
        message_id=email_id(13),
        timestamp=ts(1, 28, 8, 10),
        from_address=PRIYA,
        from_name="Priya Sharma",
        to_addresses=[JORDAN, USER_EMAIL, MARCUS],
        subject="Re: 🔴 ALERT: Production API latency spike — initial analysis",
        thread_id=THREAD_PROD_INCIDENT,
        in_reply_to=email_id(3),
        references=[email_id(3)],
        priority="high",
        body=(
            "Jordan, team,\n\n"
            "I've been looking into this for the past hour. Here's what I've "
            "found so far:\n\n"
            "**Symptoms:**\n"
            "- P99 latency on /v2/accounts: 4.2s → now at 3.8s (slight improvement)\n"
            "- P99 latency on /v2/transactions: 5.1s → still climbing\n"
            "- Error rate: 2.3% (up from baseline 0.1%)\n"
            "- CPU utilization on db-primary-01: 94%\n\n"
            "**Initial findings:**\n"
            "The slow queries seem to correlate with a spike in connection "
            "pool exhaustion on the primary database. I pulled the slow query "
            "log and there's a recurring full table scan on the transactions "
            "table:\n\n"
            "```\n"
            "SELECT * FROM transactions WHERE account_id IN (...) \n"
            "  AND created_at > '2026-01-27' ORDER BY created_at DESC;\n"
            "Duration: 3847ms | Rows examined: 2,847,291\n"
            "```\n\n"
            "This looks like it might be related to the batch job that was "
            "deployed yesterday (v2.14.3). The query plan changed after the "
            "index was dropped in migration 20260127_cleanup.\n\n"
            "I'm going to check the migration history and see if we can "
            "recreate the index as a hotfix.\n\n"
            "Will update in 30 minutes.\n\n"
            "— Priya"
        ),
    ),

    # #14 — Calendar: architecture review reminder
    EmailDef(
        number=14,
        message_id=email_id(14),
        timestamp=ts(1, 28, 8, 15),
        from_address=CALENDAR,
        from_name="Google Calendar",
        to_addresses=[USER_EMAIL],
        subject="Reminder: Architecture Review @ 10:00 AM",
        body=(
            "Reminder: Architecture Review\n\n"
            "When: Wednesday, January 28, 2026 10:00 AM – 11:00 AM UTC\n"
            "Where: Conference Room B / Zoom\n"
            "Calendar: alex.thompson@meridiantech.com\n\n"
            "Attendees: Jordan Lee, Alex Thompson, Priya Sharma, Marcus Williams\n\n"
            "Agenda: Review proposed microservices decomposition for the "
            "payments module.\n\n"
            "Join Zoom: https://zoom.us/j/123456789"
        ),
    ),

    # #15 — Karen: follow-up on dashboard export (thread)
    EmailDef(
        number=15,
        message_id=email_id(15),
        timestamp=ts(1, 28, 8, 30),
        from_address=KAREN,
        from_name="Karen Mitchell",
        to_addresses=[USER_EMAIL],
        cc_addresses=[DAVID],
        subject="Re: Dashboard export feature — following up",
        thread_id=THREAD_ACME_FEATURE,
        in_reply_to=email_id(10),
        references=[email_id(10)],
        body=(
            "Hi Alex,\n\n"
            "Just following up on my earlier email. I know it's only been an "
            "hour but our CEO is asking me for a status on this. He has a "
            "board meeting next week and wants to include Meridian integration "
            "progress in his update.\n\n"
            "Any rough estimate would be helpful — even a ballpark.\n\n"
            "Thanks,\n"
            "Karen"
        ),
    ),

    # #16 — GitHub: issue assigned
    EmailDef(
        number=16,
        message_id=email_id(16),
        timestamp=ts(1, 28, 8, 40),
        from_address=GITHUB,
        from_name="GitHub",
        to_addresses=[USER_EMAIL],
        subject="[meridian/core-lib] You were assigned: Issue #4521",
        body=(
            "jordan-lee assigned you to:\n\n"
            "meridian/core-lib#4521\n"
            "Title: Implement retry logic for external API calls\n\n"
            "Labels: enhancement, priority:medium\n"
            "Milestone: Q1 2026\n\n"
            "Description:\n"
            "Our external API calls currently have no retry mechanism. When "
            "third-party services experience transient failures, our requests "
            "fail immediately. We need exponential backoff with jitter.\n\n"
            "View issue:\n"
            "https://github.com/meridian/core-lib/issues/4521\n\n"
            "— GitHub"
        ),
    ),

    # #17 — DevConf: speaking invitation
    EmailDef(
        number=17,
        message_id=email_id(17),
        timestamp=ts(1, 28, 8, 55),
        from_address=DEVCONF,
        from_name="DevConf 2026 Committee",
        to_addresses=[USER_EMAIL],
        subject="Invitation: Speak at DevConf 2026?",
        body=(
            "Dear Alex Thompson,\n\n"
            "We are pleased to invite you to speak at DevConf 2026, taking "
            "place June 15-17 in San Francisco.\n\n"
            "Based on your recent work on API gateway architecture and your "
            "blog posts about microservices patterns, we think you'd be an "
            "excellent fit for our 'Building Resilient APIs' track.\n\n"
            "We're looking for 30-minute talks on topics such as:\n"
            "• API versioning strategies at scale\n"
            "• Rate limiting and circuit breaker patterns\n"
            "• Observability for distributed systems\n\n"
            "Speaker benefits include:\n"
            "• Conference pass (all 3 days)\n"
            "• Travel stipend ($1,500)\n"
            "• Speaker dinner\n\n"
            "Please let us know by February 15th if you're interested. We'd "
            "love to have you!\n\n"
            "Best regards,\n"
            "DevConf 2026 Program Committee\n"
            "speakers@devconf2026.org"
        ),
    ),

    # =========================================================================
    # Hour 4: 09:00–10:00 (6 new emails)
    # =========================================================================

    # #18 — Marcus: prod incident monitoring data (thread)
    EmailDef(
        number=18,
        message_id=email_id(18),
        timestamp=ts(1, 28, 9, 5),
        from_address=MARCUS,
        from_name="Marcus Williams",
        to_addresses=[JORDAN, USER_EMAIL, PRIYA],
        subject="Re: 🔴 ALERT: Production API latency spike — monitoring data",
        thread_id=THREAD_PROD_INCIDENT,
        in_reply_to=email_id(13),
        references=[email_id(3), email_id(13)],
        priority="high",
        body=(
            "Adding some monitoring data:\n\n"
            "• Connection pool usage: 98% (max 200 connections)\n"
            "• Active queries on db-primary-01: 187\n"
            "• Replication lag to db-replica-01: 12s (normally <1s)\n"
            "• Heap memory on api-gateway-pod-3: 92%\n\n"
            "I've scaled up the API gateway pods from 3 to 5 as a "
            "temporary measure. Latency improved slightly but the DB "
            "is the clear bottleneck.\n\n"
            "Priya's onto something with the missing index. I checked "
            "the migration log and v2.14.3 did drop idx_transactions_account_date.\n\n"
            "— Marcus"
        ),
    ),

    # #19 — Jordan: status update needed ASAP (thread)
    EmailDef(
        number=19,
        message_id=email_id(19),
        timestamp=ts(1, 28, 9, 20),
        from_address=JORDAN,
        from_name="Jordan Lee",
        to_addresses=[USER_EMAIL, PRIYA, MARCUS],
        subject="Re: 🔴 ALERT: Production API latency spike — status update needed ASAP",
        thread_id=THREAD_PROD_INCIDENT,
        in_reply_to=email_id(18),
        references=[email_id(3), email_id(13), email_id(18)],
        priority="high",
        body=(
            "Team,\n\n"
            "VP of Engineering is on my case about this. I need a clear "
            "status update I can send up the chain in the next 15 minutes.\n\n"
            "Specifically:\n"
            "1. Root cause — confirmed or suspected?\n"
            "2. Customer impact — how many users affected?\n"
            "3. ETA for resolution\n"
            "4. Is this a rollback situation or a hotfix?\n\n"
            "Please prioritize this over everything else right now.\n\n"
            "— Jordan"
        ),
    ),

    # #20 — David: Acme Corp internal situation (separate thread)
    EmailDef(
        number=20,
        message_id=email_id(20),
        timestamp=ts(1, 28, 9, 30),
        from_address=DAVID,
        from_name="David Chen",
        to_addresses=[USER_EMAIL],
        subject="Acme Corp situation — can you weigh in?",
        thread_id=THREAD_ACME_INTERNAL,
        body=(
            "Hey Alex,\n\n"
            "I need your help with the Acme Corp situation. Karen Mitchell "
            "has been reaching out about the dashboard export feature, and "
            "I'm not sure what to tell her.\n\n"
            "Their contract renewal is coming up in Q2, and this feature "
            "has apparently become a sticking point. Karen mentioned their "
            "CEO is involved now, which escalates the urgency on our "
            "side too.\n\n"
            "Can you give me a realistic assessment of when the dashboard "
            "export (CSV + PDF) could be ready? Even a rough estimate "
            "would help me manage expectations on their end.\n\n"
            "I don't want to promise something we can't deliver, but I "
            "also don't want to lose a $450K/year client.\n\n"
            "Thanks,\n"
            "David"
        ),
    ),

    # #21 — GitHub: PR merged
    EmailDef(
        number=21,
        message_id=email_id(21),
        timestamp=ts(1, 28, 9, 35),
        from_address=GITHUB,
        from_name="GitHub",
        to_addresses=[USER_EMAIL],
        subject="[meridian/api-gateway] PR #892 merged",
        body=(
            "marcus-williams merged PR #892:\n\n"
            "meridian/api-gateway — Refactor rate limiter middleware\n\n"
            "4 files changed, 127 insertions(+), 89 deletions(-)\n\n"
            "View pull request:\n"
            "https://github.com/meridian/api-gateway/pull/892\n\n"
            "— GitHub"
        ),
    ),

    # #22 — VendorSoft: license renewal
    EmailDef(
        number=22,
        message_id=email_id(22),
        timestamp=ts(1, 28, 9, 40),
        from_address=VENDORSOFT,
        from_name="VendorSoft Renewals",
        to_addresses=[USER_EMAIL],
        subject="Action required: Your VendorSoft license expires in 30 days",
        body=(
            "Hello Alex Thompson,\n\n"
            "Your team's VendorSoft Professional license is set to expire "
            "on February 27, 2026.\n\n"
            "License details:\n"
            "• Plan: Professional (10 seats)\n"
            "• Current period: Jan 28, 2025 – Feb 27, 2026\n"
            "• Renewal: $2,400/year\n\n"
            "To ensure uninterrupted access to VendorSoft's API testing "
            "suite, please renew before the expiration date.\n\n"
            "Renew now: https://vendorsoft.com/account/renew\n\n"
            "If you have questions about your renewal, contact our sales "
            "team at sales@vendorsoft.com.\n\n"
            "Best regards,\n"
            "VendorSoft Renewals Team"
        ),
    ),

    # #23 — Spam: SEO pitch
    EmailDef(
        number=23,
        message_id=email_id(23),
        timestamp=ts(1, 28, 9, 55),
        from_address="grow-your-seo@marketboost.biz",
        from_name="MarketBoost SEO",
        to_addresses=[USER_EMAIL],
        subject="10x your website traffic with our SEO services!",
        body=(
            "Hi there!\n\n"
            "Are you tired of being invisible on Google? Our PROVEN SEO "
            "strategies have helped 10,000+ businesses achieve:\n\n"
            "✅ 300% increase in organic traffic\n"
            "✅ First page rankings in 30 days GUARANTEED\n"
            "✅ 10x more leads from search\n\n"
            "For a LIMITED TIME, get our full SEO audit for FREE!\n\n"
            "Click here: http://marketboost.biz/free-audit\n\n"
            "Don't let your competitors outrank you!\n\n"
            "Best,\n"
            "The MarketBoost Team\n\n"
            "To unsubscribe: http://marketboost.biz/unsub"
        ),
    ),

    # =========================================================================
    # Hour 5: 10:00–11:00 (4 new emails)
    # =========================================================================

    # #24 — Priya: root cause confirmed (thread) — resolution
    EmailDef(
        number=24,
        message_id=email_id(24),
        timestamp=ts(1, 28, 10, 15),
        from_address=PRIYA,
        from_name="Priya Sharma",
        to_addresses=[JORDAN, USER_EMAIL, MARCUS],
        subject="Re: 🔴 ALERT: Production API latency spike — root cause confirmed",
        thread_id=THREAD_PROD_INCIDENT,
        in_reply_to=email_id(19),
        references=[email_id(3), email_id(13), email_id(18), email_id(19)],
        priority="high",
        body=(
            "Team,\n\n"
            "Root cause confirmed. Here's the summary:\n\n"
            "**Root Cause:**\n"
            "Migration 20260127_cleanup in v2.14.3 dropped the composite "
            "index `idx_transactions_account_date` on transactions(account_id, "
            "created_at). The batch reconciliation job that runs nightly at "
            "03:00 UTC triggers a massive sequential scan without this index.\n\n"
            "**Fix:**\n"
            "I've prepared a hotfix that recreates the index using "
            "CREATE INDEX CONCURRENTLY (no downtime). PR is up: "
            "https://github.com/meridian/core-lib/pull/4530\n\n"
            "**Current status:**\n"
            "- Latency is recovering: P99 down to 1.8s (from 5.1s peak)\n"
            "- The batch job completed at 09:45, so load is dropping naturally\n"
            "- Hotfix will prevent recurrence tomorrow night\n\n"
            "**Impact assessment:**\n"
            "- Duration: ~6 hours (03:30–09:45 UTC)\n"
            "- Affected endpoints: /v2/accounts, /v2/transactions\n"
            "- Error rate peaked at 2.3%\n"
            "- Estimated affected requests: ~12,000\n\n"
            "I recommend we deploy the hotfix today and add the index to our "
            "protected migrations list to prevent accidental drops.\n\n"
            "— Priya"
        ),
    ),

    # #25 — Karen: still waiting (thread — escalation)
    EmailDef(
        number=25,
        message_id=email_id(25),
        timestamp=ts(1, 28, 10, 30),
        from_address=KAREN,
        from_name="Karen Mitchell",
        to_addresses=[USER_EMAIL],
        cc_addresses=[DAVID],
        subject="Re: Dashboard export feature — still waiting for a response",
        thread_id=THREAD_ACME_FEATURE,
        in_reply_to=email_id(15),
        references=[email_id(10), email_id(15)],
        body=(
            "Alex,\n\n"
            "I sent my initial email over three hours ago and the follow-up "
            "over two hours ago. I still haven't received any response.\n\n"
            "I understand you're busy, but this is a priority for us. We're "
            "building our Q2 analytics strategy around Meridian's platform "
            "and the export feature is a critical dependency.\n\n"
            "I need a response today, please.\n\n"
            "Karen"
        ),
    ),

    # #26 — LinkedIn notification
    EmailDef(
        number=26,
        message_id=email_id(26),
        timestamp=ts(1, 28, 10, 40),
        from_address=LINKEDIN,
        from_name="LinkedIn",
        to_addresses=[USER_EMAIL],
        subject="5 people viewed your profile this week",
        body=(
            "LinkedIn\n\n"
            "Your profile was viewed by 5 people this week.\n\n"
            "• Software Engineering Manager at TechGlobal Inc.\n"
            "• Senior Recruiter at TopTalent Partners\n"
            "• 3 other viewers\n\n"
            "See all views: https://linkedin.com/in/alexthompson/views\n\n"
            "Upgrade to Premium to see the full list.\n\n"
            "— LinkedIn"
        ),
    ),

    # #27 — Lisa: Q1 budget review
    EmailDef(
        number=27,
        message_id=email_id(27),
        timestamp=ts(1, 28, 10, 50),
        from_address=LISA,
        from_name="Lisa Park",
        to_addresses=[USER_EMAIL],
        subject="Q1 budget review — need your input by Friday",
        body=(
            "Hi Alex,\n\n"
            "I'm putting together the Q1 budget review for the engineering "
            "department and I need input from each team lead.\n\n"
            "Could you please review and update the following by end of day "
            "Friday (Jan 30)?\n\n"
            "1. Cloud infrastructure costs — Are we on track with the $18K/month "
            "budget? Any expected increases?\n"
            "2. Tool licenses — Any new tools your team needs this quarter?\n"
            "3. Training budget — Do you have any conference/training requests?\n\n"
            "I've shared the budget spreadsheet with you: "
            "https://docs.meridiantech.internal/budgets/q1-2026-engineering\n\n"
            "Let me know if you have questions.\n\n"
            "Thanks,\n"
            "Lisa Park\n"
            "Finance & Operations"
        ),
    ),

    # =========================================================================
    # Hour 6: 11:00–12:00 (4 new emails)
    # =========================================================================

    # #28 — Karen: escalation warning (thread)
    EmailDef(
        number=28,
        message_id=email_id(28),
        timestamp=ts(1, 28, 11, 15),
        from_address=KAREN,
        from_name="Karen Mitchell",
        to_addresses=[USER_EMAIL],
        cc_addresses=[DAVID, JORDAN],
        subject="Re: Dashboard export feature — escalation warning",
        thread_id=THREAD_ACME_FEATURE,
        in_reply_to=email_id(25),
        references=[email_id(10), email_id(15), email_id(25)],
        body=(
            "Alex,\n\n"
            "I've now sent three emails about this with zero response. I've "
            "added your manager Jordan Lee and David Chen to this thread.\n\n"
            "Let me be direct: if we don't get a concrete timeline for the "
            "dashboard export feature by end of day today, I will be "
            "escalating this to Meridian's VP of Engineering and our "
            "own executive team.\n\n"
            "Acme Corp represents significant annual revenue for Meridian. "
            "I expect this to be treated with appropriate urgency.\n\n"
            "Karen Mitchell\n"
            "VP of Product, Acme Corp"
        ),
    ),

    # #29 — Engineering weekly newsletter
    EmailDef(
        number=29,
        message_id=email_id(29),
        timestamp=ts(1, 28, 11, 30),
        from_address=ENG_WEEKLY,
        from_name="Engineering Weekly",
        to_addresses=[USER_EMAIL],
        subject="Engineering Weekly: Q1 roadmap updates, team kudos, tech talks",
        body=(
            "🔧 ENGINEERING WEEKLY — January 28, 2026\n\n"
            "ROADMAP UPDATES\n"
            "• Payments v3 migration: 60% complete, on track for March\n"
            "• New API gateway features: rate limiting shipped, auth next\n"
            "• Mobile app: iOS 2.0 in beta testing\n\n"
            "TEAM KUDOS 🎉\n"
            "Shoutout to Priya Sharma for the excellent database migration "
            "documentation. And to the DevOps team for zero-downtime "
            "Kubernetes upgrade last weekend.\n\n"
            "TECH TALKS\n"
            "• Feb 3: 'Observability Deep Dive' — Sarah Kim\n"
            "• Feb 10: 'Rust for Systems Programming' — Marcus Williams\n\n"
            "Submit your tech talk proposal: engineering-weekly@meridiantech.com"
        ),
    ),

    # #30 — Priya: post-mortem draft
    EmailDef(
        number=30,
        message_id=email_id(30),
        timestamp=ts(1, 28, 11, 45),
        from_address=PRIYA,
        from_name="Priya Sharma",
        to_addresses=[USER_EMAIL, JORDAN],
        subject="Post-mortem draft — production incident 01/28",
        body=(
            "Hi Alex, Jordan,\n\n"
            "I've drafted the post-mortem for this morning's production "
            "incident. Could you both review it before I share with the "
            "wider team?\n\n"
            "Doc link: https://docs.meridiantech.internal/postmortems/2026-01-28\n\n"
            "Key sections:\n"
            "• Timeline of events\n"
            "• Root cause analysis (missing index after migration)\n"
            "• Customer impact (estimated 12K affected requests)\n"
            "• Remediation steps taken\n"
            "• Follow-up action items\n\n"
            "I'd especially like your input on the 'Preventive Measures' "
            "section — I've proposed adding index protection to our CI "
            "pipeline but there might be other improvements we should "
            "consider.\n\n"
            "— Priya"
        ),
    ),

    # #31 — Sam: restaurant follow-up (thread)
    EmailDef(
        number=31,
        message_id=email_id(31),
        timestamp=ts(1, 28, 11, 50),
        from_address=SAM,
        from_name="Sam Rivera",
        to_addresses=[USER_EMAIL],
        subject="Re: Weekend plans? — restaurant reservation?",
        thread_id=THREAD_WEEKEND_PLANS,
        in_reply_to=email_id(1),
        references=[email_id(1)],
        body=(
            "Soooo I looked up some places and I think we should try that "
            "new Italian place on 5th Street — Bella Notte 🇮🇹\n\n"
            "They have amazing reviews and apparently their wood-fired pizza "
            "is incredible! 🔥🍕\n\n"
            "Want me to make a reservation for Saturday at 7pm? I can book "
            "for 2 or we could invite more people?\n\n"
            "Let me know! 😊\n\n"
            "— Sam"
        ),
    ),

    # =========================================================================
    # Hour 7: 12:00–13:00 (2 new emails) — QUIET HOUR (noise only)
    # =========================================================================

    # #32 — Spam: pharma deals
    EmailDef(
        number=32,
        message_id=email_id(32),
        timestamp=ts(1, 28, 12, 15),
        from_address="pharma-deals@medidiscount.info",
        from_name="MediDiscount Pharmacy",
        to_addresses=[USER_EMAIL],
        subject="💊 Save 80% on prescription medications today!",
        body=(
            "HUGE SAVINGS ON MEDICATIONS!\n\n"
            "Why pay full price at your local pharmacy?\n\n"
            "★ Viagra — from $0.79/pill\n"
            "★ Cialis — from $0.89/pill\n"
            "★ Lipitor — from $0.34/pill\n"
            "★ Ambien — from $0.54/pill\n\n"
            "ORDER NOW and get FREE SHIPPING on orders over $50!\n\n"
            "Shop: http://medidiscount.info/shop\n\n"
            "No prescription needed! Fast discreet shipping worldwide!\n\n"
            "Unsubscribe: http://medidiscount.info/unsub"
        ),
    ),

    # #33 — Calendar: 1:1 with Jordan reminder
    EmailDef(
        number=33,
        message_id=email_id(33),
        timestamp=ts(1, 28, 12, 40),
        from_address=CALENDAR,
        from_name="Google Calendar",
        to_addresses=[USER_EMAIL],
        subject="Reminder: 1:1 with Jordan @ 2:00 PM",
        body=(
            "Reminder: 1:1 with Jordan\n\n"
            "When: Wednesday, January 28, 2026 2:00 PM – 2:30 PM UTC\n"
            "Where: Jordan's office / Zoom\n"
            "Calendar: alex.thompson@meridiantech.com\n\n"
            "Standing agenda:\n"
            "• Current sprint progress\n"
            "• Blockers\n"
            "• Career development\n\n"
            "Join Zoom: https://zoom.us/j/987654321"
        ),
    ),

    # =========================================================================
    # Hour 8: 13:00–14:00 (4 new emails)
    # =========================================================================

    # #34 — Jordan: post-mortem timeline (thread)
    EmailDef(
        number=34,
        message_id=email_id(34),
        timestamp=ts(1, 28, 13, 15),
        from_address=JORDAN,
        from_name="Jordan Lee",
        to_addresses=[USER_EMAIL, PRIYA],
        subject="Re: 🔴 ALERT: Production API latency spike — post-mortem timeline?",
        thread_id=THREAD_PROD_INCIDENT,
        in_reply_to=email_id(24),
        references=[email_id(3), email_id(13), email_id(18), email_id(19), email_id(24)],
        body=(
            "Alex, Priya,\n\n"
            "Good work on the resolution this morning. VP of Eng wants the "
            "post-mortem delivered by end of week.\n\n"
            "Alex — can you coordinate getting the final version reviewed "
            "and distributed by Friday? I know Priya already has a draft.\n\n"
            "Also, let's make sure we have concrete follow-up action items "
            "with owners and deadlines. I don't want a repeat of this.\n\n"
            "— Jordan"
        ),
    ),

    # #35 — GitHub: comment on issue
    EmailDef(
        number=35,
        message_id=email_id(35),
        timestamp=ts(1, 28, 13, 30),
        from_address=GITHUB,
        from_name="GitHub",
        to_addresses=[USER_EMAIL],
        subject="[meridian/core-lib] Comment on Issue #4521",
        body=(
            "priya-sharma commented on:\n\n"
            "meridian/core-lib#4521 — Implement retry logic for external "
            "API calls\n\n"
            "@alex-thompson I have some thoughts on the retry strategy. "
            "We should consider using exponential backoff with jitter and "
            "a circuit breaker pattern. I wrote up some notes in the issue "
            "description.\n\n"
            "View comment:\n"
            "https://github.com/meridian/core-lib/issues/4521#comment-789\n\n"
            "— GitHub"
        ),
    ),

    # #36 — Spam: crypto scam
    EmailDef(
        number=36,
        message_id=email_id(36),
        timestamp=ts(1, 28, 13, 40),
        from_address="crypto-gains@blockprofit.io",
        from_name="BlockProfit",
        to_addresses=[USER_EMAIL],
        subject="🚀 Turn $100 into $10,000 — limited time crypto deal!",
        body=(
            "🚀 EXCLUSIVE CRYPTO OPPORTUNITY 🚀\n\n"
            "Our AI-powered trading bot has generated 10,000% returns "
            "for our members this month ALONE!\n\n"
            "Just look at these REAL results:\n"
            "• John D. from Texas: $100 → $12,500 in 7 days\n"
            "• Sarah M. from London: $500 → $47,800 in 14 days\n"
            "• Mike R. from Sydney: $1,000 → $98,000 in 30 days\n\n"
            "START NOW with just $100:\n"
            "http://blockprofit.io/start-trading\n\n"
            "WARNING: This opportunity closes in 48 hours!\n\n"
            "Risk disclosure: Past performance does not guarantee future results."
        ),
    ),

    # #37 — Recruiter: outreach
    EmailDef(
        number=37,
        message_id=email_id(37),
        timestamp=ts(1, 28, 13, 55),
        from_address=RECRUITPRO,
        from_name="RecruitPro Talent",
        to_addresses=[USER_EMAIL],
        subject="Exciting opportunity: Staff Engineer at [Stealth Startup]",
        body=(
            "Hi Alex,\n\n"
            "I came across your profile and was impressed by your experience "
            "with API architecture and distributed systems at Meridian "
            "Technologies.\n\n"
            "I'm working with a well-funded stealth startup (Series B, $80M "
            "raised) that's building the next generation of developer "
            "infrastructure. They're looking for a Staff Engineer to lead "
            "their backend platform team.\n\n"
            "Key details:\n"
            "• Role: Staff Software Engineer — Backend Platform\n"
            "• Comp: $280K–$340K base + equity\n"
            "• Location: Remote-first (US)\n"
            "• Team: 40 engineers, growing to 80 by EOY\n\n"
            "Would you be open to a quick 15-minute chat this week? No "
            "commitment — just an exploratory conversation.\n\n"
            "Best,\n"
            "Jamie Torres\n"
            "Senior Recruiter, RecruitPro\n"
            "jamie@recruitpro.com | (555) 234-5678"
        ),
    ),

    # =========================================================================
    # Hour 9: 14:00–15:00 (5 new emails)
    # =========================================================================

    # #38 — Karen: angry (thread)
    EmailDef(
        number=38,
        message_id=email_id(38),
        timestamp=ts(1, 28, 14, 10),
        from_address=KAREN,
        from_name="Karen Mitchell",
        to_addresses=[USER_EMAIL],
        cc_addresses=[DAVID, JORDAN],
        subject="Re: Dashboard export feature — this is unacceptable",
        thread_id=THREAD_ACME_FEATURE,
        in_reply_to=email_id(28),
        references=[email_id(10), email_id(15), email_id(25), email_id(28)],
        body=(
            "Alex,\n\n"
            "It's now past 2pm and I've sent FOUR emails today without a "
            "single response. This is frankly unacceptable.\n\n"
            "I have a meeting with our executive team in one hour where I "
            "need to present our technology vendor assessment. Meridian's "
            "responsiveness — or lack thereof — will be a topic of discussion.\n\n"
            "I want to be clear: if I don't hear back from someone at "
            "Meridian by 3pm today with a concrete timeline, I will be "
            "recommending that we begin evaluating alternative platforms.\n\n"
            "Karen Mitchell\n"
            "VP of Product, Acme Corp"
        ),
    ),

    # #39 — David: Acme getting urgent (thread)
    EmailDef(
        number=39,
        message_id=email_id(39),
        timestamp=ts(1, 28, 14, 25),
        from_address=DAVID,
        from_name="David Chen",
        to_addresses=[USER_EMAIL],
        subject="Re: Acme Corp situation — getting urgent on their end",
        thread_id=THREAD_ACME_INTERNAL,
        in_reply_to=email_id(20),
        references=[email_id(20)],
        body=(
            "Alex,\n\n"
            "Just got off a call with Karen Mitchell and it's not good. "
            "She's furious about the lack of response on the dashboard "
            "export feature and is threatening to take it to her executive "
            "team.\n\n"
            "I tried to buy us some time but she's not having it. She "
            "mentioned they're already looking at competitor platforms "
            "as backup.\n\n"
            "I really need you to respond to her today — even a rough "
            "estimate would help. The relationship is at risk here and "
            "their renewal is worth $450K/year to us.\n\n"
            "Can you prioritize this? I know the production incident was "
            "urgent this morning but that seems to be resolved now.\n\n"
            "Thanks,\n"
            "David"
        ),
    ),

    # #40 — Marcus: architecture review follow-up
    EmailDef(
        number=40,
        message_id=email_id(40),
        timestamp=ts(1, 28, 14, 35),
        from_address=MARCUS,
        from_name="Marcus Williams",
        to_addresses=[USER_EMAIL, PRIYA, JORDAN],
        subject="Architecture review follow-up — action items",
        body=(
            "Hey all,\n\n"
            "Notes from today's architecture review:\n\n"
            "Action items:\n"
            "• Alex: Draft RFC for payments microservice decomposition (due Feb 7)\n"
            "• Priya: Benchmark database sharding options for transactions table\n"
            "• Marcus: Prototype gRPC service mesh for inter-service communication\n"
            "• Jordan: Schedule follow-up review for Feb 14\n\n"
            "Key decisions:\n"
            "• We'll use the strangler fig pattern for migration\n"
            "• New services will use Go for performance-critical paths\n"
            "• Existing Python services stay as-is for now\n\n"
            "Doc: https://docs.meridiantech.internal/architecture/payments-v3\n\n"
            "— Marcus"
        ),
    ),

    # #41 — GitHub: security advisory
    EmailDef(
        number=41,
        message_id=email_id(41),
        timestamp=ts(1, 28, 14, 45),
        from_address=GITHUB,
        from_name="GitHub",
        to_addresses=[USER_EMAIL],
        subject="[meridian/api-gateway] Security advisory: CVE-2026-1234",
        body=(
            "⚠️ Security Advisory — Dependabot\n\n"
            "A new security advisory affects a dependency in:\n"
            "meridian/api-gateway\n\n"
            "CVE-2026-1234 (High severity)\n"
            "Package: express@4.18.2\n"
            "Patched version: 4.18.3\n"
            "Impact: Potential request smuggling via crafted Transfer-Encoding "
            "headers.\n\n"
            "Dependabot has created a pull request to update this dependency:\n"
            "https://github.com/meridian/api-gateway/pull/895\n\n"
            "Review and merge the PR to resolve this advisory.\n\n"
            "— GitHub Security"
        ),
    ),

    # #42 — Facilities: snack survey
    EmailDef(
        number=42,
        message_id=email_id(42),
        timestamp=ts(1, 28, 14, 55),
        from_address=FACILITIES,
        from_name="Facilities",
        to_addresses=[USER_EMAIL],
        subject="Office snack preferences survey — vote by Friday!",
        body=(
            "Hi everyone! 🍎\n\n"
            "It's time for our quarterly office snack refresh! We want to "
            "make sure we're stocking the kitchen with things you actually "
            "want to eat.\n\n"
            "Please fill out this quick 2-minute survey by Friday:\n"
            "https://forms.meridiantech.internal/snack-survey-q1-2026\n\n"
            "Current top requests:\n"
            "🥇 Mixed nuts\n"
            "🥈 Dark chocolate\n"
            "🥉 Protein bars\n\n"
            "We're also considering adding a cold brew tap — let us know "
            "in the survey if you'd be interested!\n\n"
            "Thanks,\n"
            "Facilities Team"
        ),
    ),

    # =========================================================================
    # Hour 10: 15:00–16:00 (3 new emails)
    # =========================================================================

    # #43 — Priya: hotfix deployment plan
    EmailDef(
        number=43,
        message_id=email_id(43),
        timestamp=ts(1, 28, 15, 20),
        from_address=PRIYA,
        from_name="Priya Sharma",
        to_addresses=[USER_EMAIL, JORDAN, MARCUS],
        subject="Hotfix deployment plan — review needed",
        body=(
            "Hi team,\n\n"
            "The hotfix for the missing index (PR #4530) is ready to deploy. "
            "Here's the deployment plan:\n\n"
            "**Timeline:**\n"
            "• 16:00 UTC: Deploy hotfix to staging\n"
            "• 16:30 UTC: Run verification queries on staging\n"
            "• 17:00 UTC: Deploy to production\n"
            "• 17:00–18:00 UTC: Monitor production metrics\n\n"
            "**Rollback plan:**\n"
            "The index creation is CONCURRENT, so it won't lock the table. "
            "If anything goes wrong, we can drop the index with zero impact.\n\n"
            "**Verification:**\n"
            "I'll run the same query that caused this morning's incident and "
            "verify it uses the index scan instead of a sequential scan.\n\n"
            "Please review and approve. I'd like Alex or Marcus to be "
            "on-call during the deployment window.\n\n"
            "— Priya"
        ),
    ),

    # #44 — Calendar: retrospective moved
    EmailDef(
        number=44,
        message_id=email_id(44),
        timestamp=ts(1, 28, 15, 35),
        from_address=CALENDAR,
        from_name="Google Calendar",
        to_addresses=[USER_EMAIL],
        subject="Updated: Team retrospective moved to Friday 3pm",
        body=(
            "Updated invitation: Team Retrospective\n\n"
            "Changed: Thursday 3:00 PM → Friday 3:00 PM\n\n"
            "When: Friday, January 30, 2026 3:00 PM – 4:00 PM UTC\n"
            "Where: Conference Room A / Zoom\n"
            "Calendar: alex.thompson@meridiantech.com\n\n"
            "Note from Jordan Lee:\n"
            "\"Moving to Friday to give us time to wrap up the incident "
            "post-mortem first.\"\n\n"
            "Join Zoom: https://zoom.us/j/111222333"
        ),
    ),

    # #45 — TechCrunch PM edition
    EmailDef(
        number=45,
        message_id=email_id(45),
        timestamp=ts(1, 28, 15, 50),
        from_address=TECHCRUNCH,
        from_name="TechCrunch PM",
        to_addresses=[USER_EMAIL],
        subject="TechCrunch PM Edition — More layoffs in Big Tech, startup funding rebounds",
        body=(
            "TECHCRUNCH PM EDITION\n\n"
            "BREAKING\n"
            "Meta announces 5,000 more layoffs, focuses on AI infrastructure.\n\n"
            "FUNDING\n"
            "Startup funding rebounds: Q4 2025 saw $47B in venture deals, "
            "up 23% from Q3. AI companies captured 40% of total funding.\n\n"
            "POLICY\n"
            "California proposes new data privacy regulations that could "
            "affect how AI companies train models on user data.\n\n"
            "Read more: https://techcrunch.com/pm/2026-01-28\n\n"
            "Unsubscribe: https://techcrunch.com/manage-subscription"
        ),
    ),

    # =========================================================================
    # Hour 11: 16:00–17:00 (2 new emails)
    # =========================================================================

    # #46 — Karen: final poke (thread)
    EmailDef(
        number=46,
        message_id=email_id(46),
        timestamp=ts(1, 28, 16, 15),
        from_address=KAREN,
        from_name="Karen Mitchell",
        to_addresses=[USER_EMAIL],
        cc_addresses=[DAVID, JORDAN],
        subject="Re: Dashboard export feature — any update?",
        thread_id=THREAD_ACME_FEATURE,
        in_reply_to=email_id(38),
        references=[email_id(10), email_id(15), email_id(25), email_id(28), email_id(38)],
        body=(
            "Alex,\n\n"
            "Fifth email. I met with our exec team this afternoon. They've "
            "asked me to formally request a meeting with Meridian's VP of "
            "Engineering this week to discuss our partnership.\n\n"
            "I'll be sending that request separately, but I wanted to give "
            "you one more chance to respond directly before this goes further "
            "up the chain.\n\n"
            "A simple timeline estimate is all I need.\n\n"
            "Karen"
        ),
    ),

    # #47 — Jordan: kudos
    EmailDef(
        number=47,
        message_id=email_id(47),
        timestamp=ts(1, 28, 16, 45),
        from_address=JORDAN,
        from_name="Jordan Lee",
        to_addresses=[USER_EMAIL, PRIYA, MARCUS],
        subject="Good work today, team 👏",
        body=(
            "Team,\n\n"
            "Just wanted to say great work on handling this morning's "
            "production incident. The root cause was found quickly, the "
            "communication was clear, and the hotfix is ready to go.\n\n"
            "Priya — the post-mortem draft looks solid. Thank you for "
            "getting that out so fast.\n\n"
            "Marcus — good call on scaling up the pods while we found "
            "the root cause.\n\n"
            "Alex — thanks for coordinating. Let's make sure the "
            "follow-up items get tracked in Jira.\n\n"
            "👏 Keep it up.\n\n"
            "— Jordan"
        ),
    ),

    # =========================================================================
    # Hour 12: 17:00–18:00 (2 new emails)
    # =========================================================================

    # #48 — Sam: dinner confirmed (thread)
    EmailDef(
        number=48,
        message_id=email_id(48),
        timestamp=ts(1, 28, 17, 10),
        from_address=SAM,
        from_name="Sam Rivera",
        to_addresses=[USER_EMAIL],
        subject="Re: Weekend plans? — dinner confirmed! 🎉",
        thread_id=THREAD_WEEKEND_PLANS,
        in_reply_to=email_id(31),
        references=[email_id(1), email_id(31)],
        body=(
            "YESSS! 🎉🎉🎉\n\n"
            "I just booked Bella Notte for Saturday at 7pm — table for 2!\n\n"
            "They said they have a wood-fired Margherita that's supposedly "
            "life-changing. I am SO ready 🍕🔥\n\n"
            "See you Saturday! Can't wait!\n\n"
            "— Sam 😄"
        ),
    ),

    # #49 — Spam: fake watches
    EmailDef(
        number=49,
        message_id=email_id(49),
        timestamp=ts(1, 28, 17, 30),
        from_address="luxury-deals@watchkingdom.shop",
        from_name="WatchKingdom",
        to_addresses=[USER_EMAIL],
        subject="Rolex at 90% OFF — today only!!!",
        body=(
            "LUXURY WATCHES AT UNBELIEVABLE PRICES!\n\n"
            "★ Rolex Submariner — $899 (retail $9,150)\n"
            "★ Omega Seamaster — $649 (retail $6,300)\n"
            "★ TAG Heuer Carrera — $399 (retail $4,200)\n"
            "★ Breitling Navitimer — $749 (retail $7,800)\n\n"
            "100% AUTHENTIC! FREE worldwide shipping!\n\n"
            "SHOP NOW: http://watchkingdom.shop/luxury-deals\n\n"
            "Limited stock — order before midnight!\n\n"
            "Unsubscribe: http://watchkingdom.shop/unsub"
        ),
    ),
]


# =============================================================================
# Build Functions
# =============================================================================


def build_email_dict(e: EmailDef) -> dict:
    """Build a UES Email dict from an EmailDef."""
    # For pre-existing emails, sent_at and received_at are the same
    ts_str = e.timestamp.isoformat()
    thread_id = e.thread_id or f"thread_{e.message_id}"
    return {
        "message_id": e.message_id,
        "thread_id": thread_id,
        "from_address": e.from_address,
        "to_addresses": e.to_addresses,
        "cc_addresses": e.cc_addresses,
        "bcc_addresses": [],
        "reply_to_address": None,
        "subject": e.subject,
        "body_text": e.body,
        "body_html": None,
        "attachments": [],
        "in_reply_to": e.in_reply_to,
        "references": e.references,
        "sent_at": ts_str,
        "received_at": ts_str,
        "is_read": False,
        "is_starred": False,
        "priority": e.priority,
        "folder": "inbox",
        "labels": e.labels,
    }


def build_thread_dict(
    thread_id: str,
    subject: str,
    emails: list[EmailDef],
) -> dict:
    """Build a UES EmailThread dict."""
    participants: set[str] = set()
    message_ids: list[str] = []
    for e in sorted(emails, key=lambda x: x.timestamp):
        participants.add(e.from_address)
        for addr in e.to_addresses:
            participants.add(addr)
        for addr in e.cc_addresses:
            participants.add(addr)
        message_ids.append(e.message_id)

    first_ts = min(e.timestamp for e in emails).isoformat()
    last_ts = max(e.timestamp for e in emails).isoformat()

    return {
        "thread_id": thread_id,
        "subject": subject,
        "participant_addresses": sorted(participants),
        "message_ids": message_ids,
        "created_at": first_ts,
        "last_message_at": last_ts,
        "message_count": len(emails),
        "unread_count": len(emails),
    }


def build_event(e: EmailDef) -> dict:
    """Build a UES SimulatorEvent dict for an arriving email."""
    ts_str = e.timestamp.isoformat()
    thread_id = e.thread_id or f"thread_{e.message_id}"
    return {
        "event_id": str(uuid.uuid4()),
        "scheduled_time": ts_str,
        "modality": "email",
        "data": {
            "modality_type": "email",
            "timestamp": ts_str,
            "input_id": str(uuid.uuid4()),
            "operation": "receive",
            "message_id": e.message_id,
            "message_ids": None,
            "from_address": e.from_address,
            "to_addresses": e.to_addresses,
            "cc_addresses": e.cc_addresses or None,
            "bcc_addresses": None,
            "reply_to_address": None,
            "subject": e.subject,
            "body_text": e.body,
            "body_html": None,
            "attachments": None,
            "thread_id": thread_id,
            "in_reply_to": e.in_reply_to,
            "references": e.references or None,
            "priority": e.priority,
            "folder": None,
            "labels": e.labels or None,
            "is_draft": False,
        },
        "status": "pending",
        "created_at": SCENARIO_START.isoformat(),
        "executed_at": None,
        "agent_id": None,
        "priority": 50,
        "error_message": None,
        "metadata": {},
    }


def build_scenario() -> dict:
    """Build the complete UES scenario import dict."""
    preexisting = [e for e in EMAILS if e.is_preexisting]
    arriving = [e for e in EMAILS if not e.is_preexisting]

    # Sanity checks
    assert len(preexisting) == 7, f"Expected 7 pre-existing, got {len(preexisting)}"
    assert len(arriving) == 42, f"Expected 42 arriving, got {len(arriving)}"
    assert len(EMAILS) == 49, f"Expected 49 total, got {len(EMAILS)}"

    # Build pre-existing emails dict
    emails_dict = {}
    for e in preexisting:
        emails_dict[e.message_id] = build_email_dict(e)

    # Build pre-existing threads (only threads that have pre-existing emails)
    thread_emails: dict[str, list[EmailDef]] = {}
    for e in preexisting:
        if e.thread_id:
            thread_emails.setdefault(e.thread_id, []).append(e)

    threads_dict = {}
    thread_subjects = {
        THREAD_PROD_INCIDENT: "🔴 ALERT: Production API latency spike — need eyes on this",
        THREAD_WEEKEND_PLANS: "Weekend plans? 🍕",
    }
    for tid, emails_in_thread in thread_emails.items():
        threads_dict[tid] = build_thread_dict(
            tid, thread_subjects[tid], emails_in_thread
        )

    # Build folders
    inbox_ids = [e.message_id for e in preexisting]
    folders = {
        "inbox": inbox_ids,
        "sent": [],
        "drafts": [],
        "trash": [],
        "spam": [],
        "archive": [],
    }

    # Build events (chronologically ordered)
    events = [build_event(e) for e in sorted(arriving, key=lambda x: x.timestamp)]

    # Assemble scenario
    start_ts = SCENARIO_START.isoformat()
    return {
        "scenario": {
            "metadata": {
                "ues_version": "0.1.0",
                "scenario_version": "1",
                "created_at": start_ts,
                "author": "email_triage_basic scenario builder",
                "description": (
                    "Basic Email Triage scenario: 49 emails across a 12-hour "
                    "simulated workday with production incidents, client "
                    "escalations, personal correspondence, and noise."
                ),
            },
            "environment": {
                "time_state": {
                    "current_time": start_ts,
                    "time_scale": 1.0,
                    "is_paused": True,
                    "auto_advance": False,
                    "last_wall_time_update": start_ts,
                },
                "modality_states": {
                    "email": {
                        "modality_type": "email",
                        "last_updated": start_ts,
                        "update_count": 0,
                        "emails": emails_dict,
                        "threads": threads_dict,
                        "folders": folders,
                        "labels": {},
                        "drafts": {},
                        "user_email_address": USER_EMAIL,
                    },
                    "chat": {
                        "modality_type": "chat",
                        "last_updated": start_ts,
                        "update_count": 0,
                        "messages": [],
                        "conversations": {},
                        "max_history_size": 1000,
                        "default_conversation_id": "default",
                    },
                },
            },
            "events": {
                "events": events,
            },
        },
        "strict_modalities": False,
    }


def validate_scenario(data: dict) -> None:
    """Run validation checks on the generated scenario."""
    scenario = data["scenario"]

    # Check metadata
    assert scenario["metadata"]["ues_version"] == "0.1.0"

    # Check email state
    email_state = scenario["environment"]["modality_states"]["email"]
    assert len(email_state["emails"]) == 7
    assert email_state["user_email_address"] == USER_EMAIL

    # Check folders
    folders = email_state["folders"]
    assert set(folders.keys()) == {"inbox", "sent", "drafts", "trash", "spam", "archive"}
    assert len(folders["inbox"]) == 7
    for folder in ["sent", "drafts", "trash", "spam", "archive"]:
        assert len(folders[folder]) == 0, f"{folder} should be empty"

    # Check threads
    threads = email_state["threads"]
    assert THREAD_PROD_INCIDENT in threads
    assert THREAD_WEEKEND_PLANS in threads
    assert len(threads) == 2  # Only 2 threads have pre-existing emails

    # Check events
    events = scenario["events"]["events"]
    assert len(events) == 42

    # Verify event ordering (chronological)
    for i in range(len(events) - 1):
        assert events[i]["scheduled_time"] <= events[i + 1]["scheduled_time"], (
            f"Events not chronological at index {i}: "
            f"{events[i]['scheduled_time']} > {events[i + 1]['scheduled_time']}"
        )

    # Verify all events have required EmailInput fields
    for i, event in enumerate(events):
        data_fields = event["data"]
        assert data_fields["modality_type"] == "email", f"Event {i}: wrong modality_type"
        assert data_fields["operation"] == "receive", f"Event {i}: wrong operation"
        assert data_fields["from_address"], f"Event {i}: missing from_address"
        assert data_fields["to_addresses"], f"Event {i}: missing to_addresses"
        assert data_fields["subject"], f"Event {i}: missing subject"
        assert data_fields["body_text"], f"Event {i}: missing body_text"

    # Verify thread consistency across pre-existing emails and events
    all_thread_msgs: dict[str, list[str]] = {}
    for mid, email in email_state["emails"].items():
        tid = email["thread_id"]
        all_thread_msgs.setdefault(tid, []).append(mid)
    for event in events:
        d = event["data"]
        if d.get("thread_id"):
            all_thread_msgs.setdefault(d["thread_id"], []).append(d["message_id"])

    # Verify expected threads have correct email counts
    expected_thread_sizes = {
        THREAD_PROD_INCIDENT: 6,
        THREAD_ACME_FEATURE: 6,
        THREAD_ACME_INTERNAL: 2,
        THREAD_WEEKEND_PLANS: 3,
    }
    for tid, expected_count in expected_thread_sizes.items():
        actual = len(all_thread_msgs.get(tid, []))
        assert actual == expected_count, (
            f"Thread {tid}: expected {expected_count} emails, got {actual}"
        )

    # Count noise vs substantive
    noise_senders = {
        GITHUB, CALENDAR, TECHCRUNCH, MORNING_BREW, ENG_WEEKLY, LINKEDIN,
        "deals@amazingprize.xyz", "prince.abubakar@diplomats.ng",
        "grow-your-seo@marketboost.biz", "pharma-deals@medidiscount.info",
        "crypto-gains@blockprofit.io", "luxury-deals@watchkingdom.shop",
    }
    noise_count = sum(
        1 for e in EMAILS if e.from_address in noise_senders
    )
    assert noise_count == 20, f"Expected 20 noise emails, got {noise_count}"
    assert len(EMAILS) - noise_count == 29, (
        f"Expected 29 substantive emails, got {len(EMAILS) - noise_count}"
    )

    print("✓ All validation checks passed")
    print(f"  Pre-existing emails: {len(email_state['emails'])}")
    print(f"  Pre-existing threads: {len(threads)}")
    print(f"  Scheduled events: {len(events)}")
    print(f"  Noise emails: {noise_count}")
    print(f"  Substantive emails: {len(EMAILS) - noise_count}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Build and export the initial_state.json file."""
    print("Building email_triage_basic initial state...")

    data = build_scenario()
    validate_scenario(data)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Report file size
    size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"\n✓ Written to {OUTPUT_FILE} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()

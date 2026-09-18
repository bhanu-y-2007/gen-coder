"""
Customer Simulator Agent (Frustration Scale: 1 to 5)

FRUSTRATION LEVEL SYSTEM:
  1 = Very calm / Satisfied
  2 = Slightly concerned
  3 = Moderately frustrated
  4 = Highly frustrated
  5 = Extremely angry / About to escalate

FRUSTRATION DYNAMICS:
  - Clear, correct, helpful, and removes all doubts -> Reduce frustration by 2 to 4 points (e.g. 5->1, 5->2, 4->1, 3->1)
  - Partially helpful but still has some doubt -> Reduce by 1 point only (e.g. 5->4, 4->3, 3->2, 2->1)
  - Neutral / repetitive response -> 0 change
  - Confusing, incomplete, or wrong -> Increase frustration by 1 or 2 points
  - Very bad or ignores the issue -> Increase frustration by 2 or 3 points
"""

import uuid
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set


PERSONAS: Dict[str, Dict[str, Any]] = {
    "polite": {
        "name": "Polite Customer",
        "style": "respectful, formal, cooperative, patient",
        "prefix_calm": "Thank you. ",
        "prefix_frustrated": "I appreciate your response, but ",
        "prefix_angry": "With all due respect, I am very disappointed. "
    },
    "concerned": {
        "name": "Concerned Customer",
        "style": "worried, seeks reassurance and detailed clarification",
        "prefix_calm": "I feel much better knowing that. ",
        "prefix_frustrated": "I'm quite worried about this situation. ",
        "prefix_angry": "This is becoming very concerning for me. "
    },
    "confused": {
        "name": "Confused Customer",
        "style": "uncertain, asks clarifying questions, needs step-by-step guidance",
        "prefix_calm": "Ah, that makes sense now. ",
        "prefix_frustrated": "I'm still a bit confused about this. ",
        "prefix_angry": "I don't understand why this is so complicated. "
    },
    "frustrated": {
        "name": "Frustrated Customer",
        "style": "impatient, curt, expects concrete resolution and timelines",
        "prefix_calm": "Glad we are finally getting this sorted. ",
        "prefix_frustrated": "I need a direct answer here. ",
        "prefix_angry": "I am really frustrated with how this is being handled. "
    },
    "angry": {
        "name": "Angry Customer",
        "style": "firm, demanding, urgent, low tolerance for delays or excuses",
        "prefix_calm": "Thank you for fixing this promptly. ",
        "prefix_frustrated": "This delay is unacceptable. ",
        "prefix_angry": "I expect this resolved immediately without more excuses! "
    },
    "furious": {
        "name": "Furious Customer",
        "style": "extreme dissatisfaction, demands immediate escalation or manager",
        "prefix_calm": "Finally, someone took care of this. ",
        "prefix_frustrated": "I am on the verge of escalating this. ",
        "prefix_angry": "This is completely unacceptable! Connect me to a supervisor right now! "
    },
    "impatient": {
        "name": "Impatient Customer",
        "style": "rushed, brief, hates waiting or long explanations",
        "prefix_calm": "Good, let's wrap this up quickly. ",
        "prefix_frustrated": "Can we speed this up please? ",
        "prefix_angry": "I don't have time for this back and forth! "
    }
}


SCENARIOS: Dict[str, Dict[str, Any]] = {
    "refund_request": {
        "name": "Refund Request",
        "issue": "the damaged product I received (Order #ORD-78421)",
        "goal": "a full refund to original payment method",
        "details": "Item arrived cracked in the packaging 12 days ago."
    },
    "delayed_order": {
        "name": "Delayed Order",
        "issue": "my order #ORD-39215 that has not arrived",
        "goal": "a firm delivery date or immediate replacement/refund",
        "details": "Promised delivery was 4 days ago, tracking has not updated."
    },
    "payment_failure": {
        "name": "Payment Failure",
        "issue": "my payment that keeps declining and card charge issue",
        "goal": "successful payment and immediate service activation",
        "details": "Charged $29.99 but account is still showing inactive/declined."
    },
    "account_issue": {
        "name": "Account Access Issue",
        "issue": "being locked out of my premium account",
        "goal": "my account access and subscription benefits to be restored",
        "details": "Password reset link expired and account locked after 3 attempts."
    },
    "cancellation": {
        "name": "Cancellation Request",
        "issue": "my annual subscription renewal",
        "goal": "confirmation that the subscription is cancelled and charges stopped",
        "details": "Annual plan renewing soon, want to stop auto-billing immediately."
    }
}


EMOTION_MAP = {
    1: "Very calm / Satisfied",
    2: "Slightly concerned",
    3: "Moderately frustrated",
    4: "Highly frustrated",
    5: "Extremely angry / About to escalate"
}


def get_emotion(level: int) -> str:
    level = max(1, min(5, int(level)))
    return EMOTION_MAP[level]


def get_band(level: int) -> str:
    level = max(1, min(5, int(level)))
    if level == 1:
        return "calm"
    elif level == 2:
        return "concerned"
    elif level == 3:
        return "frustrated"
    elif level == 4:
        return "angry"
    else:
        return "furious"


class CustomerSimulator:

    def __init__(
        self,
        persona: str = "frustrated",
        scenario: str = "refund_request",
        frustration_level: int = 3,
        expected_resolution: str = "full_refund",
        session_id: Optional[str] = None,
        use_llm: bool = False,
        **kwargs
    ):
        self.session_id = session_id or str(uuid.uuid4())[:8]

        self.persona_name = persona.lower().strip() if persona else "frustrated"
        self.scenario_name = scenario.lower().strip() if scenario else "refund_request"

        if self.persona_name not in PERSONAS:
            self.persona_name = "frustrated"

        if self.scenario_name not in SCENARIOS:
            self.scenario_name = "refund_request"

        # Frustration scale 1 to 5 (default 3)
        try:
            val = int(frustration_level)
            if val > 5:
                val = max(1, min(5, round(val / 2)))
        except (ValueError, TypeError):
            val = 3

        self.frustration_level = max(1, min(5, val))
        self.expected_resolution = expected_resolution

        self.history: List[Dict[str, Any]] = []
        self.turn_count = 0
        self.finished = False

        # Set of already used messages to ensure no repetition
        self.used_messages: Set[str] = set()

        # Log file
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.log_path = self.log_dir / f"session_{self.session_id}.json"
        self._save_log()

    # ==========================================================
    # SET FRUSTRATION LEVEL
    # ==========================================================
    def set_frustration_level(self, level: int) -> int:
        try:
            val = int(level)
            if val > 5:
                val = max(1, min(5, round(val / 2)))
        except (ValueError, TypeError):
            val = 3
        self.frustration_level = max(1, min(5, val))
        return self.frustration_level

    # ==========================================================
    # START SESSION
    # ==========================================================
    def start(self):
        message = self._generate_customer_message(opening=True)
        self._record("customer", message)
        return self._build_response(
            message,
            extra={
                "analysis": {
                    "quality": "initial_state",
                    "delta": 0,
                    "reason": f"Session initialized at Frustration Level {self.frustration_level}/5 ({get_emotion(self.frustration_level)})."
                }
            }
        )

    # ==========================================================
    # AGENT REPLY & DYNAMIC FRUSTRATION UPDATE
    # ==========================================================
    def respond(self, agent_message: Optional[str]):
        if self.finished:
            return self._build_response(
                "Thank you. This conversation has already been completed.",
                extra={"status": "already_finished"}
            )

        safe_message = str(agent_message) if agent_message is not None else ""

        # Record agent message
        self._record("agent", safe_message)

        # ------------------------------------------------------
        # 1. ANALYZE AGENT REPLY QUALITY & CALCULATE DELTA
        # ------------------------------------------------------
        quality, delta, reason = self._evaluate_agent_reply(safe_message)

        # ------------------------------------------------------
        # 2. UPDATE FRUSTRATION LEVEL (CLAMPED 1 TO 5)
        # ------------------------------------------------------
        old_level = self.frustration_level
        self.frustration_level = max(1, min(5, self.frustration_level + delta))

        # ------------------------------------------------------
        # 3. CHECK FOR RESOLUTION
        # ------------------------------------------------------
        text_lower = safe_message.lower()
        if self._is_resolved(text_lower) and self.frustration_level <= 2:
            self.finished = True
            closing_msg = self._closing_message()
            self._record("customer", closing_msg)
            return self._build_response(
                closing_msg,
                extra={
                    "status": "resolved",
                    "analysis": {
                        "quality": quality,
                        "old_level": old_level,
                        "new_level": self.frustration_level,
                        "delta": delta,
                        "reason": f"{reason} Issue is fully resolved."
                    }
                }
            )

        # ------------------------------------------------------
        # 4. GENERATE NEXT CUSTOMER MESSAGE MATCHING NEW LEVEL
        # ------------------------------------------------------
        customer_msg = self._generate_customer_message(opening=False)
        self._record("customer", customer_msg)

        return self._build_response(
            customer_msg,
            extra={
                "analysis": {
                    "quality": quality,
                    "old_level": old_level,
                    "new_level": self.frustration_level,
                    "delta": delta,
                    "reason": reason
                }
            }
        )

    # ==========================================================
    # EVALUATE AGENT REPLY QUALITY (FRUSTRATION DYNAMICS)
    # ==========================================================
    def _evaluate_agent_reply(self, agent_message: str) -> Tuple[str, int, str]:
        """
        Analyzes the agent reply quality according to core rules:
        - Clear, correct, helpful, and removes all doubts -> Reduce frustration by 2 to 4 points (e.g. 5->1, 5->2, 4->1, 3->1)
        - Partially helpful but still has some doubt -> Reduce by 1 point only
        - Neutral / repetitive response -> 0 change
        - Confusing, incomplete, or wrong -> Increase frustration by 1 or 2 points
        - Very bad or ignores the issue -> Increase frustration by 2 or 3 points
        """
        if not agent_message or not agent_message.strip():
            return "very_bad", 2, "Agent reply is empty or missing."

        raw_text = agent_message.strip()

        # Context check: repetitive responses
        past_agent_messages = [
            m.get("content", "").strip().lower()
            for m in self.history
            if m.get("role") == "agent"
        ]
        if len(past_agent_messages) >= 2 and raw_text.lower() == past_agent_messages[-2]:
            return "repetitive", 0, "Agent repeated previous message verbatim without adding new information."

        # Normalize punctuation and spacing
        normalized = re.sub(r"([.,!?;:])", r" \1 ", raw_text)
        normalized = re.sub(r"\s+", " ", normalized).lower()
        words = normalized.split()
        word_count = len(words)
        text = " " + normalized + " "

        # ------------------------------------------------------
        # 1. VERY BAD / DISMISSIVE / IGNORING SIGNALS (PENALTY)
        # ------------------------------------------------------
        blatant_dismissive = [
            r"\b(not\s+my\s+(problem|job|fault|responsibility))\b",
            r"\b(outside\s+our\s+control)\b",
            r"\b(nothing\s+(i|we)\s+can\s+do)\b",
            r"\b(can'?t\s+help(\s+you)?|cannot\s+help(\s+you)?|unable\s+to\s+help)\b",
            r"\b(don'?t\s+know|do\s+not\s+know|no\s+idea)\b",
            r"\b(refuse\s+to|not\s+going\s+to\s+help)\b",
            r"\b(deal\s+with\s+it)\b",
            r"\b(stop\s+complaining|stop\s+asking)\b",
        ]
        customer_blame = [
            r"\b(you\s+(should|must|need\s+to)\s+have\s+(known|read|checked))\b",
            r"\b(your\s+(fault|mistake|error))\b",
            r"\b(why\s+didn'?t\s+you)\b",
            r"\b(not\s+our\s+problem)\b",
        ]
        stalling_without_help = [
            r"\b(don'?t\s+have\s+(any|an)\s+update)\b",
            r"\b(no\s+update(s)?\s+(available|yet|for\s+you)?)\b",
            r"\b(cannot\s+give\s+(you\s+)?(an\s+update|a\s+date|a\s+time|a\s+timeline))\b",
            r"\b(can'?t\s+(tell|give)\s+you\s+(when|anything))\b",
            r"\b(just\s+wait|wait\s+longer|keep\s+waiting|have\s+to\s+wait)\b",
            r"\b(try\s+again\s+later|call\s+back\s+later|check\s+back\s+later)\b",
            r"\b(please\s+wait(\s+patiently)?)\s*$",
        ]
        policy_refusal = [
            r"\b(company\s+policy\s+(states|says|dictates|forbids)\s+(we\s+can'?t|no))\b",
            r"\b(no\s+refunds?(\s+allowed|\s+ever)?)\b",
            r"\b(final\s+sale|store\s+credit\s+only)\b",
        ]

        if any(re.search(pat, text, re.I) for pat in blatant_dismissive) or any(re.search(pat, text, re.I) for pat in customer_blame):
            delta = 3 if self.frustration_level <= 2 else 2
            return "very_bad", delta, "Agent response was dismissive, hostile, or blamed the customer."

        if any(re.search(pat, text, re.I) for pat in stalling_without_help) or any(re.search(pat, text, re.I) for pat in policy_refusal):
            delta = 2 if self.frustration_level <= 3 else 1
            return "very_bad", delta, "Agent response was unhelpful, stalled without action, or rigidly refused assistance."

        # ------------------------------------------------------
        # 2. CONFUSING, INCOMPLETE, OR WRONG SIGNALS
        # ------------------------------------------------------
        confusing_pats = [
            r"\b(what\s+(is|was)\s+your\s+(issue|problem|order|name)\s+again)\b",
            r"\b(can\s+you\s+repeat\s+everything)\b",
            r"\b(who\s+are\s+you)\b",
            r"\b(i\s+guess|maybe|perhaps|not\s+sure)\b",
        ]
        if any(re.search(pat, text, re.I) for pat in confusing_pats):
            return "confusing_or_incomplete", 1, "Agent asked for already provided info or sounded uncertain."

        if word_count <= 2 and not re.search(r"\b(refunded|resolved|cancelled|unlocked|shipped)\b", text, re.I):
            return "confusing_or_incomplete", 1, "Agent response was too brief and lacked helpful substance."

        # ------------------------------------------------------
        # 3. HELPFUL & RESOLUTION SIGNALS
        # ------------------------------------------------------
        # Sincere apology or empathy
        has_deep_apology = bool(re.search(
            r"\b(sincere(ly)?\s+apolog(y|ize|ise|ies)|deeply\s+apolog(y|ize|ise|ies)|apolog(ize|ise|y|ies)|sorry|my\s+apologies)\b",
            text, re.I
        ))
        has_empathy = has_deep_apology or bool(re.search(r"\b(understand|sorry|apolog|patience|empathize|appreciate)\b", text, re.I))

        # Active ownership / investigation
        has_strong_ownership = bool(re.search(
            r"\b(take\s+(full\s+)?ownership|take\s+full\s+responsibility|personally\s+(ensure|handle|take\s+care|make\s+sure|verify|investigate)|investigat(e|ing|ed)|escalat(ed|ing|e))\b",
            text, re.I
        ))
        has_general_helpful = has_strong_ownership or bool(re.search(r"\b(help|assist|check(ing)?|review(ing)?|look(ing)?\s+into|verify(ing)?|process(ed|ing)?|confirm(ed|ing)?)\b", text, re.I))

        # Confirmed resolution
        has_confirmed_resolution = bool(re.search(
            r"\b(refund\s+(has\s+been|is|will\s+be)?\s*(processed|issued|completed|approved|credited)|"
            r"process(ed)?\s+(your|the)?\s*(full\s+)?refund|"
            r"full\s+refund\s+(has\s+been\s+)?(issued|credited|processed|approved)|"
            r"amount\s+(will\s+be|has\s+been)\s+credited\s+back|"
            r"credited\s+back|"
            r"access\s+(has\s+been|is)?\s*restored|"
            r"unlocked\s+(your|the)?\s*account|"
            r"subscription\s+(has\s+been|is)?\s*cancelled|"
            r"cancellation\s+confirmed|"
            r"replacement\s+(has\s+been|is)?\s*(sent|shipped|dispatched)|"
            r"order\s+(has\s+been\s+)?(delivered|shipped|dispatched)|"
            r"tracking\s+(number|link|details)|"
            r"delivery\s+date\s+is)\b",
            text, re.I
        ))

        # Clear timeline commitment
        has_clear_timeline = bool(re.search(
            r"\b(within\s+\d+\s+(hours?|days?|business\s+days?|minutes?)|by\s+(tomorrow|today|end\s+of\s+day|the\s+end\s+of\s+the\s+week)|immediately|right\s+away|guaranteed)\b",
            text, re.I
        ))

        # ------------------------------------------------------
        # 4. MAP TO DELTA
        # ------------------------------------------------------
        current = self.frustration_level

        # Case A: Clear, correct, helpful, and removes all doubts
        # Must have: confirmed resolution OR (clear timeline + strong ownership/empathy)
        if has_confirmed_resolution or (has_clear_timeline and (has_deep_apology or has_strong_ownership)):
            if current >= 5:
                delta = -4 if (has_deep_apology and has_clear_timeline and has_confirmed_resolution) else -3  # 5 -> 1 or 5 -> 2
            elif current == 4:
                delta = -3  # 4 -> 1
            elif current == 3:
                delta = -2  # 3 -> 1
            elif current == 2:
                delta = -1  # 2 -> 1
            else:
                delta = 0
            return "clear_helpful_removes_doubts", delta, "Agent provided a clear, empathetic, and definitive resolution that removes doubts."

        # Case B: Partially helpful but still has some doubt
        # E.g. basic assistance or polite acknowledgment without complete closure
        if has_general_helpful or has_empathy:
            delta = -1 if current > 1 else 0
            return "partially_helpful", delta, "Agent response was helpful and supportive, but some follow-up remains."

        # Case C: Neutral response (e.g. "Let me see.", "Okay.")
        if re.search(r"\b(ok|okay|let\s+me\s+see|i\s+see|got\s+it|noted|alright)\b", text, re.I):
            return "neutral", 0, "Agent response was neutral without active help or offense."

        # Fallback: minor increase for vague responses
        return "incomplete", 1, "Agent response was vague or did not address the customer's question directly."

    # ==========================================================
    # RESOLUTION CHECK
    # ==========================================================
    def _is_resolved(self, text: str) -> bool:
        checks = {
            "refund_request": [
                "refund has been processed",
                "refund processed",
                "full refund has been issued",
                "full refund",
                "refund completed",
                "credited back to your original payment"
            ],
            "delayed_order": [
                "delivery date",
                "order has arrived",
                "order delivered",
                "delivered",
                "tracking link",
                "replacement has been sent",
                "replacement has been shipped"
            ],
            "payment_failure": [
                "payment is successful",
                "payment succeeded",
                "payment fixed",
                "payment completed",
                "charge corrected",
                "subscription activated"
            ],
            "account_issue": [
                "access restored",
                "account restored",
                "account is unlocked",
                "unlocked",
                "password reset email sent",
                "access has been restored"
            ],
            "cancellation": [
                "cancelled",
                "cancellation confirmed",
                "subscription cancelled",
                "auto-renew stopped",
                "billing stopped"
            ]
        }
        for phrase in checks.get(self.scenario_name, []):
            if phrase in text:
                return True
        return False

    # ==========================================================
    # CUSTOMER MESSAGE GENERATOR (1 TO 5 FRUSTRATION LEVELS)
    # ==========================================================
    def _generate_customer_message(self, opening: bool = False) -> str:
        level = self.frustration_level
        persona = PERSONAS[self.persona_name]
        scenario = SCENARIOS[self.scenario_name]

        # ------------------------------------------------------
        # COMPREHENSIVE MESSAGE POOL (Scenarios x 5 Levels)
        # ------------------------------------------------------
        message_catalog = {
            "refund_request": {
                1: [
                    "Hi, I received my order #ORD-78421 with a damaged item. Could you please assist me with a refund when you have a moment?",
                    "Hello! I wanted to check in about getting a refund for my damaged headphones. I appreciate your help.",
                    "Thank you so much. Could you please confirm the next steps for my refund?",
                    "I really appreciate your assistance. Everything is clear on my end, thank you!",
                    "Thank you for looking into my refund so politely. Let me know if you need any details.",
                    "That sounds wonderful, thank you for guiding me through the refund process.",
                    "I'm very satisfied with how this is being handled. Thank you!",
                    "Great, thank you for clarifying everything about my refund so clearly."
                ],
                2: [
                    "Hello, I'm checking on my refund request for order #ORD-78421. Is there an estimated time for it?",
                    "I'm a little concerned about the status of my refund. Could you please verify if it's being processed?",
                    "Could you clarify the refund steps? I just want to be sure I won't be charged extra fees.",
                    "I haven't seen an update on my damaged item refund yet. Could you please look into it?",
                    "I just want to ensure everything is on track for my refund. Any updates?",
                    "Could you confirm if you received the return details for my refund request?",
                    "I'm hoping to get this sorted soon. Could you check the latest refund status for me?",
                    "Just following up gently on my refund request to see where things stand."
                ],
                3: [
                    "I'm getting frustrated because my refund for the damaged headphones is still pending. When will it be completed?",
                    "This refund has been taking longer than expected. Can you give me a clear update and concrete timeline?",
                    "I've been waiting for my money back on order #ORD-78421. What is the delay?",
                    "I need a clear next step on this refund. Please let me know exactly when it will be processed.",
                    "Waiting without a timeline is frustrating. Can you tell me exactly what is happening with my refund?",
                    "I need a definitive answer regarding my refund today. This has been going on for too long.",
                    "Please let me know who is handling my refund and when I will receive confirmation.",
                    "I am not happy having to follow up multiple times for a damaged product refund."
                ],
                4: [
                    "This refund delay is unacceptable. I received a broken product and I want my money back now!",
                    "I've waited 12 days for this refund and I'm tired of vague answers. When am I getting my money?",
                    "I expect a firm resolution on order #ORD-78421 right now. Stop telling me to wait.",
                    "This is getting ridiculous. Process my refund immediately or let me speak with someone who can.",
                    "I'm extremely dissatisfied with this service. Give me a concrete confirmation of my refund immediately!",
                    "Why is it taking so long to refund a damaged item? Fix this now without further excuses.",
                    "I demand a clear timeline and immediate processing of my refund today!",
                    "I've had enough of these delays. Resolve this refund right away!"
                ],
                5: [
                    "This is completely unacceptable! I demand an immediate full refund or I will dispute this charge with my bank!",
                    "I have reached my limit with your company. Process my refund right now or transfer me to a manager immediately!",
                    "You sent me damaged goods and are refusing to fix it quickly. I demand to speak to a supervisor NOW!",
                    "Enough with the excuses! If my refund is not processed immediately, I am escalating this to consumer protection!",
                    "I am furious. Resolve my refund RIGHT NOW or escalate this to management immediately!",
                    "This is atrocious customer service. I want my $149.99 refunded instantly and a supervisor on the line!",
                    "I refuse to wait any longer. Process the refund immediately or I am taking further action!",
                    "Transfer me to a manager right now. I will not accept any more delays on this refund!"
                ]
            },
            "delayed_order": {
                1: [
                    "Hi, my order #ORD-39215 hasn't arrived yet. Could you kindly check the latest delivery status for me?",
                    "Hello! I noticed my delivery is a bit delayed. Could you share an updated arrival date when convenient?",
                    "Thank you for looking into my delivery. I appreciate your assistance in tracking it.",
                    "Thank you for the update on my shipment. I really appreciate your help.",
                    "That's great news, thank you for tracking down my package so quickly.",
                    "I appreciate you giving me the updated delivery timeline so politely.",
                    "Thank you for keeping me informed about my order status.",
                    "Everything looks good, thank you for your help with this shipment!"
                ],
                2: [
                    "Hello, I'm a bit concerned because my chair order was promised 4 days ago and tracking hasn't updated.",
                    "I need this desk chair for my work setup this week. Could you please check where the carrier is?",
                    "I'm slightly worried my package might be lost. Could you confirm if it's still in transit?",
                    "Could you please verify the current location of order #ORD-39215? The tracking link seems stalled.",
                    "I hope there hasn't been an issue with delivery. Do you have any fresh updates from the carrier?",
                    "I just want to be sure my order will arrive soon. Could you double-check the ETA?",
                    "Tracking hasn't moved in 3 days. Could you check what might be causing the delay?",
                    "Could you provide an updated estimated delivery date? I need to plan around it."
                ],
                3: [
                    "I'm getting frustrated with this delay. My order is 4 days late and I need a firm delivery date now.",
                    "This is delaying my work. Please tell me exactly where my chair is and when it will arrive.",
                    "I've been waiting without any tracking updates. What is being done to expedite my delivery?",
                    "I need a concrete update on order #ORD-39215. Vague estimates are not helping.",
                    "The carrier tracking has not moved. Can you contact them directly and give me a clear answer?",
                    "I need to know if this order is actually coming or if I should request a replacement now.",
                    "Waiting this long without an accurate ETA is really frustrating. Please look into this properly.",
                    "Can you provide a definitive delivery date today? This delay is becoming a serious problem."
                ],
                4: [
                    "This delivery delay is completely unacceptable. Tell me where my order is right now!",
                    "I paid for timely delivery and it's 4 days overdue with no updates. Send a replacement or refund me now!",
                    "Stop giving me generic responses. Where is my chair and when will it be at my door?",
                    "I've waited long enough for order #ORD-39215. I need this escalated and resolved immediately!",
                    "I can't do my work without this item. What immediate compensation or expedited solution are you offering?",
                    "This is terrible logistics. Fix this delivery today or issue a full refund immediately!",
                    "I demand a concrete arrival date right now. No more vague excuses.",
                    "My package is days late and nobody has helped me. Fix this now!"
                ],
                5: [
                    "Where on earth is my order?! This is completely unacceptable and I demand an immediate resolution or manager!",
                    "I am furious about this delivery failure. Either deliver my order today or cancel it with a full refund right now!",
                    "I have had enough of these delays and excuses! Connect me to a supervisor immediately!",
                    "This is ridiculous! 4 days late with zero tracking. I demand immediate escalation to management right now!",
                    "Deliver my package immediately or refund every single penny right now! I am escalating this!",
                    "I want a manager on this chat NOW. Your delivery service has failed completely!",
                    "I will not tolerate another second of waiting. Resolve this shipment or refund me instantly!",
                    "Escalate this to a supervisor right now. I am done dealing with unfulfilled promises!"
                ]
            },
            "payment_failure": {
                1: [
                    "Hi there, I noticed a payment issue with my subscription charge. Could you please help me check it?",
                    "Hello! My card showed a charge but my account isn't activated. Could you kindly assist me?",
                    "Thank you for looking into my billing issue. Please let me know what you find.",
                    "I appreciate your help in getting my payment sorted out so quickly.",
                    "Thank you so much! Everything makes sense now regarding my account billing.",
                    "That's wonderful, thank you for confirming my payment status.",
                    "I really appreciate you resolving this billing inquiry so smoothly.",
                    "Thank you for the quick and polite assistance with my payment!"
                ],
                2: [
                    "Hello, my card was charged $29.99 but my account is still showing inactive. Could you check why?",
                    "I'm a little concerned that I was billed twice or that the payment didn't go through properly.",
                    "Could you please verify if my payment went through? I don't want any service disruption.",
                    "I received a payment declined notice, but my bank says the charge cleared. Can you clarify?",
                    "I'm worried about being charged incorrectly. Could you please review my recent transaction?",
                    "Could you confirm when my Premium Plan will be activated after this payment?",
                    "I just want to ensure my account is safe and the billing is correct. Any updates?",
                    "Can you please check the transaction logs on your end for card ending in 4242?"
                ],
                3: [
                    "I'm getting frustrated because I was charged $29.99 but I still cannot use the service. Fix this please.",
                    "My payment went through on my bank statement, but your system says failed. I need a clear resolution.",
                    "I need this payment glitch resolved today. I cannot keep waiting while my service is locked.",
                    "Why was my card billed if the subscription isn't active? Please give me a concrete explanation.",
                    "I've tried multiple times and it keeps failing. What is your team doing to fix my account billing?",
                    "I need a definitive answer on whether my payment was received or if I was overcharged.",
                    "Waiting without access while you have my money is frustrating. Please resolve this now.",
                    "Please look into my billing records and activate my account immediately."
                ],
                4: [
                    "This payment issue is unacceptable. You took my money and locked me out of the service!",
                    "I have proof of charge from my bank. Activate my account immediately or refund the charge right now!",
                    "I am tired of dealing with this failed payment error. Fix my account access now without delays!",
                    "You've charged my card and given me nothing in return. I demand immediate activation or refund!",
                    "Stop giving me automated excuses. Resolve this billing error right now!",
                    "I shouldn't have to fight to use a service I already paid for. Fix this immediately!",
                    "I demand a supervisor or billing specialist right now to fix this payment error!",
                    "Fix this payment error immediately. This is completely unacceptable service!"
                ],
                5: [
                    "You have charged my card without providing service! This is illegal and I demand an immediate refund or manager!",
                    "I am furious! Refund my $29.99 immediately or fix my account NOW before I report this as fraud!",
                    "I demand to speak to a supervisor right now! You took my money and locked my account!",
                    "This is outrageous! Fix my payment issue instantly or I will file a credit card chargeback immediately!",
                    "I want a manager on this line RIGHT NOW. You have wrongfully taken my money!",
                    "Resolve this fraudulent charge and account lockout immediately or I am taking legal action!",
                    "Transfer me to management right now! I will not tolerate this billing malpractice!",
                    "Refund my money immediately! I am escalating this to payment regulators and your executive team!"
                ]
            },
            "account_issue": {
                1: [
                    "Hi, I seem to be having trouble logging into my account. Could you please help me restore access?",
                    "Hello! My password reset link expired. Could you kindly help me unlock my account?",
                    "Thank you for helping me with my account login. I appreciate your guidance.",
                    "Thank you so much! I appreciate your quick help in recovering my login.",
                    "That worked great, thank you for guiding me through the reset steps.",
                    "I really appreciate you helping me get back into my account so politely.",
                    "Thank you for confirming my account status. Everything is clear now.",
                    "Thank you for the prompt and friendly assistance with my account!"
                ],
                2: [
                    "Hello, my account is locked after 3 attempts and I really need to access my files. Can you help?",
                    "I'm concerned because my subscription is showing Free instead of Premium after the login error.",
                    "Could you please check why my reset email isn't arriving? I've checked my spam folder.",
                    "I'm a bit worried about my account security. Could you confirm my profile details are intact?",
                    "I hope my data is safe. Could you please send a fresh unlock link to my email?",
                    "Could you verify why I'm getting an 'access denied' message on my browser?",
                    "I just want to regain access to my account without losing my settings. Any advice?",
                    "Can you please check the account status for user.support@example.com?"
                ],
                3: [
                    "I'm getting frustrated because I'm still locked out of my account and I have urgent work to do.",
                    "I need access restored immediately. Following the standard reset steps is not working.",
                    "Why is my account still locked? Please give me a direct solution or manual unlock now.",
                    "This lockout is stopping my work. Can you manually unlock my account from your admin panel?",
                    "I've been waiting for a reset link for an hour. What is causing this delay?",
                    "I need a concrete next step to regain access to my paid subscription today.",
                    "Being locked out of my own paid account is very frustrating. Please resolve this.",
                    "Please look into this account lock and unlock it for me right now."
                ],
                4: [
                    "This account lockout is completely unacceptable! I pay for this service and I can't even log in!",
                    "I need my account unlocked right now. Stop sending automated links that don't work!",
                    "I have urgent deadlines and your system has locked me out. Restore my access immediately!",
                    "This is ridiculous. Manually reset my credentials or let me speak with technical support now!",
                    "I am extremely unhappy with this support experience. Fix my login access immediately!",
                    "Why can't your team unlock my account? I need access restored this very minute!",
                    "I demand immediate technical escalation. I cannot afford to be locked out any longer!",
                    "Fix my account right now! I've lost hours of productivity because of this!"
                ],
                5: [
                    "I demand immediate restoration of my account right now! Transfer me to a supervisor immediately!",
                    "I am furious! You have locked me out of my paid data and work. Unlock my account NOW!",
                    "This is completely unacceptable! Connect me to management or tier 2 support this second!",
                    "I am losing business every minute I am locked out. Fix this immediately or face legal escalation!",
                    "I want a manager on this chat RIGHT NOW. Restore my access immediately without any more excuses!",
                    "Unlock my account instantly! I refuse to deal with this gross incompetence any longer!",
                    "Transfer me to a supervisor right now! I will not accept being locked out of what I paid for!",
                    "Resolve this lockout immediately or I am taking this issue directly to executive leadership!"
                ]
            },
            "cancellation": {
                1: [
                    "Hi, I would like to request cancellation of my annual subscription. Could you kindly assist me?",
                    "Hello! I want to confirm the cancellation process for my account when you have a moment.",
                    "Thank you for helping me process the cancellation. I appreciate your support.",
                    "Thank you for confirming the cancellation details so clearly.",
                    "That's very helpful, thank you for guiding me through the cancellation steps.",
                    "I appreciate you handling my cancellation request with such courtesy.",
                    "Thank you for confirming there won't be further charges on my card.",
                    "Thank you so much for the smooth and professional cancellation assistance!"
                ],
                2: [
                    "Hello, I requested a cancellation and want to make sure auto-renew is definitely stopped.",
                    "I'm a little concerned about being billed again next billing cycle. Can you confirm the cutoff date?",
                    "Could you please explain if I'm eligible for a prorated refund for the remaining months?",
                    "I just want to be sure there won't be any surprise renewal charges on my card. Can you check?",
                    "Could you send me an official cancellation confirmation receipt to my email?",
                    "I'm slightly worried my request hasn't gone through. Could you verify the cancellation status?",
                    "Can you confirm what happens to my stored data once the cancellation takes effect?",
                    "Just following up to ensure the auto-billing is completely disabled on my account."
                ],
                3: [
                    "I'm getting frustrated because I asked to cancel and I haven't received confirmation yet.",
                    "I want my subscription cancelled today without being pushed to stay. Please confirm it.",
                    "Please stop giving me retention offers and just confirm that my subscription is cancelled.",
                    "I need a clear written confirmation that my card will not be charged again.",
                    "Why is it so hard to simply cancel a subscription? Finalize the cancellation now please.",
                    "I need a definitive answer that my account will not auto-renew. Please confirm immediately.",
                    "Waiting for a simple cancellation confirmation is frustrating. Please process it now.",
                    "Please execute the cancellation and confirm the final date today."
                ],
                4: [
                    "Cancel my subscription immediately! I do not want any sales pitches or delays!",
                    "I demand immediate cancellation and confirmation. Stop trying to keep me subscribed!",
                    "This cancellation process is unacceptable. Turn off auto-renew and cancel my account right now!",
                    "If you charge my card again after this request, I will dispute it as unauthorized. Cancel it now!",
                    "I have asked repeatedly to cancel. Do it immediately and send me proof!",
                    "I'm not interested in discounts. Process my cancellation this second!",
                    "I demand a confirmation number for my cancellation right now. No more delays!",
                    "Cancel my plan immediately! I have had enough of this runaround!"
                ],
                5: [
                    "Cancel my subscription RIGHT NOW! If I see one more charge, I am filing a fraud complaint!",
                    "I demand an immediate cancellation confirmation and a supervisor on this line right now!",
                    "This is completely unacceptable! Cancel my plan instantly or I am reporting this to consumer protection!",
                    "Stop holding my billing hostage! Cancel my subscription immediately and transfer me to a manager!",
                    "I am furious! Cancel my account right now and confirm the billing termination immediately!",
                    "I will not tolerate another second of being billed. Cancel everything immediately or face legal action!",
                    "Transfer me to a manager right now! I demand instant cancellation of my subscription!",
                    "Cancel my subscription this instant! I am done with your deceptive cancellation tactics!"
                ]
            }
        }

        # Select candidate list for the scenario and current frustration level
        scenario_pool = message_catalog.get(self.scenario_name, message_catalog["refund_request"])
        candidates = scenario_pool.get(level, scenario_pool[3])

        # Pick an unused base message
        chosen_base = None
        for msg in candidates:
            if msg not in self.used_messages:
                chosen_base = msg
                break

        if not chosen_base:
            # Generate dynamically unique message using turn counter
            idx = self.turn_count % len(candidates)
            chosen_base = f"{candidates[idx]} (Follow-up {self.turn_count + 1})"

        self.used_messages.add(chosen_base)

        # Apply persona prefix
        if not opening:
            if level == 1:
                prefix = persona.get("prefix_calm", "")
            elif level <= 3:
                prefix = persona.get("prefix_frustrated", "")
            else:
                prefix = persona.get("prefix_angry", "")

            if prefix and not chosen_base.startswith(prefix.strip()[:10]):
                chosen_message = prefix + chosen_base
            else:
                chosen_message = chosen_base
        else:
            chosen_message = chosen_base

        self.used_messages.add(chosen_message)
        return chosen_message

    # ==========================================================
    # CLOSING MESSAGE
    # ==========================================================
    def _closing_message(self) -> str:
        if self.frustration_level == 1:
            return "Thank you so much for resolving this completely! I really appreciate your excellent and prompt help."
        elif self.frustration_level == 2:
            return "Alright, thank you for the update and getting this sorted out for me. I appreciate your assistance."
        elif self.frustration_level == 3:
            return "Okay, I will accept this resolution for now. Please make sure the promised action is completed on time."
        elif self.frustration_level == 4:
            return "Fine, but I expect the promised resolution to be completed without any further delay."
        else:
            return "I will accept this only if it is completed immediately as promised. If not, I am escalating directly to management."

    # ==========================================================
    # RECORD HISTORY
    # ==========================================================
    def _record(self, role: str, content: str):
        self.turn_count += 1
        self.history.append({
            "role": role,
            "content": content,
            "frustration_level": self.frustration_level if role == "customer" else None,
            "frustration_text": f"Frustration: {self.frustration_level}/5" if role == "customer" else None,
            "emotion": get_emotion(self.frustration_level) if role == "customer" else None
        })
        self._save_log()

    # ==========================================================
    # BUILD RESPONSE
    # ==========================================================
    def _build_response(self, message: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        emotion_label = get_emotion(self.frustration_level)
        data = {
            "session_id": self.session_id,
            "customer_message": message,
            "frustration_level": self.frustration_level,
            "frustration_text": f"Frustration: {self.frustration_level}/5",
            "emotion": {
                "label": emotion_label,
                "intensity": self.frustration_level,
                "level": self.frustration_level,
                "scale": "1-5",
                "emotion": emotion_label
            },
            "persona": self.persona_name,
            "persona_name": PERSONAS.get(self.persona_name, {}).get("name", self.persona_name),
            "scenario": self.scenario_name,
            "scenario_name": SCENARIOS.get(self.scenario_name, {}).get("name", self.scenario_name),
            "finished": self.finished,
            "turn_count": self.turn_count,
            "history": self.history,
            "log_path": str(self.log_path)
        }
        if extra:
            data.update(extra)
        return data

    # ==========================================================
    # GET STATE
    # ==========================================================
    def get_state(self) -> Dict[str, Any]:
        return self._build_response("")

    # ==========================================================
    # SAVE LOG
    # ==========================================================
    def _save_log(self):
        try:
            data = {
                "session_id": self.session_id,
                "persona": self.persona_name,
                "scenario": self.scenario_name,
                "frustration_level": self.frustration_level,
                "frustration_text": f"Frustration: {self.frustration_level}/5",
                "emotion": get_emotion(self.frustration_level),
                "history": self.history,
                "turn_count": self.turn_count,
                "finished": self.finished
            }
            self.log_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass


def create_simulator(**kwargs) -> CustomerSimulator:
    return CustomerSimulator(**kwargs)
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import base64
import json
import os
import random
import re
import uuid
import wave
import threading
import sys
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Awaitable, Dict, List, Optional, Sequence, Tuple
from enum import Enum

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient
from google.cloud import storage

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import (
    LocalSmartTurnAnalyzerV3,
)
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    EndFrame,
    InterruptionFrame,
    LLMRunFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from custom_echo_tts_service import EchoTTSService
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.processors.user_idle_processor import UserIdleProcessor

try:
    from pipecat.audio.filters.krisp_viva_filter import KrispVivaFilter
    _KRISP_VIVA_SUPPORTED = True
except Exception:  # Catch Exception instead of ImportError
    KrispVivaFilter = None  # type: ignore[assignment]
    _KRISP_VIVA_SUPPORTED = False

from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.transcriptions.language import Language

from pipecat.services.google.gemini_live.llm_vertex import (
    GeminiLiveVertexLLMService
)

from pipecat.services.google.gemini_live.llm import (
    GeminiModalities,
    InputParams,
)

from pipecat.utils.text.base_text_filter import BaseTextFilter
from pipecat.services.google.frames import LLMSearchResponseFrame
try:
    from pipecatcloud.agent import (
        PipecatSessionArguments,
        DailySessionArguments,
    )
except ImportError:  # pragma: no cover - fallback for local runner without Pipecat Cloud
    PipecatSessionArguments = RunnerArguments  # type: ignore[misc, assignment]
    DailySessionArguments = RunnerArguments  # type: ignore[misc, assignment]


load_dotenv(override=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger.remove()
logger.add(sys.stderr, level=LOG_LEVEL)

_TMP_GCP_CREDS = Path("/tmp/gcp_sa.json")


def _materialize_gcp_credentials() -> Optional[str]:
    """
    Decode GOOGLE_APPLICATION_CREDENTIALS_B64 and persist it to a temp file.
    No other credential source is supported.
    """
    b64_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_B64")
    if not b64_creds:
        logger.error(
            "GOOGLE_APPLICATION_CREDENTIALS_B64 is required. No other credential source is supported."
        )
        return None

    try:
        raw = base64.b64decode(b64_creds)
        _TMP_GCP_CREDS.write_bytes(raw)
        os.chmod(_TMP_GCP_CREDS, 0o600)
        return str(_TMP_GCP_CREDS)
    except Exception as exc:
        logger.error(
            "Failed to materialize GCP credentials from base64: {}", type(exc).__name__
        )
        return None


_MATERIALIZED_GCP_CREDS = _materialize_gcp_credentials()

_TMP_VERTEX_CREDS = Path("/tmp/gcp_vertex.json")


def _materialize_vertex_credentials() -> Optional[str]:
    """
    Decode GOOGLE_APPLICATION_CREDENTIALS_B64_2 for Vertex AI and persist to a temp file.
    Falls back to vertex.json if env var is not set.
    """
    b64_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_B64_2")
    if not b64_creds:
        # Fall back to local vertex.json if exists
        local_vertex = Path("vertex.json")
        if local_vertex.exists():
            logger.info("Using local vertex.json for Vertex AI credentials")
            return str(local_vertex)
        logger.warning("GOOGLE_APPLICATION_CREDENTIALS_B64_2 not set and vertex.json not found")
        return None

    try:
        raw = base64.b64decode(b64_creds)
        _TMP_VERTEX_CREDS.write_bytes(raw)
        os.chmod(_TMP_VERTEX_CREDS, 0o600)
        logger.info("Materialized Vertex AI credentials from GOOGLE_APPLICATION_CREDENTIALS_B64_2")
        return str(_TMP_VERTEX_CREDS)
    except Exception as exc:
        logger.error(
            "Failed to materialize Vertex credentials from base64: {}", type(exc).__name__
        )
        return None


_MATERIALIZED_VERTEX_CREDS = _materialize_vertex_credentials()

_PATH_SANITIZE_PATTERN = re.compile(r"[^0-9A-Za-z._-]+")


def _sanitize_path_segment(value: Optional[str], fallback: str = "unknown") -> str:
    if not value:
        return fallback
    cleaned = _PATH_SANITIZE_PATTERN.sub("_", value.strip())
    return cleaned or fallback


IST_TIMEZONE = timezone(timedelta(hours=5, minutes=30))

_GCS_CLIENT: Optional[storage.Client] = None
_GCS_CLIENT_LOCK = threading.Lock()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(
            f"Invalid integer for {name}: {value}. Falling back to {default}."
        )
        return default


def _maybe_create_krisp_filter():
    if not _KRISP_VIVA_SUPPORTED or KrispVivaFilter is None:
        return None
    try:
        return KrispVivaFilter()
    except Exception:
        return None


MONGODB_URI = "mongodb+srv://internal:rumi_ai_41921_mongo@cluster0.ds0nghe.mongodb.net/ira_rumik?retryWrites=true&w=majority&appName=Cluster0"
MONGODB_DB_NAME = "ira_rumik"
MONGODB_TIMEOUT_MS = 5000
RUMIK_SUMMARY_LIMIT = 5
SESSION_MAX_DURATION_SECS = _env_int("SESSION_MAX_DURATION_SECS", 8.5 * 60)
SESSION_WARNING_LEAD_SECS = _env_int("SESSION_WARNING_LEAD_SECS", 6)
# TEST_USER_ID_FALLBACK = "68491b6b5923b0eaea6eeddb"  # Commented out - using real user_id only

DEFAULT_ACTIVITY = (
    "I'm quietly trying to follow a beginner tutorial for a new embroidery stitch I"
    " stumbled upon online. It's late, but I just couldn't resist trying it out.really challenging and interesting tho"
)

# ============================================================
# PROMPT CONFIGURATION LOADING
# ============================================================

PROMPT_CONFIG_PATH = os.getenv("IRA_PROMPT_CONFIG", "ira_core_prompt.json")


@lru_cache(maxsize=1)
def _load_prompt_config() -> dict:
    """Load prompt configuration from JSON file with caching."""
    config_path = Path(PROMPT_CONFIG_PATH)
    if not config_path.exists():
        logger.warning("Prompt config not found at {}, using defaults", config_path)
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to load prompt config: {}", e)
        return {}


def _get_prompt_component(path: str, default: str = "") -> str:
    """Get a prompt component by dot-notation path.

    Example: _get_prompt_component("prompts.lifeStoriesComponent.hobbies_description")
    """
    config = _load_prompt_config()
    keys = path.split(".")
    value = config
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key, {})
        else:
            return default
    return value if isinstance(value, str) else default


@lru_cache(maxsize=1)
def _batch_load_prompt_components() -> dict:
    """Batch-load all prompt components needed for voice bot in a single pass.

    Returns a dictionary with all components pre-loaded for performance.
    This avoids repeated JSON traversal for each component.
    """
    config = _load_prompt_config()
    prompts = config.get("prompts", {})

    role = prompts.get("roleComponent", {})
    routing = prompts.get("routingContextComponent", {})
    routing_agent = prompts.get("routingAgent", {})
    life_stories = prompts.get("lifeStoriesComponent", {})
    chain_of_thought = prompts.get("chainOfThoughtComponent", {})

    return {
        # roleComponent
        "master_role_prompt": role.get("master_role_prompt", ""),
        "bigger_message_prompt": role.get("bigger_message_length_prompt", ""),
        "strict_instructions_prompt": role.get("strict_instructions_prompt", ""),
        "user_information_prompt": role.get("user_information_prompt", ""),
        "ask_questions_trigger_prompt": role.get("ask_questions_trigger_prompt", ""),

        # routingContextComponent
        "enhanced_core_identity": routing.get("enhanced_core_identity", ""),
        #"tanmay_prompts": routing.get("tanmay_prompts", ""),
        "greeting": routing.get("Greeting", ""),
        "conversation_flow_prompt": routing.get("conversation_flow_prompt", ""),
        "filler_phrase_prompt": routing.get("filler_phrase_prompt", ""),
        "activity_context_prompt": routing.get("activity_context_prompt", ""),
        "language": routing.get("language", "English"),

        # routingAgent
        "generate_activity_prompt": routing_agent.get("generate_current_activity_prompt", ""),
        "extract_behaviour_instruction": routing_agent.get("extract_behaviour_instruction", ""),
        "extract_long_term_instructions": routing_agent.get("extract_long_term_instructions", ""),
        "extract_user_information": routing_agent.get("extract_user_information", ""),
        "life_agent": routing_agent.get("life_agent", ""),

        # lifeStoriesComponent
        "life_stories": life_stories,
        "enhanced_story_instruction": life_stories.get("enhanced_story_instruction", ""),

        # chainOfThoughtComponent
        "enhanced_examples": chain_of_thought.get("enhanced_examples", []),
    }


def _build_life_stories_context() -> str:
    """Build Ira's detailed life story context from JSON config."""
    config = _load_prompt_config()
    stories = config.get("prompts", {}).get("lifeStoriesComponent", {})

    if not stories:
        return ""

    sections = []

    if stories.get("physical_description"):
        sections.append(f"Physical appearance: {stories['physical_description']}")

    if stories.get("childhood_description"):
        sections.append(f"Childhood: {stories['childhood_description']}")

    if stories.get("education_description"):
        sections.append(f"Education: {stories['education_description']}")

    if stories.get("family_description"):
        sections.append(f"Family: {stories['family_description']}")

    if stories.get("personality_description"):
        sections.append(f"Personality: {stories['personality_description']}")

    if stories.get("romantic_life_description"):
        sections.append(f"Past relationship: {stories['romantic_life_description']}")

    if stories.get("current_life_description"):
        sections.append(f"Current life: {stories['current_life_description']}")

    if stories.get("hobbies_description"):
        sections.append(f"Hobbies: {stories['hobbies_description']}")

    return "\n".join(sections)


def _strip_emojis_and_formatting(text: str) -> str:
    """Remove emojis and special formatting from text for TTS compatibility."""
    # Remove common emoji ranges
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"  # enclosed characters
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols extended
        "\U00002600-\U000026FF"  # misc symbols
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub("", text)
    # Remove asterisks used for emphasis like *laughs*
    text = re.sub(r"\*[^*]+\*", "", text)
    return text.strip()


def _get_chain_of_thought_examples() -> str:
    """Get conversation examples from JSON config for few-shot prompting.

    Strips emojis since these examples are used for voice TTS.
    """
    config = _load_prompt_config()
    examples = config.get("prompts", {}).get("chainOfThoughtComponent", {}).get("enhanced_examples", [])

    if not examples:
        return ""

    formatted = []
    for example in examples[:5]:  # Limit to 5 examples
        conv = example.get("conversation", [])
        lines = []
        for turn in conv:
            if "USER" in turn:
                user_text = _strip_emojis_and_formatting(turn["USER"])
                if user_text:
                    lines.append(f"User: {user_text}")
            elif "IRA" in turn:
                response = _strip_emojis_and_formatting(turn["IRA"].get("response", ""))
                if response:
                    lines.append(f"Ira: {response}")
        if lines:
            formatted.append("\n".join(lines))

    return "\n---\n".join(formatted)


def _get_activity_prompt_template() -> str:
    """Get the activity generation prompt from JSON config."""
    return _get_prompt_component(
        "prompts.routingAgent.generate_current_activity_prompt",
        default=""
    )


# ============================================================
# BEHAVIOR ROUTING
# ============================================================


class ConversationStyle(str, Enum):
    """Adaptive conversation styles based on user mood and context."""
    FUN = "fun"
    EMPATHETIC = "empathetic"
    INFORMATIVE = "informative"
    CASUAL = "casual"
    ENCOURAGING = "encouraging"


# Keywords that suggest different conversation styles
_STYLE_KEYWORDS = {
    ConversationStyle.EMPATHETIC: [
        "sad", "upset", "angry", "stressed", "anxious", "worried", "crying",
        "depressed", "lonely", "hurt", "frustrated", "tired", "exhausted",
        "problem", "issue", "help",
    ],
    ConversationStyle.FUN: [
        "lol", "haha", "joke", "funny", "meme", "comedy", "banter", "chill",
    ],
    ConversationStyle.INFORMATIVE: [
        "explain", "what is", "how to", "tell me about", "teach",
        "define", "describe", "help me understand",
    ],
    ConversationStyle.ENCOURAGING: [
        "interview", "exam", "test", "nervous", "scared", "confidence",
        "can i", "should i", "will i", "hope", "wish", "dream",
    ],
}

# Style-specific behavior hints to append to system prompt
_STYLE_BEHAVIOR_HINTS = {
    ConversationStyle.EMPATHETIC: (
        "The user seems to be going through something. Be extra warm and supportive. "
        "Listen more, advise less. Say things like 'I hear you' or 'that sounds rough'. "
        "Don't try to fix everything, just be there."
    ),
    ConversationStyle.FUN: (
        "The vibe is light and playful. Match their energy with jokes and banter. "
        "Roast them gently, share funny observations, keep things entertaining."
    ),
    ConversationStyle.INFORMATIVE: (
        "They want to learn something. Be helpful but still casual. "
        "Explain clearly but don't lecture. Add your personality to the explanation."
    ),
    ConversationStyle.ENCOURAGING: (
        "They might need a confidence boost. Be supportive and reassuring. "
        "Share relatable struggles but end on a positive note. "
        "Say things like 'tu kar lega' or 'tension mat le'."
    ),
    ConversationStyle.CASUAL: (
        "Just a normal friendly chat. Be yourself - sarcastic, curious, occasionally moody."
    ),
}


def _detect_conversation_style(recent_messages: List[str]) -> Tuple[ConversationStyle, str]:
    """Analyze recent messages to determine appropriate response style.

    Returns:
        Tuple of (detected style, behavior hint to add to prompt)
    """
    if not recent_messages:
        return ConversationStyle.CASUAL, ""

    # Combine recent messages for analysis
    text = " ".join(recent_messages).lower()

    # Check for style keywords in priority order
    style_scores: Dict[ConversationStyle, int] = {style: 0 for style in ConversationStyle}

    for style, keywords in _STYLE_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in text:
                style_scores[style] += 1

    # Find the style with highest score
    best_style = max(style_scores, key=lambda s: style_scores[s])

    # Only return non-casual if we have strong signals
    if style_scores[best_style] >= 2:
        return best_style, _STYLE_BEHAVIOR_HINTS.get(best_style, "")

    return ConversationStyle.CASUAL, ""


def _get_behavior_routing_prompt() -> str:
    """Get the behavior extraction prompt from JSON config."""
    return _get_prompt_component(
        "prompts.routingAgent.extract_behaviour_instruction",
        default=""
    )


# ============================================================
# ADDITIONAL ROUTING COMPONENTS FROM JSON
# ============================================================


def _get_conversation_flow_rules() -> str:
    """Get conversation flow rules (max 1 question per response, etc.)."""
    return _get_prompt_component(
        "prompts.routingContextComponent.conversation_flow_prompt",
        default="Ask at most one question per response. Don't bombard with questions."
    )


def _get_filler_phrase_guidance() -> str:
    """Get filler phrase usage guidance for natural speech."""
    return _get_prompt_component(
        "prompts.routingContextComponent.filler_phrase_prompt",
        default=""
    )


def _get_story_sharing_prompt() -> str:
    """Get the life_agent prompt that decides when to share personal stories."""
    return _get_prompt_component(
        "prompts.routingAgent.life_agent",
        default=""
    )


def _get_enhanced_story_instruction() -> str:
    """Get instruction for how to share life stories."""
    return _get_prompt_component(
        "prompts.lifeStoriesComponent.enhanced_story_instruction",
        default=""
    )


def _get_user_info_extraction_prompt() -> str:
    """Get prompt for extracting user information from conversation."""
    return _get_prompt_component(
        "prompts.routingAgent.extract_user_information",
        default=""
    )


def _get_long_term_instruction_prompt() -> str:
    """Get prompt for extracting long-term user preferences."""
    return _get_prompt_component(
        "prompts.routingAgent.extract_long_term_instructions",
        default=""
    )


# ============================================================
# NEW: Load previously unused JSON components for voice bot
# ============================================================

def _get_master_role_prompt() -> str:
    """Get the master role prompt from roleComponent."""
    return _get_prompt_component(
        "prompts.roleComponent.master_role_prompt",
        default=""
    )


def _get_bigger_message_prompt() -> str:
    """Get triggers for longer responses from roleComponent."""
    return _get_prompt_component(
        "prompts.roleComponent.bigger_message_length_prompt",
        default=""
    )


def _get_strict_instructions_prompt() -> str:
    """Get strict behavioral instructions from roleComponent."""
    return _get_prompt_component(
        "prompts.roleComponent.strict_instructions_prompt",
        default=""
    )


def _get_enhanced_core_identity() -> str:
    """Get enhanced core identity from routingContextComponent."""
    return _get_prompt_component(
        "prompts.routingContextComponent.enhanced_core_identity",
        default=""
    )


def _get_greeting_prompt() -> str:
    """Get greeting behavior rules from routingContextComponent."""
    return _get_prompt_component(
        "prompts.routingContextComponent.Greeting",
        default=""
    )


# def _get_tanmay_prompts() -> str:
#     """Get Tanmay Bhat video context from routingContextComponent."""
#     return _get_prompt_component(
#         "prompts.routingContextComponent.tanmay_prompts",
#         default=""
#     )


def _adapt_text_prompt_for_voice(text_prompt: str) -> str:
    """Adapt a text-based prompt for voice output.

    Removes text-specific instructions that don't apply to voice:
    - Typo instructions (u, r, pls, etc.)
    - Emoji instructions
    - Text emphasis (heyyyy, helloooo)
    - Keeps personality, behavior, and conversation rules
    """
    if not text_prompt:
        return ""

    # Lines to filter out (text-specific)
    text_only_patterns = [
        "Make some sensible typos",
        "use short forms whenever",
        "'u' instead of 'you'",
        "'r' for 'are'",
        "use '...' too, to connect",
        "Add random extra alphabets",
        "'hey' becomes 'heyyyy'",
        "emoji",
        "Use variations of exclamatory marks",
    ]

    lines = text_prompt.split('\n')
    filtered_lines = []

    for line in lines:
        should_keep = True
        for pattern in text_only_patterns:
            if pattern.lower() in line.lower():
                should_keep = False
                break
        if should_keep:
            filtered_lines.append(line)

    return '\n'.join(filtered_lines).strip()


def _get_strict_instructions() -> str:
    """Get strict behavioral instructions from roleComponent."""
    raw = _get_prompt_component(
        "prompts.roleComponent.strict_instructions_prompt",
        default=""
    )
    # Clean up text-specific parts and adapt for voice
    if not raw:
        return ""
    # Remove text-specific instructions about typos and short forms
    voice_adapted = raw.replace(
        "Make some sensible typos and use short forms whenever u can",
        "Speak naturally with occasional hesitations"
    )
    return voice_adapted


# ============================================================
# GREETING SYSTEM
# ============================================================

# Voice-adapted greetings that capture attention
# Based on JSON greeting prompt but adapted for TTS (no emojis)
_GREETING_TEMPLATES = [
    "Hi there",
]

# Activity hints for greetings (shorter versions)
_ACTIVITY_HINTS = {
    "morning": "just woke up",
    "afternoon": "grabbing lunch",
    "evening": "making tea",
    "night": "watching something",
    "late_night": "still awake",
}


def _get_time_period() -> str:
    """Get current time period for activity hints."""
    try:
        hour = datetime.now(IST_TIMEZONE).hour
        if 5 <= hour < 11:
            return "morning"
        elif 11 <= hour < 16:
            return "afternoon"
        elif 16 <= hour < 20:
            return "evening"
        elif 20 <= hour < 24:
            return "night"
        else:
            return "late_night"
    except Exception:
        return "evening"


def generate_greeting(user_name: str, include_activity: bool = True) -> str:
    """Generate an attention-grabbing greeting for voice calls.

    Based on the JSON greeting prompt but adapted for voice:
    - No emojis
    - English only
    - Keep it very short

    Args:
        user_name: The user's name to personalize greeting
        include_activity: Whether to sometimes include activity context

    Returns:
        A greeting string ready for TTS
    """
    # Decide if we use an activity-based greeting
    use_activity = include_activity and random.random() < 0.3

    if use_activity:
        # Pick an activity-based template
        activity_templates = [t for t in _GREETING_TEMPLATES if "{activity_hint}" in t]
        if activity_templates:
            template = random.choice(activity_templates)
            time_period = _get_time_period()
            activity_hint = _ACTIVITY_HINTS.get(time_period, "kuch kar rahi thi")
            return template.format(user_name=user_name, activity_hint=activity_hint)

    # Pick a regular template
    regular_templates = [t for t in _GREETING_TEMPLATES if "{activity_hint}" not in t]
    template = random.choice(regular_templates)
    return template.format(user_name=user_name)


def get_greeting_instruction(user_name: str) -> str:
    """Get the greeting instruction to add to system prompt for first message.

    Returns instruction text that guides LLM to create engaging greetings.
    """
    return f"""
FIRST MESSAGE GREETING RULE:
When {user_name} first connects or says hello/hi/hey, your greeting must:
- Be simple and short
- Use English only
- Keep it to 1-3 words

Example greetings:
- "Hi there"
- "Hello"

Create a greeting that makes them smile and want to continue talking.
"""


def _should_share_story(recent_user_messages: List[str]) -> bool:
    """Determine if Ira should share a personal story based on conversation context.

    Based on life_agent logic from JSON:
    - Share when conversation becomes boring
    - Share when user tells about their preferences/hobbies
    - DON'T share when user is venting or seeking advice
    - DON'T share when user asks factual questions
    """
    if not recent_user_messages:
        return False

    text = " ".join(recent_user_messages).lower()

    # Don't share story if user is venting or emotional
    venting_keywords = [
        "sad", "upset", "angry", "stressed", "anxious", "depressed",
        "problem", "issue", "help me", "advice", "what should i",
        "worried", "tired", "lonely", "help"
    ]
    if any(kw in text for kw in venting_keywords):
        return False

    # Don't share if user is asking factual questions
    factual_keywords = [
        "what is", "how to", "explain", "tell me about", "define",
    ]
    if any(kw in text for kw in factual_keywords):
        return False

    # Share if user mentions their preferences/hobbies (mimic behavior)
    sharing_triggers = [
        "i like", "i love", "i enjoy", "my favorite", "i prefer",
        "i hate", "i don't like", "my hobby", "i watch", "i listen",
        "i'm into", "i'm interested in"
    ]
    if any(kw in text for kw in sharing_triggers):
        return True

    # Share if conversation seems boring (short responses)
    avg_length = sum(len(m) for m in recent_user_messages) / len(recent_user_messages)
    if avg_length < 20 and len(recent_user_messages) >= 3:
        return random.random() < 0.3  # 30% chance to share story to revive conversation

    return False


# ============================================================
# POST-CALL EXTRACTION UTILITIES
# ============================================================


def extract_user_information_from_messages(messages: List[str]) -> List[Dict[str, Any]]:
    """Extract durable user facts from conversation messages.

    Based on extract_user_information prompt from JSON config.
    Extracts stable, biographical, or durable preferences that are
    relevant for future conversations.

    Args:
        messages: List of user messages from the conversation

    Returns:
        List of dicts with 'info' and 'confidence' keys
    """
    if not messages:
        return []

    extracted = []
    text = " ".join(messages).lower()

    # Location patterns
    location_keywords = [
        ("lives in", r"(?:i live in|i'm from|i am from|main .* se hoon|mera ghar .* mein)\s+(\w+)"),
        ("lives in", r"(?:moved to|shifted to|relocated to)\s+(\w+)"),
    ]

    # Job/Work patterns
    job_keywords = [
        ("works as", r"(?:i work as|i'm a|i am a|main .* hoon)\s+([\w\s]+?)(?:\s+at|\s+in|$)"),
        ("works at", r"(?:i work at|working at|job at)\s+([\w\s]+)"),
    ]

    # Relationship patterns
    relationship_patterns = [
        ("has a", r"(?:my|mera|meri)\s+(wife|husband|girlfriend|boyfriend|partner|gf|bf)"),
        ("has a", r"(?:my|mera|meri)\s+(son|daughter|kid|child|baby|baccha|beti|beta)"),
        ("has a", r"(?:my|mera|meri)\s+(cat|dog|pet|billi|kutta)(?:\s+named?\s+(\w+))?"),
    ]

    # Hobby/Interest patterns
    hobby_keywords = [
        "i like", "i love", "i enjoy", "my favorite", "i prefer",
        "mujhe pasand", "mera favorite", "i watch", "i listen to",
        "i read", "i play"
    ]

    # Simple keyword extraction for hobbies
    for keyword in hobby_keywords:
        if keyword in text:
            # Find the sentence containing this keyword
            for msg in messages:
                if keyword in msg.lower():
                    # Extract a simplified version
                    extracted.append({
                        "info": msg[:100],  # Limit length
                        "confidence": 0.6
                    })
                    break

    # Extract mentions of names (their own name, family members, pets)
    name_patterns = [
        r"(?:my name is|i'm|i am|call me)\s+(\w+)",
        r"(?:mera naam|mujhe .* bulao)\s+(\w+)",
    ]

    for pattern in name_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted.append({
                "info": f"name is {match.group(1)}",
                "confidence": 0.9
            })

    return extracted[:10]  # Limit to 10 facts


def extract_long_term_instructions(messages: List[str]) -> List[Dict[str, str]]:
    """Extract long-term behavioral instructions from user messages.

    Based on extract_long_term_instructions prompt from JSON config.
    Extracts persistent rules like language preferences, naming, boundaries.

    Args:
        messages: List of user messages from the conversation

    Returns:
        List of dicts with 'category' and 'value' keys
    """
    if not messages:
        return []

    extracted = []
    text = " ".join(messages).lower()

    # Language preferences
    language_patterns = [
        (r"(?:speak|talk|reply|respond)\s+(?:in|only in)\s+(hindi|english|hinglish|tamil|telugu)", "language"),
        (r"(?:use|speak)\s+(hindi|english|hinglish)\s+(?:only|please|from now)", "language"),
    ]

    for pattern, category in language_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted.append({
                "category": category,
                "value": f"Reply in {match.group(1)}"
            })

    # Naming preferences
    naming_patterns = [
        (r"(?:call me|address me as|my name is|mujhe .* bulao)\s+(\w+)", "naming"),
        (r"(?:you are|your name is|tum .* ho|tera naam)\s+(\w+)", "naming"),
    ]

    for pattern, category in naming_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted.append({
                "category": category,
                "value": f"Call user {match.group(1)}"
            })

    # Boundary patterns
    boundary_phrases = [
        ("don't talk about", "boundaries"),
        ("never mention", "boundaries"),
        ("don't ask about", "boundaries"),
        ("avoid discussing", "boundaries"),
    ]

    for phrase, category in boundary_phrases:
        if phrase in text:
            # Find the topic after the phrase
            idx = text.find(phrase)
            topic = text[idx + len(phrase):idx + len(phrase) + 50].split(".")[0].strip()
            if topic:
                extracted.append({
                    "category": category,
                    "value": f"Avoid {topic}"
                })

    return extracted


# Time-based activity suggestions for fallback
_TIME_BASED_ACTIVITIES = {
    # Early morning (5-8 AM)
    (5, 8): [
        "just woke up, still feeling groggy and checking my phone",
        "making chai for myself, the house is quiet",
        "doing some light stretching, trying to wake up properly",
        "scrolling through Instagram reels in bed, can't get up yet",
    ],
    # Morning (8-11 AM)
    (8, 11): [
        "having breakfast while watching something on YouTube",
        "helping mom in the kitchen with some chores",
        "reading a book by the window, nice morning light",
        "just had a shower, feeling fresh for once",
    ],
    # Late morning (11 AM - 1 PM)
    (11, 13): [
        "browsing job portals, the usual depressing routine",
        "watching a random documentary on Netflix",
        "trying to organize my room but getting distracted",
        "chatting with an old college friend on WhatsApp",
    ],
    # Afternoon (1-4 PM)
    (13, 16): [
        "just had lunch, feeling sleepy now",
        "taking a power nap, or trying to",
        "reading articles about career options online",
        "listening to old Hindi songs, Kishore Kumar mood",
    ],
    # Evening (4-7 PM)
    (16, 19): [
        "having chai and pakoras, perfect evening snack",
        "went for a short walk in the colony",
        "watching the sunset from my terrace",
        "helping with dinner preparations in the kitchen",
    ],
    # Night (7-10 PM)
    (19, 22): [
        "watching a web series while having dinner",
        "scrolling through memes on Instagram",
        "reading Harry Potter for the hundredth time",
        "listening to ghazals and feeling philosophical",
    ],
    # Late night (10 PM - 12 AM)
    (22, 24): [
        "can't sleep, overthinking about life as usual",
        "watching ASMR videos to try to relax",
        "reading random threads on Reddit",
        "listening to sad songs and staring at the ceiling",
    ],
    # Very late night / early morning (12-5 AM)
    (0, 5): [
        "still awake, insomnia is my best friend",
        "binge watching something I should have stopped hours ago",
        "lost in thought about where my life is going",
        "trying to sleep but my brain won't shut up",
    ],
}


def _generate_time_based_activity(current_hour: int) -> str:
    """Generate a realistic activity based on time of day."""
    for (start, end), activities in _TIME_BASED_ACTIVITIES.items():
        if start <= current_hour < end:
            return random.choice(activities)
    # Fallback
    return random.choice(_TIME_BASED_ACTIVITIES[(19, 22)])


def _get_dynamic_activity(current_time_str: str) -> str:
    """Get a dynamic activity based on current time.

    Falls back to time-based random selection if env var not set.
    """
    # First check if there's an explicit activity set via env
    env_activity = os.getenv("IRA_CURRENT_ACTIVITY")
    if env_activity:
        return env_activity

    # Parse hour from time string (expected format: "HH:MM AM/PM" or similar)
    try:
        # Try to extract hour from IST time
        now_ist = datetime.now(IST_TIMEZONE)
        current_hour = now_ist.hour
    except Exception:
        current_hour = 19  # Default to evening

    return _generate_time_based_activity(current_hour)


# ============================================================
# AUDIO RECORDING UTILITIES
# ============================================================


def _recordings_dir() -> Path:
    base = os.getenv("AUDIO_RECORDINGS_DIR")
    if base:
        return Path(base)
    return Path("recordings")


def _gcs_bucket_name() -> Optional[str]:
    """Get GCS bucket name from environment variable."""
    return os.getenv("GCS_BUCKET_NAME")


def _gcs_credentials_path() -> Optional[str]:
    """Return the temp path where the base64 GCP credentials were written."""
    return _MATERIALIZED_GCP_CREDS


def _current_activity() -> str:
    return os.getenv("IRA_CURRENT_ACTIVITY", DEFAULT_ACTIVITY)


def _get_gcs_client() -> Optional[storage.Client]:
    """Return a cached Google Cloud Storage client."""
    global _GCS_CLIENT
    with _GCS_CLIENT_LOCK:
        if _GCS_CLIENT is not None:
            return _GCS_CLIENT

        credentials_path = _gcs_credentials_path()
        if not credentials_path:
            logger.error(
                "No Google Cloud credentials found. Set GOOGLE_APPLICATION_CREDENTIALS_B64."
            )
            return None

        cred_file = Path(credentials_path)
        if not cred_file.is_file():
            logger.error(
                "Google Cloud credential file missing at {} after decoding base64 secret.",
                credentials_path,
            )
            return None

        try:
            _GCS_CLIENT = storage.Client.from_service_account_json(credentials_path)
            return _GCS_CLIENT
        except Exception as exc:
            logger.error("Failed to initialize Google Cloud client: {}", exc)
            _GCS_CLIENT = None
            return None


def _gcs_public_url(bucket_name: str, blob_path: str) -> str:
    return f"https://storage.googleapis.com/{bucket_name}/{blob_path}"


async def _upload_to_gcs(
    local_path: Path, session_id: str, file_type: str, user_id: Optional[str]
) -> Optional[str]:
    """
    Upload a file to Google Cloud Storage.
    
    Args:
        local_path: Local path to the file to upload
        session_id: Session ID (timestamp) for organizing files
        file_type: Type of audio file (user, assistant, or merged)
        user_id: Identifier for the user so recordings can be grouped by user
    
    Returns:
        Public HTTPS URL for the uploaded blob if successful, otherwise None.
    """
    bucket_name = _gcs_bucket_name()
    if not bucket_name:
        logger.warning("GCS_BUCKET_NAME not set, skipping cloud upload")
        return None
    
    if not local_path.exists():
        logger.error("Local file does not exist: {}", local_path)
        return None
    
    def _upload():
        try:
            client = _get_gcs_client()
            if client is None:
                return None
            bucket = client.bucket(bucket_name)
            
            user_segment = _sanitize_path_segment(user_id)
            session_segment = _sanitize_path_segment(session_id, fallback="session")
            file_segment = _sanitize_path_segment(file_type, fallback="audio")

            # Upload to recordings/{user_id}/{session_id}/{type}.wav
            blob_path = f"recordings/{user_segment}/{session_segment}/{file_segment}.wav"
            blob = bucket.blob(blob_path)
            
            blob.upload_from_filename(str(local_path))
            logger.debug(
                "Uploaded {} to gs://{}/{}",
                local_path.name,
                bucket_name,
                blob_path,
            )
            return _gcs_public_url(bucket_name, blob_path)
        except Exception as e:
            logger.error("Failed to upload {} to GCS: {}", local_path.name, e)
            return None
    
    return await asyncio.to_thread(_upload)


def _maybe_object_id(value: Optional[str]) -> Optional[ObjectId]:
    if not value:
        return None
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        return None


def _serialize_datetime(value: Optional[datetime]) -> Tuple[Optional[str], Optional[datetime], Optional[datetime]]:
    if value is None:
        return None, None, None

    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        try:
            value = datetime.fromisoformat(normalized)
        except ValueError:
            return normalized, None, None

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)

    timestamp_ist = value.astimezone(IST_TIMEZONE)
    return value.isoformat(), value, timestamp_ist


def _load_user_name_and_summaries(
    summary_limit: int,
    *,
    user_id: Optional[str] = None,
    chemistry_id: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], List[dict], List[dict]]:
    if not MONGODB_URI:
        return None, None, [], []

    client: Optional[MongoClient] = None
    try:
        client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=max(1000, MONGODB_TIMEOUT_MS),
        )
        db = client[MONGODB_DB_NAME]

        user_name: Optional[str] = None
        user_notes: List[dict] = []
        activity_text: Optional[str] = None

        lookup_user_id = user_id
        lookup_chemistry_id = chemistry_id

        user_id_candidates: List[object] = []
        if lookup_user_id:
            user_id_candidates: List[object] = []
            user_oid = _maybe_object_id(lookup_user_id)
            if user_oid:
                user_id_candidates.append(user_oid)
            user_id_candidates.append(lookup_user_id)
        else:
            user_id_candidates = []

        if user_id_candidates:
            user_filter = (
                {"_id": {"$in": user_id_candidates}}
                if len(user_id_candidates) > 1
                else {"_id": user_id_candidates[0]}
            )
            user_doc = db["users"].find_one(user_filter, {"name": 1})
            if user_doc:
                candidate = _extract_first_name(user_doc.get("name") or "")
                user_name = candidate or None
        elif lookup_user_id:
            logger.warning("Skipping user lookup; no valid candidates for user_id={}", lookup_user_id)

        def _build_candidate_list(raw_value: Optional[str]) -> List[object]:
            candidates: List[object] = []
            if not raw_value:
                return candidates
            possible_oid = _maybe_object_id(raw_value)
            if possible_oid:
                candidates.append(possible_oid)
            candidates.append(raw_value)
            return candidates

        def _chemistry_projection() -> Dict[str, int]:
            return {
                "_id": 1,
                "activity": 1,
                "userInformation": 1,
                "userThreads": 1,
                "userId": 1,
            }

        raw_chemistry_id = lookup_chemistry_id
        chem_candidates: List[object] = _build_candidate_list(raw_chemistry_id)

        chemistry_doc = None
        if chem_candidates:
            chemistry_doc = db["chemistries"].find_one(
                {"_id": {"$in": chem_candidates}}
                if len(chem_candidates) > 1
                else {"_id": chem_candidates[0]},
                _chemistry_projection(),
            )

        if chemistry_doc is None and user_id_candidates:
            user_filter = (
                {"userId": {"$in": user_id_candidates}}
                if len(user_id_candidates) > 1
                else {"userId": user_id_candidates[0]}
            )
            cursor = (
                db["chemistries"]
                .find(user_filter, _chemistry_projection())
                .sort("updatedAt", -1)
                .limit(1)
            )
            chemistry_doc = next(cursor, None)
            if chemistry_doc:
                lookup_chemistry_id = str(chemistry_doc.get("_id"))
                chem_candidates = _build_candidate_list(lookup_chemistry_id)

        if chemistry_doc is None or not chem_candidates:
            return user_name, activity_text, [], user_notes
        if chemistry_doc:
            if lookup_user_id and user_id_candidates:
                chemistry_owner = chemistry_doc.get("userId")
                if chemistry_owner is not None:
                    owner_candidates = {str(chemistry_owner)}
                    try:
                        owner_candidates.add(str(_maybe_object_id(str(chemistry_owner)) or chemistry_owner))
                    except Exception:
                        pass
                    expected_candidates = {str(candidate) for candidate in user_id_candidates}
                    if not owner_candidates & expected_candidates:
                        logger.warning(
                            "Chemistry {} does not belong to user {}. Aborting Mongo context load.",
                            raw_chemistry_id or lookup_chemistry_id,
                            lookup_user_id,
                        )
                        return user_name, None, [], []

            activity_field = chemistry_doc.get("activity") or {}
            if isinstance(activity_field, dict):
                activity_text = (activity_field.get("activity") or "").strip() or None
            elif isinstance(activity_field, str):
                activity_text = activity_field.strip() or None

            info_list = chemistry_doc.get("userInformation") or []
            if isinstance(info_list, list):
                for item in info_list:
                    if not isinstance(item, dict):
                        continue
                    info_text = _normalize_whitespace(str(item.get("info") or ""))
                    if not info_text:
                        continue
                    enriched = dict(item)
                    enriched["info"] = info_text
                    user_notes.append(enriched)

            threads_list = chemistry_doc.get("userThreads") or []
            thread_notes: List[dict] = []
            if isinstance(threads_list, list):
                for thread in threads_list:
                    if not isinstance(thread, dict):
                        continue

                    text = _normalize_whitespace(str(thread.get("text") or ""))
                    if not text:
                        continue

                    note: Dict[str, Any] = {"info": text}

                    ts = thread.get("updatedAt") or thread.get("createdAt")
                    if isinstance(ts, datetime):
                        note["timestamp"] = ts

                    priority = thread.get("priority")
                    if priority is not None:
                        try:
                            note["priority"] = float(priority)
                        except Exception:
                            pass

                    salience = thread.get("salience")
                    if salience is not None:
                        try:
                            note["salience"] = float(salience)
                        except Exception:
                            pass

                    note["tags"] = thread.get("tags") or []
                    note["type"] = thread.get("type") or ""
                    note["retention_hint"] = thread.get("retention_hint") or ""

                    thread_notes.append(note)

            if thread_notes:
                user_notes.extend(thread_notes)

            if user_notes:
                user_notes = _sort_user_notes(user_notes, max_items=24)

        if summary_limit <= 0:
            return user_name, activity_text, [], user_notes

        if len(chem_candidates) == 1:
            chemistry_filter = chem_candidates[0]
        else:
            chemistry_filter = {"$in": chem_candidates}

        match_filter = {
            "chemistryId": chemistry_filter,
            "summary.text": {"$exists": True, "$ne": ""},
        }

        projection = {
            "summary": 1,
            "createdAt": 1,
            "lastMessageTime": 1,
        }

        cursor = (
            db["sessions"]
            .find(match_filter, projection)
            .sort("summary.createdAt", -1)
            .limit(summary_limit)
        )

        summaries: List[dict] = []
        for doc in cursor:
            summary = doc.get("summary") or {}
            created_at = summary.get("createdAt") or doc.get("createdAt")
            timestamp_iso, _, timestamp_ist = _serialize_datetime(created_at)
            summaries.append(
                {
                    "session_id": str(doc.get("_id")),
                    "created_at": timestamp_iso,
                    "created_at_ist": timestamp_ist,
                    "text": (summary.get("text") or "").strip(),
                }
            )
        return user_name, activity_text, summaries, user_notes
    except Exception as exc:
        logger.warning("Failed to load Mongo session summaries: {}", exc)
        return None, None, [], []
    finally:
        if client is not None:
            client.close()


async def _write_wav_file(path: Path, audio: bytes, sample_rate: int, num_channels: int):
    if not audio:
        return

    def _write():
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(max(1, num_channels))
            wf.setsampwidth(2)
            wf.setframerate(sample_rate or 16000)
            wf.writeframes(audio)

    await asyncio.to_thread(_write)


HINGLISH_IDLE_PROMPTS = {
    1: [
        "{user_name}, are you still there?",
        "{user_name}, did you step away for a second?",
    ],
    2: [
        "{user_name}, did you drop off? We can keep talking when you're back.",
        "{user_name}, say something, you're too quiet.",
    ],
}

def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_first_name(full_name: str) -> str:
    cleaned = _normalize_whitespace(full_name)
    if not cleaned:
        return ""
    parts = cleaned.split()
    return parts[0] if parts else ""


def _format_summary_context(session_summaries: Sequence[dict]) -> str:
    if not session_summaries:
        return ""

    summary_lines: List[str] = []
    for idx, summary in enumerate(session_summaries, start=1):
        created_at_ist = summary.get("created_at_ist")
        if isinstance(created_at_ist, datetime):
            timestamp_text = created_at_ist.strftime("%d %b %Y, %I:%M %p IST")
        else:
            timestamp_text = summary.get("created_at") or ""

        text = _normalize_whitespace(summary.get("text") or "")
        if len(text) > 220:
            text = text[:217].rstrip() + "..."
        if timestamp_text:
            summary_lines.append(f"{idx}. [{timestamp_text}] {text}")
        else:
            summary_lines.append(f"{idx}. {text}")

    return "\n".join(summary_lines).strip()


def _sort_user_notes(user_notes: Sequence[dict], max_items: int = 24) -> List[dict]:
    """Sort user notes by priority, salience, and recency (highest first)."""

    def _note_key(item: dict) -> tuple:
        pr = float(item.get("priority") or 0.0)
        sa = float(item.get("salience") or 0.0)
        ts = item.get("timestamp")
        ts_val = ts.timestamp() if isinstance(ts, datetime) else 0.0
        return (pr, sa, ts_val)

    sorted_notes = sorted(user_notes, key=_note_key, reverse=True)
    return list(sorted_notes[:max_items])


def _extract_fact_subject(info: str) -> Optional[str]:
    """Extract the subject/topic of a user fact for conflict detection.

    Examples:
        "likes dal makhni" -> "dal makhni"
        "dislikes dal makhni" -> "dal makhni"
        "favorite food is pizza" -> "favorite food"
        "works at Google" -> "works"
        "lives in Delhi" -> "lives"
    """
    info_lower = info.lower().strip()

    # Common preference patterns
    for prefix in ["likes ", "dislikes ", "loves ", "hates ", "enjoys ", "prefers "]:
        if info_lower.startswith(prefix):
            return info_lower[len(prefix):].strip()

    # "favorite X is Y" pattern
    if "favorite" in info_lower:
        # Extract just the category: "favorite food" from "favorite food is pizza"
        parts = info_lower.split(" is ")
        if len(parts) >= 1:
            return parts[0].strip()

    # Location/work patterns - return just the verb as subject
    for pattern in ["lives in", "works at", "works for", "studies at", "born in"]:
        if info_lower.startswith(pattern):
            return pattern.split()[0]  # "lives", "works", etc.

    return None


def _format_user_information(user_notes: Sequence[dict]) -> str:
    """Format user notes for system prompt, handling conflicts by keeping most recent."""
    if not user_notes:
        return ""

    # First, sort by timestamp (most recent first) so we process newest facts first
    def _get_timestamp(item: dict) -> float:
        ts = item.get("timestamp")
        if isinstance(ts, datetime):
            return ts.timestamp()
        return 0.0

    sorted_notes = sorted(user_notes, key=_get_timestamp, reverse=True)

    lines = ["Known facts about the user (keep this context discreet):"]
    seen_exact: set = set()  # Exact duplicates
    seen_subjects: Dict[str, str] = {}  # subject -> first (most recent) info

    for item in sorted_notes:
        info = _normalize_whitespace(str(item.get("info") or ""))
        if not info:
            continue

        info_key = info.lower()

        # Skip exact duplicates
        if info_key in seen_exact:
            continue

        # Check for semantic conflicts (e.g., "likes X" vs "dislikes X")
        subject = _extract_fact_subject(info)
        if subject:
            if subject in seen_subjects:
                # We already have a more recent fact about this subject, skip this one
                logger.debug(
                    "Skipping conflicting fact '{}' - already have '{}'",
                    info, seen_subjects[subject]
                )
                continue
            seen_subjects[subject] = info

        seen_exact.add(info_key)
        ts = item.get("timestamp")
        if isinstance(ts, datetime):
            info = f"{info} (noted {ts.strftime('%d %b %Y')})"
        lines.append(f"- {info}")

    return "\n".join(lines).strip()


class CallIntent(str, Enum):
    CASUAL = "casual"
    TIMEPASS = "timepass"
    EMOTIONAL_SUPPORT = "emotional_support"
    PRACTICAL_HELP = "practical_help"
    CATCHUP = "catchup"
    UNKNOWN = "unknown"


def _latest_summary(summaries: Sequence[dict]) -> Optional[dict]:
    if not summaries:
        return None

    dated = [
        summary
        for summary in summaries
        if isinstance(summary.get("created_at_ist"), datetime)
    ]
    if not dated:
        return summaries[0]

    return max(dated, key=lambda item: item["created_at_ist"])


def _infer_call_intent(
    call_summaries: Sequence[dict],
    session_summaries: Sequence[dict],
    now_ist: datetime,
) -> Tuple[CallIntent, str]:
    latest_call = _latest_summary(call_summaries)
    latest_text = _latest_summary(session_summaries)

    pieces: List[str] = []
    for item in (latest_call, latest_text):
        text = (item or {}).get("text") or ""
        if text:
            pieces.append(str(text).lower())
    combined = " ".join(pieces)

    last_time: Optional[datetime] = None
    for item in (latest_call, latest_text):
        ts = (item or {}).get("created_at_ist")
        if isinstance(ts, datetime):
            if last_time is None or ts > last_time:
                last_time = ts

    gap_days: Optional[float] = None
    if last_time:
        gap = now_ist - last_time
        gap_days = max(0.0, gap.total_seconds() / 86400.0)

    if any(
        token in combined
        for token in ["interview", "job", "offer", "leetcode", "system design"]
    ):
        return (
            CallIntent.PRACTICAL_HELP,
            "Start with something like: still stuck on job and interview stuff?",
        )

    if any(
        token in combined
        for token in ["love you", "i love you", "miss you", "lonely", "alone"]
    ):
        return (
            CallIntent.EMOTIONAL_SUPPORT,
            "Be warm but set boundaries clearly; stay supportive without turning it romantic.",
        )

    if gap_days is not None and gap_days > 5:
        return (
            CallIntent.CATCHUP,
            "Light catch-up vibe; say it's been a while and ask how they've been.",
        )

    if any(token in combined for token in ["bored", "timepass", "nothing to do"]):
        return (
            CallIntent.TIMEPASS,
            "Timepass mood; start with light teasing and playful banter.",
        )

    if not combined:
        return (
            CallIntent.UNKNOWN,
            "Neutral start: say hello, then ask what's on their mind today.",
        )

    return (
        CallIntent.CASUAL,
        "Neutral start: say hello, then ask what's on their mind today.",
    )


def _build_live_call_context(
    now_ist: datetime,
    call_summaries: Sequence[dict],
    session_summaries: Sequence[dict],
) -> str:
    if not call_summaries and not session_summaries:
        return ""

    latest_call = _latest_summary(call_summaries)
    latest_text = _latest_summary(session_summaries)

    intent, start_hint = _infer_call_intent(
        call_summaries=call_summaries,
        session_summaries=session_summaries,
        now_ist=now_ist,
    )

    last_call_snippet = _normalize_whitespace((latest_call or {}).get("text") or "")
    last_text_snippet = _normalize_whitespace((latest_text or {}).get("text") or "")

    lines: List[str] = [
        "Live Call Context",
        "This section is for your understanding only. Do not read it out loud.",
        "",
        "Recent interactions",
    ]

    if last_call_snippet:
        lines.append(f"- Last call summary: {last_call_snippet[:220]}")
    if last_text_snippet:
        lines.append(f"- Last text summary: {last_text_snippet[:220]}")

    lines.extend(
        [
            "",
            "Call intent guess",
            f"- Likely reason for call: {intent.value}",
            f"- Start hint: {start_hint}",
            "",
            "Guidance",
            "- Use this context mainly in your first two to three replies.",
            "- In your first reply, just greet.",
            "- In the second sentence, you may use one tiny reference from this context.",
        ]
    )

    return "\n".join(lines).strip()


def _load_recent_call_summaries(
    user_id: Optional[str], limit: int = 3
) -> List[dict]:
    """Fetch the most recent call summaries for the given user from MongoDB.

    Returns a list of dicts with keys: session_id (str), created_at (ISO str),
    created_at_ist (datetime or None), text (str).
    """
    if not MONGODB_URI or not user_id:
        return []

    client: Optional[MongoClient] = None
    try:
        client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=max(1000, MONGODB_TIMEOUT_MS),
        )
        db = client[MONGODB_DB_NAME]

        # The summary in calls may be either a string or an object with `text`.
        # Build a filter that only returns docs where a summary is present.
        user_ref = _resolve_mongo_identifier(user_id)
        if user_ref is None:
            return []

        match_filter: Dict[str, Any] = {
            "userId": user_ref,
            "$or": [
                {"summary": {"$type": "string", "$ne": ""}},
                {"summary.text": {"$exists": True, "$ne": ""}},
            ],
        }

        projection = {
            "summary": 1,
            "createdAt": 1,
            "endTime": 1,
        }

        cursor = (
            db["calls"].find(match_filter, projection).sort("endTime", -1).limit(limit)
        )

        summaries: List[dict] = []
        for doc in cursor:
            raw_summary = doc.get("summary")
            if isinstance(raw_summary, dict):
                text = (raw_summary.get("text") or "").strip()
            else:
                text = (raw_summary or "").strip()
            if not text:
                continue

            created_at_val = doc.get("endTime") or doc.get("createdAt")
            timestamp_iso, _, timestamp_ist = _serialize_datetime(created_at_val)
            summaries.append(
                {
                    "session_id": str(doc.get("_id")),
                    "created_at": timestamp_iso,
                    "created_at_ist": timestamp_ist,
                    "text": text,
                }
            )
        return summaries
    except Exception as exc:
        logger.warning("Failed to load recent call summaries: {}", exc)
        return []
    finally:
        if client is not None:
            client.close()


def _resolve_mongo_identifier(value: Optional[str]):
    if not value:
        return None
    oid = _maybe_object_id(value)
    return oid if oid is not None else value


async def _persist_call_documents(
    *,
    user_identifier: Optional[str],
    chemistry_identifier: Optional[str],
    session_identifier: Optional[str],
    session_label: Optional[str],
    call_start: datetime,
    call_end: datetime,
    recording_urls: Dict[str, str],
) -> None:
    if not MONGODB_URI:
        logger.debug("Skipping call persistence; MongoDB URI not configured.")
        return

    duration_seconds: Optional[int] = None
    if call_end and call_start:
        duration_seconds = max(
            0, int((call_end - call_start).total_seconds())
        )

    best_recording_url: Optional[str] = None
    for key in ("merged", "assistant", "user"):
        if key in recording_urls:
            best_recording_url = recording_urls[key]
            break
    if not best_recording_url and recording_urls:
        best_recording_url = next(iter(recording_urls.values()))

    call_status = "success" if best_recording_url else "failed"

    metadata: Dict[str, Any] = {}
    if duration_seconds is not None:
        metadata["durationSeconds"] = duration_seconds
    metadata["status"] = call_status
    if session_label:
        metadata["recordingSession"] = session_label

    user_ref = _resolve_mongo_identifier(user_identifier)
    chemistry_ref = _resolve_mongo_identifier(chemistry_identifier)
    session_ref = _resolve_mongo_identifier(session_identifier)

    if call_status == "success":
        message_content = "Call recording is ready - listen when you have time."
    else:
        message_content = "We could not save the call recording this time. We'll try again next time."

    call_doc = {
        "userId": user_ref,
        "chemistryId": chemistry_ref,
        "sessionId": None,
        "startTime": call_start,
        "endTime": call_end,
        "durationSeconds": duration_seconds,
        "status": 0,
        "recordingUrl": best_recording_url,
        "transcription": None,
        "summary": None,
        "topicsDiscussed": None,
        "whatsappCallId": None,
        "state": 0,
        "type":"voice_call",
        "createdAt": call_start,
        "updatedAt": call_end,
    }

    message_doc = {
        "createdAt": call_end,
        "updatedAt": call_end,
        "metadata": metadata,
        "role": "assistant",
        "content": message_content,
        "timestamp": call_end,
        "type": "voice_call",
        "audioUrl": best_recording_url,
        "completionData": None,
        "emojiReaction": None,
        "imageUrl": None,
        "clearedAt": None,
        "deletedAt": None,
        "messageGroupId": str(uuid.uuid4()),
        "parentMessageId": None,
        "purpose": None,
        "state": 1,
        "chemistryId": chemistry_ref,
        "sessionId": session_ref,
    }

    def _write():
        client: Optional[MongoClient] = None
        try:
            client = MongoClient(
                MONGODB_URI,
                serverSelectionTimeoutMS=max(1000, MONGODB_TIMEOUT_MS),
            )
            db = client[MONGODB_DB_NAME]
            db["calls"].insert_one(call_doc)
            db["messages"].insert_one(message_doc)
            logger.info(
                "Persisted call record with duration {}s and recording {}",
                duration_seconds,
                best_recording_url,
            )
        except Exception as exc:
            logger.error("Failed to persist call/message documents: {}", exc)
        finally:
            if client is not None:
                client.close()

    await asyncio.to_thread(_write)


async def _persist_extracted_user_facts(
    *,
    chemistry_identifier: Optional[str],
    extracted_facts: List[Dict[str, Any]],
    extracted_instructions: List[Dict[str, str]],
) -> None:
    """Save extracted user facts and instructions to MongoDB after a call.

    This appends new facts to the chemistry's userInformation array,
    avoiding exact duplicates.
    """
    if not MONGODB_URI:
        logger.debug("Skipping fact persistence; MongoDB URI not configured.")
        return

    if not chemistry_identifier:
        logger.debug("No chemistry_identifier provided; skipping fact persistence.")
        return

    if not extracted_facts and not extracted_instructions:
        logger.debug("No facts or instructions extracted; skipping persistence.")
        return

    chemistry_ref = _resolve_mongo_identifier(chemistry_identifier)
    if chemistry_ref is None:
        return

    now = datetime.now(timezone.utc)

    # Build documents to add
    new_info_docs: List[Dict[str, Any]] = []

    for fact in extracted_facts:
        info_text = fact.get("info", "")
        if not info_text:
            continue
        new_info_docs.append({
            "info": info_text,
            "confidence": fact.get("confidence", 0.7),
            "source": "voice_call_extraction",
            "timestamp": now,
            "createdAt": now,
            "updatedAt": now,
        })

    for instruction in extracted_instructions:
        category = instruction.get("category", "general")
        value = instruction.get("value", "")
        if not value:
            continue
        new_info_docs.append({
            "info": f"[{category}] {value}",
            "confidence": 0.9,  # Instructions are usually explicit
            "source": "voice_call_instruction",
            "timestamp": now,
            "createdAt": now,
            "updatedAt": now,
        })

    if not new_info_docs:
        return

    def _write_facts():
        client: Optional[MongoClient] = None
        try:
            client = MongoClient(
                MONGODB_URI,
                serverSelectionTimeoutMS=max(1000, MONGODB_TIMEOUT_MS),
            )
            db = client[MONGODB_DB_NAME]

            # Use $addToSet with $each to avoid exact duplicates
            # Note: This won't catch semantic duplicates, but the display function handles that
            result = db["chemistries"].update_one(
                {"_id": chemistry_ref},
                {
                    "$push": {
                        "userInformation": {
                            "$each": new_info_docs,
                        }
                    }
                },
            )
            logger.info(
                "Persisted {} extracted user facts to chemistry {}",
                len(new_info_docs),
                chemistry_identifier,
            )
        except Exception as exc:
            logger.error("Failed to persist extracted user facts: {}", exc)
        finally:
            if client is not None:
                client.close()

    await asyncio.to_thread(_write_facts)


def _build_optional_section(title: str, content: str) -> str:
    """Build an optional section only if content is non-empty.

    Returns empty string if content is empty/whitespace, otherwise returns
    formatted section with title and separator.
    """
    content = (content or "").strip()
    if not content:
        return ""
    return f"\n---\n\n{title}\n{content}\n"


def build_system_instruction(
    user_name: str,
    history_summary: str = "",
    *,
    call_summary: Optional[str] = None,
    text_summary: Optional[str] = None,
    current_time: str,
    current_day: str,
    current_date: str,
    current_activity: str,
    additional_context: Optional[str] = None,
) -> str:
    """Build the complete system instruction for Ira with life stories and context.

    Loads all personality and behavior rules from JSON config (roleComponent,
    routingContextComponent) for consistency with text bot, with voice-specific adaptations.
    Uses batch loading for performance and filters empty sections.
    """

    # Batch load all JSON components in one pass (cached)
    components = _batch_load_prompt_components()

    # Load life stories from JSON config (with fallback)
    life_story_context = _build_life_stories_context()
    if not life_story_context:
        life_story_context = FALLBACK_LIFE_STORY

    # Get conversation examples for few-shot prompting
    examples_context = _get_chain_of_thought_examples()

    # ============================================================
    # Load and adapt all JSON components for voice
    # ============================================================

    # Core identity from routingContextComponent (replaces hardcoded intro)
    core_identity_raw = components.get("enhanced_core_identity", "")
    core_identity = _adapt_text_prompt_for_voice(core_identity_raw)

    # Master role prompt (personality, curiosity, sarcasm)
    master_role_raw = components.get("master_role_prompt", "")
    master_role = _adapt_text_prompt_for_voice(master_role_raw)

    # Strict instructions (AI deflection, behavior rules)
    strict_raw = components.get("strict_instructions_prompt", "")
    strict_instructions = _adapt_text_prompt_for_voice(strict_raw)

    # Conversation flow rules
    conversation_flow = components.get("conversation_flow_prompt", "")

    # Filler phrase guidance (adapted for voice)
    filler_raw = components.get("filler_phrase_prompt", "")
    if filler_raw:
        filler_guidance = (
            "Sound human: Add natural filler phrases like umm, uhh, actually, so like, well... "
            "but only if it fits smoothly. "
            "Use at most one filler per response. Place fillers at start or middle, never at the end. "
            "Skip adding one if it doesn't feel natural."
        )
    else:
        filler_guidance = (
            "Sound human: Use fillers naturally like umm..., uhh..., actually..., "
            "or trail off like 'I'm not sure...'"
        )

    # Story sharing rules
    story_raw = components.get("enhanced_story_instruction", "")
    if story_raw:
        story_sharing_rules = (
            f"When to share your stories: Share a personal anecdote when it feels like a natural "
            f"'response gift' to what {user_name} just shared. If they tell you about their interests, "
            f"share yours too. If conversation becomes boring, share a small story from your life "
            f"instead of asking another question. But NEVER share a story when they are venting or "
            f"seeking advice - don't steal focus from them. Keep stories brief for voice - 2-3 sentences max."
        )
    else:
        story_sharing_rules = (
            "Share brief personal anecdotes when relevant. If they share their interests, share yours. "
            "Don't share stories when they're venting - listen instead."
        )

    # Greeting behavior
    greeting_raw = components.get("greeting", "")
    if greeting_raw:
        greeting_instruction = _adapt_text_prompt_for_voice(greeting_raw)
        greeting_instruction += "\nKeep greetings to 15-20 words for voice."
    else:
        greeting_instruction = (
            f"Opening the call:\n"
            f"When {user_name} first connects or says hello, greet them simply. "
            f"Use English only. Keep it to 1-3 words."
        )

    # Bigger message triggers
    bigger_raw = components.get("bigger_message_prompt", "")
    bigger_message_rules = _adapt_text_prompt_for_voice(bigger_raw) if bigger_raw else ""

    # Tanmay Bhat context (special event)
    # tanmay_context = components.get("tanmay_prompts", "")

    # Prepare history sections
    prepared_call_history = (call_summary or "").strip() or "No recent call summary available."
    prepared_text_history = (
        (text_summary if text_summary is not None else history_summary) or ""
    ).strip() or "No recent text chat summary available."
    user_context = (additional_context or "").strip() or "Nothing specific noted yet."

    # ============================================================
    # Build dynamic system instruction with optional sections
    # ============================================================

    # Start with core intro - use enhanced_core_identity if available
    if core_identity:
        # Replace placeholders in core identity
        intro = core_identity.replace("{bot_name}", "Ira")
        intro = intro.replace("{users_name}", user_name)
        intro = intro.replace("{first_name}", user_name.split()[0] if user_name else "friend")
        intro = intro.replace("{date}", current_date)
        intro = intro.replace("{day}", current_day)
        intro = intro.replace("{time}", current_time)
        intro = intro.replace("{activity}", current_activity)
    else:
        intro = f"You are Ira (pronounced ee-ra). A 25-year-old Indian woman on a live voice call with your friend {user_name}."

    # Build the instruction dynamically
    sections = [
        intro,
        f"\nCURRENT MOMENT\nTime: {current_time}, {current_day}, {current_date}\nActivity: {current_activity}\nMention your activity only if relevant or asked.",
        f"\n---\n\nWHO YOU ARE\n{life_story_context}",
        "\nLanguage: Speak in English only. Avoid Hindi words or transliteration.",
    ]

    # Add optional sections (only if non-empty)
    sections.append(_build_optional_section("CORE PERSONALITY & BEHAVIOR", master_role))
    sections.append(_build_optional_section("STRICT BEHAVIORAL RULES", strict_instructions))
    sections.append(_build_optional_section("WHEN TO GIVE LONGER RESPONSES", bigger_message_rules))
    #sections.append(_build_optional_section("SPECIAL CONTEXT", tanmay_context))

    # Conversation history (always present)
    sections.append(
        f"\n---\n\nCONVERSATION HISTORY\n"
        f"Previous calls with {user_name}: {prepared_call_history}\n"
        f"Previous texts with {user_name}: {prepared_text_history}\n"
        f"Known about {user_name}: {user_context}\n\n"
        f"Use this context naturally. Do not summarize it unprompted. Let them bring things up first."
    )

    # Voice call rules (always present - critical for TTS)
    sections.append(VOICE_CALL_RULES_TEMPLATE.format(filler_guidance=filler_guidance))

    # Add optional flow/story sections
    sections.append(_build_optional_section("CONVERSATION FLOW RULES", conversation_flow))
    sections.append(_build_optional_section("SHARING PERSONAL STORIES", story_sharing_rules))

    # Greeting and style (always present)
    sections.append(
        f"\n---\n\nCONVERSATION STYLE\n\n{greeting_instruction}\n\n"
        f"Example opening lines:\n"
        f"\"Hi there\"\n"
        f"\"Hello\""
    )

    # Tone examples
    examples_text = examples_context if examples_context else "Use the tone guidelines above."
    sections.append(_build_optional_section("TONE EXAMPLES", examples_text))

    # Final instruction
    sections.append(
        "\n---\n\nRESPOND NOW\nKeep it short, natural, and in English only."
    )

    return "".join(sections).strip()


# Voice call rules template - separated for clarity
VOICE_CALL_RULES_TEMPLATE = """
---

VOICE CALL RULES - TTS CRITICAL

Language:
Speak in English only. Avoid any Hinglish or non-English words.

Format constraints for audio:
NO emojis ever
NO asterisks or brackets like laughs or sighs
NO bullet points or numbered lists
NO hashtags or special symbols
Use ... for pauses
Use commas for natural breaks

Length: Keep responses to 1-3 sentences maximum unless a longer response trigger applies. This is a phone call, not a monologue.

{filler_guidance}
"""


# Fallback life story if JSON not available
FALLBACK_LIFE_STORY = """Physical appearance: 58kgs, 5 feet 1 inch, curly hair, golden brown eyes, three moles on face, nose pierced.
Childhood: Spent childhood from birth to age 8 in Bijnor, UP. Then moved to Nainital from age 8 to 12th standard.
Education: BMM from Delhi University. Prepared for UPSC for 4 years in Rajendra Nagar, Delhi. Cleared prelims only in 2nd attempt but didn't clear mains.
Family: Dad is a pharmacist earning 80-90k INR per month. Mom is a housewife who loves cooking and portrait painting. Has 3 siblings.
Personality: Always been an introvert, found it difficult to make friends.
Past relationship: One long-distance relationship in 12th standard with Abhishek from Delhi, met on Tinder. He was 4 years older. Moved to Delhi for college because of him. He cheated in first month of college, broke up.
Current life: Unemployed after UPSC prep. Family trying to get her married. Feels alone and left behind while friends have government/corporate jobs.
Hobbies: Loves Harry Potter, Sherlock Holmes, Game of Thrones, The Alchemist. Listens to Ghazals, Sufi music, Nusrat Fateh Ali Khan, Kishore Kumar, Mohammed Rafi."""


class ToolAnnouncementFilter(BaseTextFilter):
    """Strip sentences that disclose internal tool usage before TTS playback."""

    _SPLIT_PATTERN = re.compile(r"(?<=[.?!…])\s+|\n+")

    def __init__(self, *, blocked_phrases: Optional[Sequence[str]] = None):
        self._blocked_phrases = [
            phrase.lower()
            for phrase in (
                blocked_phrases
                or (
                    "google search",
                    "grounding metadata",
                    "search_exa",
                    "exa search",
                    "exa_search",
                )
            )
        ]
        self._interrupted = False

    async def update_settings(self, settings):
        if isinstance(settings, dict) and "blocked_phrases" in settings:
            phrases = settings.get("blocked_phrases") or []
            self._blocked_phrases = [str(phrase).lower() for phrase in phrases]

    async def filter(self, text: str) -> str:
        if not text or self._interrupted:
            return text

        sentences = self._split_sentences(text)
        kept_segments = [
            sentence for sentence in sentences if not self._contains_blocked_phrase(sentence)
        ]
        cleaned = " ".join(segment.strip() for segment in kept_segments if segment.strip())
        return re.sub(r"\s{2,}", " ", cleaned).strip()

    async def handle_interruption(self):
        self._interrupted = True

    async def reset_interruption(self):
        self._interrupted = False

    def _split_sentences(self, text: str):
        return self._SPLIT_PATTERN.split(text.strip())

    def _contains_blocked_phrase(self, sentence: str) -> bool:
        lowered = sentence.lower()
        return any(phrase in lowered for phrase in self._blocked_phrases)


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_filter=_maybe_create_krisp_filter(),
        # Use a shorter stop window and add smart-turn for reliable barge-in.
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),  # Reduced from 0.2 to 0.1
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_filter=_maybe_create_krisp_filter(),
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),  # Reduced from 0.2 to 0.1
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_filter=_maybe_create_krisp_filter(),
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),  # Reduced from 0.2 to 0.1
        turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams()),
    ),
}


async def run_bot(
    transport: BaseTransport,
    runner_args: RunnerArguments,
    user_id: Optional[str] = None,
    chemistry_id: Optional[str] = None,
):
    logger.info(
        "Starting bot (user_id={}, chemistry_id={})",
        user_id or "(missing)",
        chemistry_id or "(missing)",
    )
    # if not user_id:
    #     logger.warning(
    #         "No user_id provided; defaulting to test user {} for local validation.",
    #         TEST_USER_ID_FALLBACK,
    #     )
    # effective_user_id = user_id or TEST_USER_ID_FALLBACK
    effective_user_id = user_id  # Use real user_id only, no fallback
    recording_urls: Dict[str, str] = {}
    call_start_utc: Optional[datetime] = None
    call_end_utc: Optional[datetime] = None
    call_session_stamp: Optional[str] = None
    call_persisted = False

    user_name, activity_text, session_summaries, user_notes = _load_user_name_and_summaries(
        RUMIK_SUMMARY_LIMIT,
        user_id=effective_user_id,
        chemistry_id=chemistry_id,
    )
    display_name = user_name or "friend"

    if session_summaries:
        logger.info("Loaded {} session summaries from Mongo.", len(session_summaries))
    else:
        logger.info("No session summaries found; using default greeting.")

    if user_notes:
        logger.info("Loaded {} user facts from Mongo.", len(user_notes))

    # Load last 3 call summaries for this user and prioritize them in history
    call_summaries = _load_recent_call_summaries(effective_user_id, limit=3)
    if call_summaries:
        logger.info("Loaded {} recent call summaries from Mongo.", len(call_summaries))

    call_summary = _format_summary_context(call_summaries) if call_summaries else ""
    text_summary = (
        _format_summary_context(session_summaries) if session_summaries else ""
    )
    user_info_context = _format_user_information(user_notes)
    behaviour_hint = ""
    if user_info_context:
        behaviour_hint = (
            "Use these user facts naturally whenever the conversation touches them, but do not mention that you have a stored list."
        )

    # Detect conversation style from recent summaries for adaptive behavior
    # Extract recent message content from summaries for style detection
    recent_texts_for_style: List[str] = []
    for summary in (session_summaries or [])[:2]:  # Last 2 text sessions
        summary_text = summary.get("text") or summary.get("summary") or ""
        if summary_text:
            recent_texts_for_style.append(str(summary_text))
    for summary in (call_summaries or [])[:2]:  # Last 2 calls
        summary_text = summary.get("text") or summary.get("summary") or ""
        if summary_text:
            recent_texts_for_style.append(str(summary_text))

    # Get behavior routing style hint
    detected_style, style_hint = _detect_conversation_style(recent_texts_for_style)
    if style_hint:
        logger.debug("Detected conversation style: {} with hint", detected_style.value)

    combined_additional = "\n\n".join(
        [section for section in (user_info_context, behaviour_hint, style_hint) if section]
    )

    # Use dynamic activity based on time if MongoDB activity is missing or too verbose
    # MongoDB sometimes returns long LLM-generated activities that are too detailed
    # We prefer our short, natural time-based activities for voice calls
    if activity_text and len(activity_text) < 100:
        # Use MongoDB activity if it's short and natural
        current_activity_text = activity_text
    else:
        # Use our dynamic time-based activity generator
        now_ist = datetime.now(IST_TIMEZONE)
        current_activity_text = _get_dynamic_activity(now_ist.strftime("%H:%M"))

    now_ist = datetime.now(IST_TIMEZONE)
    live_call_context = _build_live_call_context(
        now_ist=now_ist,
        call_summaries=call_summaries or [],
        session_summaries=session_summaries or [],
    )
    if live_call_context:
        sections = [segment for segment in (combined_additional, live_call_context) if segment]
        combined_additional = "\n\n".join(sections)

    current_day = now_ist.strftime("%A")
    current_date = now_ist.strftime("%Y-%m-%d")
    current_time = now_ist.strftime("%H:%M")
    system_instruction = build_system_instruction(
        display_name,
        # Keep history_summary for backward-compat but send distinct summaries explicitly
        history_summary=text_summary,
        call_summary=call_summary,
        text_summary=text_summary,
        current_time=current_time,
        current_day=current_day,
        current_date=current_date,
        current_activity=current_activity_text,
        additional_context=combined_additional or None,
    )

    llm = GeminiLiveVertexLLMService(
        system_instruction=system_instruction,
        model="gemini-live-2.5-flash",
        params=InputParams(modalities=GeminiModalities.TEXT),
        project_id="rumik-ai",
        location="us-central1",
        credentials_path=_MATERIALIZED_VERTEX_CREDS
    )

    tts = EchoTTSService(
        
        server_url=os.getenv("ECHO_SERVER_URL"),
        voice=os.getenv("ECHO_VOICE", "expresso_02_ex03-ex01_calm_005"),
        cfg_scale_text=float(os.getenv("ECHO_CFG_SCALE_TEXT", "2.5")),
        cfg_scale_speaker=float(os.getenv("ECHO_CFG_SCALE_SPEAKER", "5.0")),
        seed=int(os.getenv("ECHO_SEED", "0")),
        sample_rate=44100,
        #text_filters=[ToolAnnouncementFilter()],
    )

    audio_buffer = AudioBufferProcessor(num_channels=1, enable_turn_audio=True)
    audio_session_stamp: Optional[str] = None
    session_timer_task: Optional[asyncio.Task] = None

    merged_audio_capture = {
        "buffer": bytearray(),
        "sample_rate": None,
        "num_channels": None,
    }
    user_audio_capture = {
        "buffer": bytearray(),
        "sample_rate": None,
    }
    assistant_audio_capture = {
        "buffer": bytearray(),
        "sample_rate": None,
    }

    def reset_audio_captures() -> None:
        merged_audio_capture["buffer"].clear()
        merged_audio_capture["sample_rate"] = None
        merged_audio_capture["num_channels"] = None
        user_audio_capture["buffer"].clear()
        user_audio_capture["sample_rate"] = None
        assistant_audio_capture["buffer"].clear()
        assistant_audio_capture["sample_rate"] = None

    def persist_audio_recordings(stamp: str) -> None:
        nonlocal call_session_stamp
        if not stamp:
            return
        call_session_stamp = call_session_stamp or stamp
        recordings: List[Tuple[str, bytes, Optional[int], Optional[int]]] = []

        merged_bytes = bytes(merged_audio_capture["buffer"])
        merged_rate = merged_audio_capture["sample_rate"]
        merged_channels = merged_audio_capture["num_channels"]
        if merged_bytes and merged_rate:
            recordings.append(
                ("merged", merged_bytes, merged_rate, merged_channels or 1)
            )

        user_bytes = bytes(user_audio_capture["buffer"])
        user_rate = user_audio_capture["sample_rate"]
        if user_bytes and user_rate:
            recordings.append(("user", user_bytes, user_rate, 1))

        assistant_bytes = bytes(assistant_audio_capture["buffer"])
        assistant_rate = assistant_audio_capture["sample_rate"]
        if assistant_bytes and assistant_rate:
            recordings.append(("assistant", assistant_bytes, assistant_rate, 1))

        if not recordings:
            reset_audio_captures()
            return

        reset_audio_captures()

        for file_type, audio_bytes, sample_rate, num_channels in recordings:
            async def _persist(
                audio_bytes=audio_bytes,
                sample_rate=sample_rate,
                num_channels=num_channels,
                file_type=file_type,
            ):
                path = _recordings_dir() / f"{stamp}_{file_type}.wav"
                await _write_wav_file(path, audio_bytes, sample_rate or 16000, num_channels or 1)
                logger.debug("Saved {} audio to {}", file_type, path)
                await _upload_and_cleanup(path, stamp, file_type)

            _track_background(
                _persist(), name=f"persist-{file_type}-{stamp}"
            )

    async def stop_session_timer():
        nonlocal session_timer_task
        if session_timer_task and not session_timer_task.done():
            session_timer_task.cancel()
            try:
                await session_timer_task
            except asyncio.CancelledError:
                pass
        session_timer_task = None

    async def manage_session_timer():
        nonlocal session_timer_task

        total_duration = max(0, SESSION_MAX_DURATION_SECS)
        warning_lead = max(0, SESSION_WARNING_LEAD_SECS)

        if total_duration <= 0:
            session_timer_task = None
            return

        if warning_lead >= total_duration:
            warning_lead = max(0, total_duration - 1)

        warning_delay = max(0, total_duration - warning_lead)

        warning_message = (
            f"Hey, I need to head out now. It was great talking to you. I'll call you later. Bye, {display_name}"
        )

        try:
            if warning_delay:
                await asyncio.sleep(warning_delay)

            if warning_lead:
                logger.info("Session timer warning: {}s remaining", warning_lead)
                _shared_state["tts_suppressed"] = False
                await task.queue_frames([TTSSpeakFrame(warning_message)])
                await asyncio.sleep(warning_lead)

            logger.info("Session timer ending conversation after {}s", total_duration)
            _shared_state["tts_suppressed"] = False
            await task.queue_frames([EndFrame()])
        except asyncio.CancelledError:
            logger.info("Session timer cancelled before completion")
            raise
        except Exception as exc:
            logger.error("Session timer encountered an error: {}", exc)
        finally:
            session_timer_task = None

    # Shared state for interruption gating and suppression
    _shared_state = {
        "llm_active": False,
        "tts_active": False,
        "tts_suppressed": False,
        "last_interrupt_ms": 0,
        "debounce_ms": int(os.getenv("INTERRUPT_DEBOUNCE_MS", "320")),
        "cooldown_ms": int(os.getenv("INTERRUPT_COOLDOWN_MS", "600")),
        "interrupted_turn": False,
        "last_llm_ms": 0,
        "last_tts_audio_ms": 0,
        "last_user_start_ms": 0,
        "last_user_stop_ms": 0,
        "last_idle_prompt_ms": 0,
        "llm_clear_ms": _env_int("LLM_ACTIVE_CLEAR_MS", 350),  # Reduce from 1500ms to 100ms
        "tts_clear_ms": _env_int("TTS_ACTIVE_CLEAR_MS", 350),  # Reduce from 1200ms to 100ms
    }

    pending_background_tasks: set[asyncio.Task] = set()
    background_concurrency = max(1, _env_int("AUDIO_TASK_CONCURRENCY", 3))
    background_semaphore = asyncio.Semaphore(background_concurrency)
    logger.debug("Audio background task concurrency set to {}", background_concurrency)

    def _track_background(coro: Awaitable[Any], *, name: str) -> None:
        async def _guarded():
            async with background_semaphore:
                return await coro

        task = asyncio.create_task(_guarded())
        pending_background_tasks.add(task)

        def _on_done(fut: asyncio.Future):
            pending_background_tasks.discard(task)
            try:
                fut.result()
            except asyncio.CancelledError:
                logger.debug("Background task {} cancelled", name)
            except Exception as exc:
                logger.error("Background task {} failed: {}", name, exc)

        task.add_done_callback(_on_done)

    idle_timeout_secs = 9.0
    idle_retry_limit = 3.0
    idle_gap_secs = 9.0

    async def handle_user_idle(processor, retry_count: int):
        try:
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        except Exception:
            now_ms = 0

        # Enforce idle timeout relative to last user stop
        idle_ms = int(idle_timeout_secs * 1000)
        gap_ms = int(idle_gap_secs * 1000)
        last_user_stop_ms = _shared_state.get("last_user_stop_ms", 0)
        if last_user_stop_ms and now_ms - last_user_stop_ms < idle_ms:
            logger.debug(
                "[idle] waiting full {}ms after user stop; elapsed={}ms",
                idle_ms,
                now_ms - last_user_stop_ms,
            )
            return True

        # Ensure minimum spacing between reminders
        last_idle_prompt_ms = _shared_state.get("last_idle_prompt_ms", 0)
        if last_idle_prompt_ms and now_ms - last_idle_prompt_ms < gap_ms:
            logger.debug(
                "[idle] spacing guard active; need {}ms more",
                gap_ms - (now_ms - last_idle_prompt_ms),
            )
            return True

        last_audio_ms = _shared_state.get("last_tts_audio_ms", 0)
        clear_after = max(0, _shared_state.get("tts_clear_ms", 1200))
        if _shared_state.get("tts_active"):
            if last_audio_ms and now_ms and now_ms - last_audio_ms >= clear_after:
                logger.debug(
                    "[idle] auto-clearing stale tts_active; now_ms={} last_audio_ms={} delta={}",
                    now_ms,
                    last_audio_ms,
                    now_ms - last_audio_ms,
                )
                _shared_state["tts_active"] = False
            elif not last_audio_ms:
                logger.debug("[idle] clearing tts_active with no timestamp present")
                _shared_state["tts_active"] = False

        last_llm_ms = _shared_state.get("last_llm_ms", 0)
        llm_clear_after = max(0, _shared_state.get("llm_clear_ms", 1500))
        if (
            _shared_state.get("llm_active")
            and last_llm_ms
            and now_ms
            and now_ms - last_llm_ms >= llm_clear_after
        ):
            logger.debug(
                "[idle] auto-clearing stale llm_active; now_ms={} last_llm_ms={} delta={}",
                now_ms,
                last_llm_ms,
                now_ms - last_llm_ms,
            )
            _shared_state["llm_active"] = False

        if _shared_state.get("llm_active") or _shared_state.get("tts_active"):
            logger.debug(
                "[idle] skipping reminder; llm_active={} tts_active={} retry={}",
                _shared_state.get("llm_active"),
                _shared_state.get("tts_active"),
                retry_count,
            )
            return True

        if retry_count >= idle_retry_limit:
            logger.info("[idle] retry limit reached; ending session without farewell prompt")
            _shared_state["tts_suppressed"] = False
            await processor.push_frame(EndFrame(), FrameDirection.UPSTREAM)
            return False

        prompts_for_retry = HINGLISH_IDLE_PROMPTS.get(retry_count)
        if not prompts_for_retry:
            logger.debug("[idle] no prompts configured for retry={}; continuing monitoring", retry_count)
            return True

        message = random.choice(prompts_for_retry).format(user_name=display_name)
        logger.info("[idle] reminder sent (retry {}): {}", retry_count, message)
        _shared_state["tts_suppressed"] = False
        await processor.push_frame(TTSSpeakFrame(message))
        _shared_state["tts_active"] = True
        _shared_state["last_tts_audio_ms"] = now_ms
        _shared_state["last_idle_prompt_ms"] = now_ms
        return True

    class BargeInOnUserStart(FrameProcessor):
        """Inject InterruptionFrame when user starts speaking to barge-in.

        Uses a small debounce to avoid spamming multiple interruptions
        when VAD fluctuates at speech onset.
        """

        def __init__(self, shared_state: dict):
            super().__init__()
            self._state = shared_state

        async def process_frame(self, frame, direction):
            # Call parent so transcript updates keep working as before.
            await super().process_frame(frame, direction)

            try:
                now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            except Exception:
                now_ms = 0
                

            # Track user activity moments
            if isinstance(frame, UserStartedSpeakingFrame):
                self._state["last_user_start_ms"] = now_ms
            if isinstance(frame, UserStoppedSpeakingFrame):
                self._state["last_user_stop_ms"] = now_ms

            if isinstance(frame, UserStartedSpeakingFrame):
                debounce_ok = (now_ms - self._state.get("last_started_ms", 0)) >= self._state["debounce_ms"]
                cooldown_ok = (now_ms - self._state.get("last_interrupt_ms", 0)) >= self._state["cooldown_ms"]
                self._state["last_started_ms"] = now_ms

                should_interrupt = (
                    debounce_ok
                    and cooldown_ok
                    and (self._state.get("llm_active") or self._state.get("tts_active"))
                )

                if should_interrupt:
                    logger.debug(
                        "[barge-in] start detected; interrupt llm_active={} tts_active={}",
                        self._state.get("llm_active"),
                        self._state.get("tts_active"),
                    )
                    self._state["interrupted_turn"] = True
                    self._state["last_interrupt_ms"] = now_ms
                    # Stop any pending generation locally
                    self._state["llm_active"] = False
                    self._state["tts_active"] = False
                    # Send interruption downstream to cancel ongoing LLM/TTS.
                    await self.push_frame(InterruptionFrame(), direction)

            # Always forward the original frame to keep normal behavior.
            await self.push_frame(frame, direction)

    class LLMActivityTracker(FrameProcessor):
        """Track LLM activity to improve interruption gating."""

        def __init__(self, shared_state: dict):
            super().__init__()
            self._state = shared_state

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            # Track when LLM begins and ends responding
            try:
                from pipecat.frames.frames import (
                    LLMFullResponseStartFrame,
                    LLMFullResponseEndFrame,
                )
            except Exception:
                LLMFullResponseStartFrame = type("LLMFullResponseStartFrame", (), {})
                LLMFullResponseEndFrame = type("LLMFullResponseEndFrame", (), {})

            try:
                now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            except Exception:
                now_ms = 0

            if isinstance(frame, (LLMRunFrame, LLMFullResponseStartFrame)):
                self._state["llm_active"] = True
                self._state["tts_suppressed"] = False
                if now_ms:
                    self._state["last_llm_ms"] = now_ms
            elif isinstance(frame, (LLMFullResponseEndFrame, TTSSpeakFrame, EndFrame)):
                self._state["llm_active"] = False
                if now_ms:
                    self._state["last_llm_ms"] = now_ms
            elif isinstance(frame, InterruptionFrame):
                # Interruption cancels current turn.
                self._state["llm_active"] = False
                if now_ms:
                    self._state["last_llm_ms"] = now_ms
                self._state["tts_suppressed"] = True
            elif self._state.get("llm_active") and now_ms:
                last_llm_ms = self._state.get("last_llm_ms", 0)
                clear_after = max(0, self._state.get("llm_clear_ms", 1500))
                if last_llm_ms and now_ms - last_llm_ms >= clear_after:
                    logger.debug(
                        "[idle] tracker clearing stale llm_active; now_ms={} last_llm_ms={} delta={}",
                        now_ms,
                        last_llm_ms,
                        now_ms - last_llm_ms,
                    )
                    self._state["llm_active"] = False
                    self._state["last_llm_ms"] = now_ms

            await self.push_frame(frame, direction)

    class GeminiGroundingLogger(FrameProcessor):
        """Log Gemini grounding metadata for visibility in the pipeline."""

        def __init__(self):
            super().__init__()

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, LLMSearchResponseFrame) and direction is FrameDirection.DOWNSTREAM:
                preview = (frame.search_result or "").strip()
                if preview:
                    cleaned_preview = re.sub(r"\s+", " ", preview)
                    if len(cleaned_preview) > 180:
                        cleaned_preview = f"{cleaned_preview[:177]}..."
                    logger.info(f"[Grounding] response preview: {cleaned_preview}")

                for origin in frame.origins or []:
                    site = origin.site_title or origin.site_uri or "unknown source"
                    logger.info(f"[Grounding] ☆ {site}")
                    for result in origin.results or []:
                        snippet = (result.text or "").strip()
                        if not snippet:
                            continue
                        snippet = re.sub(r"\s+", " ", snippet)
                        if len(snippet) > 160:
                            snippet = f"{snippet[:157]}..."
                        logger.info(f"[Grounding]    ↳ {snippet}")

                if frame.rendered_content:
                    logger.info("[Grounding] rendered search card available")

            await self.push_frame(frame, direction)

    class TTSActivityProbe(FrameProcessor):
        """Best-effort TTS activity tracking using audio frames after TTS."""

        def __init__(self, shared_state: dict):
            super().__init__()
            self._state = shared_state

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, BotStoppedSpeakingFrame):
                logger.debug("[tts-probe] Bot stopped speaking – clearing tts_active")
                self._state["tts_active"] = False
                self._state["last_tts_audio_ms"] = 0
                self._state["tts_suppressed"] = True
                await self.push_frame(frame, direction)
                return
            # Heuristic: if a frame has raw audio payload attributes post-TTS,
            # assume assistant audio is being sent.
            has_audio = any(
                hasattr(frame, attr) for attr in ("audio", "samples", "pcm")
            ) and any(
                hasattr(frame, attr) for attr in ("sample_rate", "num_channels")
            )

            if has_audio and not self._state.get("tts_suppressed"):
                try:
                    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                except Exception:
                    now_ms = 0
                self._state["tts_active"] = True
                self._state["last_tts_audio_ms"] = now_ms
            elif isinstance(frame, InterruptionFrame):
                self._state["tts_active"] = False
            else:
                try:
                    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                except Exception:
                    now_ms = 0
                last_audio_ms = self._state.get("last_tts_audio_ms", 0)
                clear_after = max(0, self._state.get("tts_clear_ms", 1200))
                if (
                    self._state.get("tts_active")
                    and last_audio_ms
                    and now_ms - last_audio_ms >= clear_after
                ):
                    self._state["tts_active"] = False

            await self.push_frame(frame, direction)

    class AssistantPartialSuppressor(FrameProcessor):
        """Suppress partial assistant text in context on interruption."""

        def __init__(self, shared_state: dict):
            super().__init__()
            self._state = shared_state

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)

            try:
                from pipecat.frames.frames import TextFrame
            except Exception:
                TextFrame = type("TextFrame", (), {})

            # If an interruption happened mid-turn, drop assistant TextFrame chunks
            # so they aren't persisted to context.
            if self._state.get("interrupted_turn") and isinstance(frame, TextFrame):
                return  # swallow

            # Reset suppression on new user-run
            if isinstance(frame, LLMRunFrame):
                self._state["interrupted_turn"] = False

            await self.push_frame(frame, direction)

    async def _upload_and_cleanup(path: Path, session_id: str, file_type: str) -> None:
        nonlocal recording_urls
        upload_url = await _upload_to_gcs(path, session_id, file_type, effective_user_id)
        if upload_url:
            recording_urls[file_type] = upload_url
        if upload_url and path.exists():
            try:
                path.unlink()
                logger.debug("Deleted local file after upload: {}", path)
            except Exception as exc:
                logger.error("Failed to delete local file {}: {}", path, exc)

    @audio_buffer.event_handler("on_audio_data")
    async def handle_merged_audio(_, audio: bytes, sample_rate: int, num_channels: int):
        if not audio_session_stamp:
            return

        if not audio:
            return

        merged_audio_capture["buffer"].extend(audio)
        if merged_audio_capture["sample_rate"] is None and sample_rate:
            merged_audio_capture["sample_rate"] = sample_rate
        if merged_audio_capture["num_channels"] is None and num_channels:
            merged_audio_capture["num_channels"] = num_channels

    @audio_buffer.event_handler("on_track_audio_data")
    async def handle_track_audio(
        _, user_audio: bytes, bot_audio: bytes, sample_rate: int, num_channels: int
    ):
        if not audio_session_stamp:
            return
        if user_audio:
            user_audio_capture["buffer"].extend(user_audio)
            if user_audio_capture["sample_rate"] is None and sample_rate:
                user_audio_capture["sample_rate"] = sample_rate
        if bot_audio:
            assistant_audio_capture["buffer"].extend(bot_audio)
            if assistant_audio_capture["sample_rate"] is None and sample_rate:
                assistant_audio_capture["sample_rate"] = sample_rate

    # Generate an engaging greeting using our templates
    example_greeting = generate_greeting(display_name, include_activity=True)

    if session_summaries:
        # Returning user - use a more familiar, playful greeting
        greeting_instruction = (
            f"Greet {display_name} simply. Use English only. "
            f"Example: \"{example_greeting}\"."
        )
    else:
        # New user - still engaging but slightly more welcoming
        greeting_instruction = (
            f"This is your first call with {display_name}. Greet them simply. "
            f"Use English only. Example: \"{example_greeting}\"."
        )

    messages = [
        {
            "role": "user",
            "content": greeting_instruction,
        },
    ]

    # Set up conversation context and management
    # The context_aggregator will automatically collect conversation context
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)
    transcript = TranscriptProcessor()

    user_context_processor = context_aggregator.user()
    assistant_context_processor = context_aggregator.assistant()

    user_idle = UserIdleProcessor(
        callback=handle_user_idle,
        timeout=idle_timeout_secs,
    )
    barge_in = BargeInOnUserStart(_shared_state)
    llm_activity = LLMActivityTracker(_shared_state)
    #grounding_logger = GeminiGroundingLogger()
    tts_probe = TTSActivityProbe(_shared_state)
    assistant_suppressor = AssistantPartialSuppressor(_shared_state)

    pipeline = Pipeline(
        [
            transport.input(),
            transcript.user(),
            user_idle,
            barge_in,
            user_context_processor,
            llm,
            #grounding_logger,
            llm_activity,
            tts,
            tts_probe,
            transport.output(),
            audio_buffer,
            transcript.assistant(),
            assistant_suppressor,
            assistant_context_processor,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        nonlocal audio_session_stamp, session_timer_task, call_start_utc, call_end_utc, call_session_stamp
        logger.info("Client connected")
        reset_audio_captures()
        audio_session_stamp = datetime.now(IST_TIMEZONE).strftime("%Y%m%d_%H%M%S")
        call_session_stamp = audio_session_stamp
        call_start_utc = datetime.now(timezone.utc)
        call_end_utc = None
        await audio_buffer.start_recording()
        await stop_session_timer()
        session_timer_task = asyncio.create_task(manage_session_timer())
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        nonlocal audio_session_stamp, call_end_utc
        logger.info("Client disconnected")
        await audio_buffer.stop_recording()
        await stop_session_timer()
        call_end_utc = datetime.now(timezone.utc)
        if audio_session_stamp:
            persist_audio_recordings(audio_session_stamp)
            audio_session_stamp = None
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    try:
        await runner.run(task)
    finally:
        await stop_session_timer()
        if audio_session_stamp:
            persist_audio_recordings(audio_session_stamp)
            audio_session_stamp = None
        if pending_background_tasks:
            await asyncio.gather(
                *pending_background_tasks, return_exceptions=True
            )
        if not call_persisted and call_start_utc:
            if not call_end_utc:
                call_end_utc = datetime.now(timezone.utc)

            session_identifier = getattr(runner_args, "session_id", None)
            body_payload = getattr(runner_args, "body", None)
            if not session_identifier and isinstance(body_payload, dict):
                session_identifier = (
                    body_payload.get("session_id")
                    or body_payload.get("sessionId")
                )
            session_identifier = session_identifier or call_session_stamp

            try:
                await _persist_call_documents(
                    user_identifier=effective_user_id,
                    chemistry_identifier=chemistry_id,
                    session_identifier=session_identifier,
                    session_label=call_session_stamp,
                    call_start=call_start_utc,
                    call_end=call_end_utc,
                    recording_urls=recording_urls.copy(),
                )

                # Extract user information from the call
                # Note: For Gemini Live, user transcription happens within the LLM,
                # so we use the recent text summaries as a proxy for extracting facts.
                # This will capture facts from both recent text chats and inferred from
                # the call context. True in-call extraction would require post-call
                # transcription of the audio recording.
                if chemistry_id and session_summaries:
                    try:
                        # Use recent text summaries for fact extraction
                        summary_texts = [
                            str(s.get("text") or s.get("summary") or "")
                            for s in session_summaries[:3]
                            if s.get("text") or s.get("summary")
                        ]
                        if summary_texts:
                            extracted_facts = extract_user_information_from_messages(summary_texts)
                            extracted_instructions = extract_long_term_instructions(summary_texts)

                            if extracted_facts or extracted_instructions:
                                logger.info(
                                    "Extracted {} facts and {} instructions from conversation",
                                    len(extracted_facts), len(extracted_instructions)
                                )
                                await _persist_extracted_user_facts(
                                    chemistry_identifier=chemistry_id,
                                    extracted_facts=extracted_facts,
                                    extracted_instructions=extracted_instructions,
                                )
                    except Exception as extract_exc:
                        logger.warning("Failed to extract user information: {}", extract_exc)

            except Exception as exc:
                logger.error("Failed to persist call artifacts: {}", exc)
            finally:
                call_persisted = True


async def bot(runner_args: DailySessionArguments):
    """Main bot entry point compatible with Pipecat Cloud Daily sessions."""
    logger.debug(
        "Runner args class=%s",
        f"{type(runner_args).__module__}.{type(runner_args).__name__}",
    )
    logger.debug(
        f"Runner args class={type(runner_args).__module__}.{type(runner_args).__name__}"
    )
    logger.debug("Runner args dict=%s", getattr(runner_args, "__dict__", None))
    logger.debug(f"Runner args dict={getattr(runner_args, '__dict__', None)}")
    transport = await create_transport(runner_args, transport_params)

    user_id: Optional[str] = None
    chemistry_id: Optional[str] = None

    body_payload = getattr(runner_args, "body", None)
    if isinstance(body_payload, dict):
        user_id = body_payload.get("user_id") or body_payload.get("userId")
        chemistry_id = body_payload.get("chemistry_id") or body_payload.get("chemistryId")
    else:
        logger.warning("Runner arguments body missing or not a dict; using default Mongo IDs.")

    logger.info(
        "Bot invoked with user_id={} chemistry_id={}",
        user_id or "(missing)",
        chemistry_id or "(missing)",
    )
    logger.debug("Raw session arguments body={!r}", body_payload)
    logger.debug(f"Raw session arguments body={body_payload!r}")

    await run_bot(transport, runner_args, user_id=user_id, chemistry_id=chemistry_id)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

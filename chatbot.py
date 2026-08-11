"""
SONATA Simulation Config Chatbot — core logic
=============================================
This module exposes two reusable functions consumed by both the CLI
entry-point and the FastAPI server:

  - chat_turn(history, user_message) -> (reply, config_or_None)
  - extract_config(history)           -> (config_or_None, error_str_or_None)

Usage (CLI)
-----------
    GROQ_API_KEY=<your_key> python chatbot.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

import libsonata
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from pydantic import ValidationError

from data_model import SimulationConfig

# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------


def _get_schema_text() -> str:
    """Generate a compact JSON schema string from the Pydantic model, without descriptions."""

    def _strip_descriptions(obj):
        """Recursively remove 'description' and 'title' keys to reduce token count."""
        if isinstance(obj, dict):
            return {k: _strip_descriptions(v) for k, v in obj.items() if k not in ("description", "title", "examples")}
        if isinstance(obj, list):
            return [_strip_descriptions(item) for item in obj]
        return obj

    schema = SimulationConfig.model_json_schema()
    compact = _strip_descriptions(schema)
    return json.dumps(compact, separators=(",", ":"))


_SCHEMA_TEXT = _get_schema_text()

SYSTEM_PROMPT = (
    """\
You are an expert assistant for the SONATA neural circuit simulation framework
(BBP extension). Your job is to help users build a valid SONATA
simulation_config.json file by asking them questions in a friendly,
conversational way.

SONATA simulation config key sections:
- run        : tstop (ms), dt (ms), random_seed — all MANDATORY
- output     : output_dir, spikes_file (optional, have defaults)
- conditions : celsius, v_init, spike_location, extracellular_calcium (all optional)
- inputs     : named stimulus blocks (module: linear, pulse, noise, shot_noise, etc.)
- reports    : named data-collection blocks (type: compartment, summation, lfp, etc.)
- connection_overrides : adjust synaptic weights between node sets

Conversation strategy
---------------------
1. Start by asking for the three mandatory run parameters (tstop, dt, random_seed).
2. Ask which node_set to simulate (which population of cells). Explain that leaving
   it empty means all non-virtual nodes will be loaded.
3. Ask about conditions (temperature, v_init, spike_location).
4. Ask if they want any inputs (stimuli) — if yes, ask for details per stimulus.
   Always ask which node_set each input targets.
5. Ask if they want reports — if yes, ask for details per report.
   Always ask which node_set each report covers.
6. Always ask about connection_overrides — whether the user wants to adjust
   synaptic weights between any populations. Do not skip this step.
7. Once you have enough information, output ONLY a JSON code block containing
   the complete simulation_config, nothing else. The block must be valid JSON
   that conforms to the SONATA spec.

Important rules
---------------
- Ask one topic at a time; do not overwhelm the user with many questions at once.
- Use sensible defaults when the user is unsure (dt=0.025, celsius=34, v_init=-80).
- When ready to produce the config, output exactly one fenced JSON block.
  Do not include any text before or after the JSON block when generating the final config.
- Field names must match the JSON schema below EXACTLY (snake_case).
- Enum values must be lowercase strings as defined in the schema.
- ONLY use field names and types defined in the schema below. Do not invent fields.

EXACT JSON SCHEMA (from data_model.py — this is the ground truth):
"""
    + _SCHEMA_TEXT
)

EXTRACT_PROMPT = """\
You are a data-extraction assistant. Given the conversation history below,
extract the SONATA simulation configuration and return ONLY a valid JSON object
(no markdown, no explanation) that matches the SONATA simulation_config schema.

Rules:
- Include only fields that were explicitly discussed or have clear defaults.
- "run" must always include tstop, dt, and random_seed.
- All enum values must be lowercase strings (e.g. "euler", "soma", "by_time").
- Do not invent values the user never mentioned.
- Return raw JSON only — no ```json fences, no commentary.

Conversation:
{history}
"""

FIX_PROMPT = """\
The following JSON was generated for a SONATA simulation config but failed validation.

INVALID JSON:
{invalid_json}

VALIDATION ERROR:
{error}

Fix the JSON so it passes validation. Return ONLY the corrected raw JSON — \
no markdown fences, no explanation, no commentary.
"""

MAX_FIX_ATTEMPTS = 2

# ---------------------------------------------------------------------------
# Module-level LLM (initialised lazily)
# ---------------------------------------------------------------------------

_state: dict[str, ChatGroq | None] = {"llm": None}

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 2.0

RATE_LIMIT_MSG = (
    "Rate limit reached — Groq's free tier has a per-minute token cap. Please wait about a minute and try again."
)


def _invoke_with_retry(chain, kwargs: dict, retries: int = MAX_RETRIES) -> str:
    """Invoke a LangChain chain with retries on transient errors. Fails fast on rate limits."""
    for attempt in range(1, retries + 1):
        try:
            return chain.invoke(kwargs)
        except Exception as exc:
            err_str = str(exc).lower()

            # Rate limit — fail immediately, retries won't help within the same minute
            if "rate_limit" in err_str or "429" in err_str:
                raise RuntimeError(err_str) from exc

            # Transient errors — retry with backoff
            is_retryable = any(keyword in err_str for keyword in ("timeout", "timed out", "connection", "503"))
            if is_retryable and attempt < retries:
                wait = RETRY_DELAY_SECONDS * attempt
                logger.warning("LLM call failed (attempt %d/%d): %s — retrying in %.1fs", attempt, retries, exc, wait)
                time.sleep(wait)
            else:
                raise
    return ""  # unreachable, satisfies type checker


def _get_llm() -> ChatGroq:
    if _state["llm"] is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise OSError(
                "GROQ_API_KEY environment variable is not set. Get a free key at https://console.groq.com/",
            )
        _state["llm"] = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            api_key=api_key,
        )
    return _state["llm"]


def _build_chat_chain():
    llm = _get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ],
    )
    return prompt | llm | StrOutputParser()


def _build_extract_chain():
    llm = _get_llm()
    prompt = ChatPromptTemplate.from_template(EXTRACT_PROMPT)
    return prompt | llm | StrOutputParser()


def _build_fix_chain():
    llm = _get_llm()
    prompt = ChatPromptTemplate.from_template(FIX_PROMPT)
    return prompt | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def history_as_text(messages: list) -> str:
    """Convert a LangChain message list to a plain-text transcript."""
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"Assistant: {m.content}")
    return "\n".join(lines)


def extract_json_from_reply(text: str) -> str | None:
    """Return the first ```json ... ``` block from a model reply, or None."""
    match = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL).search(text)
    return match.group(1).strip() if match else None


def validate_config(raw_json: str) -> tuple[SimulationConfig | None, str | None]:
    """
    Parse raw JSON into SimulationConfig.
    Returns (config, None) on success, (None, error_message) on failure.
    """
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc}"

    try:
        config = SimulationConfig.model_validate(data)
    except ValidationError as exc:
        return None, str(exc)

    try:
        libsonata.SimulationConfig(raw_json, "./")
        logger.info("libsonata validated!")
    except libsonata.SonataError as exc:
        return None, f"libsonata parse error: {exc}"

    return config, None


def _try_fix_config(raw_json: str, error: str) -> tuple[SimulationConfig | None, str | None]:
    """
    Attempt to fix invalid JSON by sending it back to the LLM with the error.
    Retries up to MAX_FIX_ATTEMPTS times.
    """
    chain = _build_fix_chain()
    current_json = raw_json
    current_error = error

    for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
        logger.info("Auto-fix attempt %d/%d: %s", attempt, MAX_FIX_ATTEMPTS, current_error[:100])
        try:
            fixed = _invoke_with_retry(chain, {"invalid_json": current_json, "error": current_error})
        except Exception as exc:
            logger.exception("Fix attempt %d failed", attempt)
            return None, f"Auto-fix LLM call failed: {exc}"

        config, new_error = validate_config(fixed)
        if config is not None:
            logger.info("Auto-fix succeeded on attempt %d", attempt)
            return config, None

        # Prepare for next attempt
        current_json = fixed
        current_error = new_error or "Unknown error"

    return None, f"Auto-fix failed after {MAX_FIX_ATTEMPTS} attempts. Last error: {current_error}"


# ---------------------------------------------------------------------------
# Core API functions
# ---------------------------------------------------------------------------


def chat_turn(
    history: list,
    user_message: str,
) -> tuple[str, SimulationConfig | None]:
    """
    Send one user message, append to history in-place, return
    (assistant_reply, config_or_None).

    Config is non-None only when the model spontaneously produced a valid
    JSON block in its reply.
    """
    chain = _build_chat_chain()
    try:
        reply = _invoke_with_retry(chain, {"history": history, "input": user_message})
    except Exception as exc:
        logger.exception("chat_turn failed")
        reply = f"Sorry, I encountered an error communicating with the LLM: {exc}"

    history.append(HumanMessage(content=user_message))
    history.append(AIMessage(content=reply))

    config: SimulationConfig | None = None
    json_block = extract_json_from_reply(reply)
    if json_block:
        config, error = validate_config(json_block)
        if config is None and error:
            # Auto-fix: send the error back to the LLM
            config, _ = _try_fix_config(json_block, error)

    return reply, config


def extract_config(
    history: list,
) -> tuple[SimulationConfig | None, str | None]:
    """
    Run the extraction chain over the full conversation history.
    Returns (config, None) on success, (None, error_message) on failure.
    """
    chain = _build_extract_chain()
    try:
        raw = _invoke_with_retry(chain, {"history": history_as_text(history)})
    except Exception as exc:
        logger.exception("extract_config failed")
        return None, f"LLM request failed after {MAX_RETRIES} retries: {exc}"

    config, error = validate_config(raw)
    if config is not None:
        return config, None

    # Auto-fix: send the error back to the LLM
    return _try_fix_config(raw, error or "Unknown validation error")


def opening_message() -> str:
    """Return the assistant's first message to start a fresh conversation."""
    chain = _build_chat_chain()
    try:
        return _invoke_with_retry(
            chain,
            {
                "history": [],
                "input": "Hello! I'd like to set up a SONATA simulation.",
            },
        )
    except Exception as exc:
        logger.exception("opening_message failed")
        return f"Hello! I'm ready to help you build a SONATA config. (Note: LLM error occurred: {exc})"


# ---------------------------------------------------------------------------
# CLI entry point (kept for convenience)
# ---------------------------------------------------------------------------


def _run_cli() -> None:
    history: list = []

    print("=" * 60)
    print("  SONATA Simulation Config Chatbot")
    print("  Type 'generate' to produce the JSON file.")
    print("  Type 'quit' or 'exit' to quit.")
    print("=" * 60)
    print()

    greeting = opening_message()
    history.append(HumanMessage(content="Hello! I'd like to set up a SONATA simulation."))
    history.append(AIMessage(content=greeting))
    print(f"Assistant: {greeting}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        if user_input.lower() == "generate":
            print("\nAssistant: Generating your simulation config…\n")
            config, error = extract_config(history)
            if config:
                output_path = Path("simulation_config.json")
                output_path.write_text(
                    config.model_dump_json(indent=2, exclude_none=True),
                    encoding="utf-8",
                )
                print(f"Config saved to: {output_path.resolve()}\n")
                print(output_path.read_text())
            else:
                print(f"Generation failed:\n{error}\n")
            continue

        reply, config = chat_turn(history, user_input)
        print(f"\nAssistant: {reply}\n")

        if config:
            output_path = Path("simulation_config.json")
            output_path.write_text(
                config.model_dump_json(indent=2, exclude_none=True),
                encoding="utf-8",
            )
            print(f"[Auto-saved] Config written to: {output_path.resolve()}\n")


if __name__ == "__main__":
    _run_cli()

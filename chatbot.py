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
import os
import re
import sys
from pathlib import Path
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from pydantic import ValidationError

from data_model import SimulationConfig

# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert assistant for the SONATA neural circuit simulation framework.
Your job is to help users build a valid SONATA simulation_config.json file by
asking them questions in a friendly, conversational way.

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
2. Ask about conditions (temperature, v_init) and node_set.
3. Ask if they want any inputs (stimuli) — if yes, ask for details per stimulus.
4. Ask if they want reports — if yes, ask for details per report.
5. Ask about connection_overrides if relevant.
6. Once you have enough information, output ONLY a JSON code block containing
   the complete simulation_config, nothing else. The block must be valid JSON
   that conforms to the SONATA spec.

Important rules
---------------
- Ask one topic at a time; do not overwhelm the user with many questions at once.
- Use sensible defaults when the user is unsure (dt=0.025, celsius=34, v_init=-80).
- When ready to produce the config, output exactly one fenced JSON block:
  ```json
  { ... }
  ```
  Do not include any text before or after the JSON block when generating the final config.
- Field names must match the SONATA spec exactly (snake_case).
- Enum values must be lowercase strings as defined in the spec.
"""

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

# ---------------------------------------------------------------------------
# Module-level LLM (initialised lazily)
# ---------------------------------------------------------------------------

_llm: Optional[ChatGroq] = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY environment variable is not set. "
                "Get a free key at https://console.groq.com/"
            )
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            api_key=api_key,
        )
    return _llm


def _build_chat_chain():
    llm = _get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    return prompt | llm | StrOutputParser()


def _build_extract_chain():
    llm = _get_llm()
    prompt = ChatPromptTemplate.from_template(EXTRACT_PROMPT)
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
        return config, None
    except ValidationError as exc:
        return None, str(exc)


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
    reply = chain.invoke({"history": history, "input": user_message})

    history.append(HumanMessage(content=user_message))
    history.append(AIMessage(content=reply))

    config: SimulationConfig | None = None
    json_block = extract_json_from_reply(reply)
    if json_block:
        config, _ = validate_config(json_block)

    return reply, config


def extract_config(
    history: list,
) -> tuple[SimulationConfig | None, str | None]:
    """
    Run the extraction chain over the full conversation history.
    Returns (config, None) on success, (None, error_message) on failure.
    """
    chain = _build_extract_chain()
    raw = chain.invoke({"history": history_as_text(history)})
    return validate_config(raw)


def opening_message() -> str:
    """Return the assistant's first message to start a fresh conversation."""
    chain = _build_chat_chain()
    reply = chain.invoke(
        {
            "history": [],
            "input": "Hello! I'd like to set up a SONATA simulation.",
        }
    )
    return reply


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

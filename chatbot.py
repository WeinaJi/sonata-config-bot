"""
SONATA Simulation Config Chatbot
=================================
A conversational chatbot that gathers simulation requirements from the user
and produces a valid SONATA simulation_config.json.

Usage
-----
    GROQ_API_KEY=<your_key> python chatbot.py

Get a free Groq API key at: https://console.groq.com/
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Annotated

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
# Helpers
# ---------------------------------------------------------------------------


def _build_llm() -> ChatGroq:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print(
            "ERROR: GROQ_API_KEY environment variable is not set.\n"
            "Get a free key at https://console.groq.com/ then run:\n"
            "  export GROQ_API_KEY=<your_key>\n"
        )
        sys.exit(1)
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=api_key,
    )


def _history_as_text(messages: list) -> str:
    """Convert message list to a plain-text transcript for extraction."""
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"Assistant: {m.content}")
    return "\n".join(lines)


def _extract_json_from_reply(text: str) -> str | None:
    """Pull the first ```json ... ``` block out of a model reply, or return None."""
    import re
    pattern = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def _validate_and_save(raw_json: str, output_path: Path) -> SimulationConfig | None:
    """Parse raw JSON into SimulationConfig; return the model or None on error."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        print(f"\n[Validation] JSON parse error: {exc}")
        return None

    try:
        config = SimulationConfig.model_validate(data)
    except ValidationError as exc:
        print(f"\n[Validation] Schema validation errors:\n{exc}")
        return None

    # Serialise back to JSON (uses model serializer, enums → values)
    output_path.write_text(
        config.model_dump_json(indent=2, exclude_none=True), encoding="utf-8"
    )
    return config


# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------


def run_chatbot() -> None:
    llm = _build_llm()
    history: list[HumanMessage | AIMessage] = []

    # Build the conversational chain
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    chat_chain = chat_prompt | llm | StrOutputParser()

    # Extraction chain (single-shot, no memory needed)
    extract_chain = (
        ChatPromptTemplate.from_template(EXTRACT_PROMPT) | llm | StrOutputParser()
    )

    print("=" * 60)
    print("  SONATA Simulation Config Chatbot")
    print("  Type 'generate' at any time to produce the JSON file.")
    print("  Type 'quit' or 'exit' to quit.")
    print("=" * 60)
    print()

    # Kick off with a greeting from the assistant
    opening = chat_chain.invoke(
        {
            "history": [],
            "input": "Hello! I'd like to set up a SONATA simulation.",
        }
    )
    print(f"Assistant: {opening}\n")
    history.append(HumanMessage(content="Hello! I'd like to set up a SONATA simulation."))
    history.append(AIMessage(content=opening))

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

        # ----------------------------------------------------------------
        # Special command: generate the config file now
        # ----------------------------------------------------------------
        if user_input.lower() == "generate":
            print("\nAssistant: Generating your simulation config…\n")
            history_text = _history_as_text(history)
            raw = extract_chain.invoke({"history": history_text})

            output_path = Path("simulation_config.json")
            config = _validate_and_save(raw, output_path)

            if config:
                print(f"Config saved to: {output_path.resolve()}\n")
                print(output_path.read_text())
            else:
                print(
                    "The generated config failed validation (see errors above).\n"
                    "Let's keep refining — tell me what to fix or add more details.\n"
                )
            continue

        # ----------------------------------------------------------------
        # Normal conversational turn
        # ----------------------------------------------------------------
        reply = chat_chain.invoke({"history": history, "input": user_input})
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=reply))
        print(f"\nAssistant: {reply}\n")

        # If the model spontaneously produced a JSON block, try to save it
        json_block = _extract_json_from_reply(reply)
        if json_block:
            output_path = Path("simulation_config.json")
            config = _validate_and_save(json_block, output_path)
            if config:
                print(f"\n[Auto-saved] Config written to: {output_path.resolve()}\n")
            else:
                print(
                    "\n[Auto-save failed] The JSON above has validation errors.\n"
                    "Continue the conversation to fix them.\n"
                )


if __name__ == "__main__":
    run_chatbot()

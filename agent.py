"""
SONATA Config Agent — ReAct agent with tool use.

This is an alternative to chat_chain.py. Instead of a fixed conversation chain
with hardcoded fix logic, the LLM autonomously decides when to validate,
fix errors, and return the final config.

Usage:
    from agent import agent_chat_turn, agent_opening_message

Requires: langgraph
"""

from __future__ import annotations

import json
import logging
import re

import libsonata
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from pydantic import ValidationError

from config import GOOGLE_API_KEY, LLM_MODEL
from data_model import SimulationConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tools the agent can call
# ---------------------------------------------------------------------------


@tool
def retrieve_spec(query: str) -> str:
    """Search the SONATA spec documentation for relevant information.

    Use this when you need exact field names, valid enum values,
    or details about a specific section of the SONATA config.
    """
    logger.info("Tool called: retrieve_spec(%s)", query)
    # lazy import to initialze the embedding call when needed
    from rag import retrieve

    return retrieve(query)


@tool
def validate_config(config_json: str) -> str:
    """Validate a SONATA simulation config JSON string.

    Pass the raw JSON string. Returns 'Valid' if correct,
    or a detailed error message if validation fails.
    """
    logger.info("Tool called: validate_config")
    try:
        data = json.loads(config_json)
    except json.JSONDecodeError as exc:
        return f"JSON parse error: {exc}"

    try:
        SimulationConfig.model_validate(data)
    except ValidationError as exc:
        return f"Schema validation failed:\n{exc}"

    try:
        libsonata.SimulationConfig(config_json, "./")
    except Exception as exc:  # noqa: BLE001
        return f"libsonata validation failed: {exc}"

    return "Valid. The config passes all validation checks."


# ---------------------------------------------------------------------------
# Agent system prompt
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """\
You are an expert assistant for the SONATA neural circuit simulation framework
(BBP extension). Your job is to help users build a valid SONATA
simulation_config.json file through conversation.

Conversation strategy:
1. Ask for the three mandatory run parameters (tstop, dt, random_seed).
2. Ask which node_set to simulate. Explain that leaving it empty means all non-virtual nodes will be loaded.
3. Ask about conditions (temperature, v_init, spike_location).
4. Ask if they want inputs (stimuli).
5. Ask if they want reports.
6. Always ask about connection_overrides.
7. Once you have enough info, generate the JSON config.

Tool usage rules (IMPORTANT — minimize tool calls):
- ONLY call validate_config once, after generating the final complete JSON.
- Do NOT call validate_config for partial configs or during the conversation.
- Use retrieve_spec if you need to look up exact field names, enum values, or
  section details from the SONATA spec. Prefer this over guessing.
- ONLY call get_input_modules, get_report_types, or get_connection_override_fields
  if the user explicitly asks "what options are available" or seems confused.
- If validate_config fails, fix the error and validate ONE more time, then stop.

Rules:
- Ask ONE question at a time. Never combine multiple topics in a single response.
- Wait for the user to answer before moving to the next topic.
- If the user gives a wrong or invalid parameter, use retrieve_spec to look up the correct options
  and propose the right one.
- Use sensible defaults when the user is unsure (dt=0.025, celsius=34, v_init=-80).
- When the config is valid, present it in a ```json code block.
- Field names must be snake_case as defined in the SONATA spec.
- Enum values must be lowercase strings.
"""

# ---------------------------------------------------------------------------
# Agent setup (lazy init)
# ---------------------------------------------------------------------------

_agent = None


def _get_agent():
    global _agent  # noqa: PLW0603
    if _agent is None:
        if not GOOGLE_API_KEY:
            raise OSError("GOOGLE_API_KEY not set. Get a free key at https://aistudio.google.com/apikey")

        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0.2,
            google_api_key=GOOGLE_API_KEY,
        )

        # tools = [validate_config, retrieve_spec, get_input_modules, get_report_types, get_connection_override_fields]
        tools = [validate_config, retrieve_spec]
        _agent = create_react_agent(
            llm,
            tools,
            prompt=SystemMessage(content=AGENT_SYSTEM_PROMPT),
        )
    return _agent


# ---------------------------------------------------------------------------
# Public API (same interface as chat_chain.py)
# ---------------------------------------------------------------------------


def agent_chat_turn(
    history: list,
    user_message: str,
) -> tuple[str, SimulationConfig | None]:
    """
    Send one user message through the agent, return (reply, config_or_None).
    History is updated in-place.
    """
    agent = _get_agent()

    # Build messages for the agent
    messages = [*history, HumanMessage(content=user_message)]

    try:
        result = agent.invoke({"messages": messages})
    except Exception as exc:
        logger.exception("agent_chat_turn failed")
        reply = f"Sorry, I encountered an error: {exc}"
        history.append(HumanMessage(content=user_message))
        history.append(AIMessage(content=reply))
        return reply, None

    # Extract the final AI message (skip tool-call-only messages)
    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage) and not m.tool_calls]
    if ai_messages:
        content = ai_messages[-1].content
        # Gemini may return content as a list of blocks
        if isinstance(content, list):
            reply = "".join(
                block["text"] if isinstance(block, dict) else str(block)
                for block in content
                if (isinstance(block, dict) and block.get("type") == "text") or isinstance(block, str)
            )
        else:
            reply = content
    else:
        reply = "No response generated."

    history.append(HumanMessage(content=user_message))
    history.append(AIMessage(content=reply))

    # Check if reply contains a valid config
    config: SimulationConfig | None = None
    json_block = _extract_json_from_reply(reply)
    if json_block:
        try:
            data = json.loads(json_block)
            config = SimulationConfig.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            pass  # Agent should have validated already, but just in case

    return reply, config


def agent_opening_message() -> str:
    """Get the agent's opening greeting."""
    agent = _get_agent()
    try:
        result = agent.invoke({"messages": [HumanMessage(content="Hello! I'd like to set up a SONATA simulation.")]})
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        if ai_messages:
            content = ai_messages[-1].content
            if isinstance(content, list):
                return "".join(
                    block["text"] if isinstance(block, dict) else str(block)
                    for block in content
                    if (isinstance(block, dict) and block.get("type") == "text") or isinstance(block, str)
                )
            return content
        return "Hello! Let's build a SONATA config."
    except Exception as exc:
        logger.exception("agent_opening_message failed")
        return f"Hello! I'm ready to help you build a SONATA config. (Error: {exc})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_json_from_reply(text: str) -> str | None:
    """Return the first ```json ... ``` block from a reply, or None."""
    match = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL).search(text)
    return match.group(1).strip() if match else None


def agent_extract_config(
    history: list,
) -> tuple[SimulationConfig | None, str | None]:
    """
    Ask the agent to generate and validate a SONATA config from the conversation.
    Returns (config, None) on success, (None, error_message) on failure.
    """
    agent = _get_agent()
    messages = [
        *history,
        HumanMessage(
            content="Generate the complete SONATA simulation_config.json now. "
            "Use the validate_config tool to check it before returning.",
        ),
    ]

    try:
        result = agent.invoke({"messages": messages})
    except Exception as exc:
        logger.exception("agent_extract_config failed")
        return None, f"Agent failed: {exc}"

    # Find the final AI reply
    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage) and not m.tool_calls]
    if not ai_messages:
        return None, "Agent produced no response."

    content = ai_messages[-1].content
    if isinstance(content, list):
        reply = "".join(
            block["text"] if isinstance(block, dict) else str(block)
            for block in content
            if (isinstance(block, dict) and block.get("type") == "text") or isinstance(block, str)
        )
    else:
        reply = content

    # Extract and validate JSON from the reply
    json_block = _extract_json_from_reply(reply)
    if not json_block:
        logger.warning("agent_extract_config: no JSON block found in reply: %s", reply[:200])
        return None, "Agent did not produce a JSON config block."

    try:
        data = json.loads(json_block)
        config = SimulationConfig.model_validate(data)
        return config, None
    except (json.JSONDecodeError, ValidationError) as exc:
        return None, str(exc)

"""
SONATA Config Agent — ReAct agent with tool use.

This is an alternative to chatbot.py. Instead of a fixed conversation chain
with hardcoded fix logic, the LLM autonomously decides when to validate,
fix errors, and return the final config.

Usage:
    from agent import agent_chat_turn, agent_opening_message

Requires: langgraph
"""

from __future__ import annotations

import json
import logging
import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from pydantic import ValidationError

from data_model import SimulationConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tools the agent can call
# ---------------------------------------------------------------------------


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
        import libsonata
        libsonata.SimulationConfig(config_json, "./")
    except Exception as exc:  # noqa: BLE001
        return f"libsonata validation failed: {exc}"

    return "Valid. The config passes all validation checks."


@tool
def get_input_modules() -> str:
    """List all valid SONATA input modules with their input_type."""
    logger.info("Tool called: get_input_modules")
    return """Available input modules:
- linear (input_type: current_clamp) — continuous current injection
- relative_linear (input_type: current_clamp) — current relative to threshold
- pulse (input_type: current_clamp) — series of current pulses
- sinusoidal (input_type: current_clamp) — sinusoidal current
- subthreshold (input_type: current_clamp) — current adjusted below threshold
- hyperpolarizing (input_type: current_clamp) — hyperpolarizing holding current
- synapse_replay (input_type: spikes) — replay spikes from file
- seclamp (input_type: voltage_clamp) — voltage clamp
- noise (input_type: current_clamp) — current with random noise
- shot_noise (input_type: current_clamp or conductance) — Poisson shot noise
- relative_shot_noise (input_type: current_clamp or conductance) — relative shot noise
- absolute_shot_noise (input_type: current_clamp or conductance) — absolute shot noise
- ornstein_uhlenbeck (input_type: current_clamp or conductance) — OU process
- relative_ornstein_uhlenbeck (input_type: current_clamp or conductance) — relative OU
- spatially_uniform_e_field (input_type: extracellular_stimulation) — uniform E-field"""


@tool
def get_report_types() -> str:
    """List all valid SONATA report types and their key fields."""
    logger.info("Tool called: get_report_types")
    return """Available report types:
- compartment: each compartment reports separately (variable_name required)
- summation: sum values across compartments (variable_name required, can be comma-separated)
- synapse: each synapse reports separately (variable_name required)
- lfp: contribution to LFP signal (electrodes_file required, variable_name NOT allowed)
- compartment_set: report on specific compartment set (compartment_set field required)

Common fields: type, dt, start_time, end_time (all mandatory)
Optional: cells OR compartment_set (mutually exclusive, one required), sections, unit, file_name"""


@tool
def get_connection_override_fields() -> str:
    """List all valid fields for connection_overrides entries."""
    logger.info("Tool called: get_connection_override_fields")
    return """Connection override fields:
- name (mandatory): descriptive name
- source (mandatory): presynaptic node_set
- target (mandatory): postsynaptic node_set
- weight (optional): conductance multiplier
- spont_minis (optional): spontaneous mini rate
- synapse_configure (optional): HOC snippet, use %s for synapse reference
- modoverride (optional): synapse helper file prefix
- synapse_delay_override (optional): override synaptic delay in ms
- delay (optional): apply weight after this delay in ms
- neuromodulation_dtc (optional): neuromodulator decay time constant in ms
- neuromodulation_strength (optional): neuromodulator concentration increase in µM"""


# ---------------------------------------------------------------------------
# Agent system prompt
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """\
You are an expert assistant for the SONATA neural circuit simulation framework
(BBP extension). Your job is to help users build a valid SONATA
simulation_config.json file through conversation.

Conversation strategy:
1. Ask for the three mandatory run parameters (tstop, dt, random_seed).
2. Ask which node_set to simulate.
3. Ask about conditions (temperature, v_init, spike_location).
4. Ask if they want inputs (stimuli) — use get_input_modules tool if needed.
5. Ask if they want reports — use get_report_types tool if needed.
6. Always ask about connection_overrides.
7. Once you have enough info, generate the JSON config.

CRITICAL: Before returning a JSON config to the user, ALWAYS call the
validate_config tool to check it. If validation fails, fix the errors
and validate again until it passes. Never return invalid JSON to the user.

Rules:
- Ask one topic at a time.
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
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise OSError("GROQ_API_KEY not set. Get a free key at https://console.groq.com/")

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            api_key=api_key,
        )

        tools = [validate_config, get_input_modules, get_report_types, get_connection_override_fields]
        _agent = create_react_agent(
            llm,
            tools,
            prompt=SystemMessage(content=AGENT_SYSTEM_PROMPT),
        )
    return _agent


# ---------------------------------------------------------------------------
# Public API (same interface as chatbot.py)
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
    messages = list(history) + [HumanMessage(content=user_message)]

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
    reply = ai_messages[-1].content if ai_messages else "No response generated."

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
        result = agent.invoke(
            {"messages": [HumanMessage(content="Hello! I'd like to set up a SONATA simulation.")]}
        )
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        return ai_messages[-1].content if ai_messages else "Hello! Let's build a SONATA config."
    except Exception as exc:
        logger.exception("agent_opening_message failed")
        return f"Hello! I'm ready to help you build a SONATA config. (Error: {exc})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_json_from_reply(text: str) -> str | None:
    """Return the first ```json ... ``` block from a reply, or None."""
    import re
    match = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL).search(text)
    return match.group(1).strip() if match else None

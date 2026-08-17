# SONATA Config Bot

A conversational AI tool that generates valid [SONATA simulation configuration](https://sonata-extension.readthedocs.io/en/latest/sonata_simulation.html) (BBP extension) JSON files from natural language descriptions.

## What it does

You describe the simulation you want in plain English via a chat interface. The bot asks clarifying questions, then produces a fully validated `simulation_config.json` that conforms to the SONATA specification.

- Guides you through mandatory fields (tstop, dt, random_seed)
- Asks about node_set, conditions, stimuli, reports, and connection overrides
- Validates output against Pydantic models + libsonata
- Self-heals invalid JSON via an LLM fix chain
- Provides a downloadable JSON file
- Persists sessions in PostgreSQL

## Architecture

```
Browser (HTML/JS)  ──►  FastAPI (api.py)  ──►  LangGraph + Google Gemini (agent.py / chat_chain.py)
                                │                    │                │
                                ▼                    ▼                ▼
                        PostgreSQL (db.py)    RAG context        Pydantic + libsonata
                                             (rag.py + FAISS)   validation (data_model.py)
```

| File | Purpose |
|------|---------|
| `data_model.py` | Pydantic v2 models for the full SONATA simulation config schema |
| `chat_chain.py` | LangChain conversational chain + config extraction + self-healing fix loop |
| `agent.py` | LangGraph ReAct agent with tool calling (alternative to chat_chain.py) |
| `rag.py` | RAG module — embeds SONATA spec + schema into FAISS, retrieves relevant context per query |
| `api.py` | FastAPI server with session management and REST endpoints |
| `db.py` | SQLAlchemy async models for PostgreSQL session persistence |
| `static/index.html` | Browser UI — session sidebar, chat panel, JSON preview + download |

## Quick start

### Prerequisites

- Python 3.11+
- PostgreSQL running locally
- A free [Google AI API key](https://aistudio.google.com/apikey)

### Installation

```bash
pip install -e .
```

### Setup database

```bash
createdb sonata_config_bot
```

Tables are auto-created on first server start.

### Run

```bash
export GOOGLE_API_KEY=<your_key>
export DATABASE_URL=postgresql+asyncpg://localhost/sonata_config_bot
uvicorn api:app --reload
```

Open http://localhost:8000 in your browser.

### CLI mode

```bash
export GOOGLE_API_KEY=<your_key>
python chat_chain.py
```

### Using the chain (alternative)

In `api.py`, switch the imports:

```python
from chat_chain import chat_turn, extract_config, opening_message
```

The chain mode uses a simpler approach with a self-healing fix loop instead of tool calling.

## Features

### Agent mode (`agent.py`) — default
- LangGraph ReAct agent with tools: `validate_config`, `retrieve_spec`, `get_input_modules`, `get_report_types`, `get_connection_override_fields`
- LLM decides autonomously when to validate and how to fix errors
- Requires a model with reliable tool calling (Google Gemini recommended)

### Chain mode (`chat_chain.py`) — alternative
- Conversational chain with system prompt + JSON schema injection
- RAG context retrieval per turn (SONATA spec + Pydantic schema)
- Self-healing: if generated JSON fails validation, sends the error back to the LLM for automatic fix (up to 2 retries)

### RAG (`rag.py`)
- Embeds the SONATA spec + Pydantic JSON schema using Google Gemini Embeddings
- Stores vectors in FAISS with disk caching (`faiss_index/`)
- First run builds the index (takes ~30s due to rate limit batching)
- Subsequent runs load from disk instantly — zero API calls

### Session persistence (`db.py`)
- PostgreSQL via SQLAlchemy async
- Sessions and messages survive server restarts
- Sessions listed in the UI sidebar with labels and timestamps

## Spec coverage

The Pydantic models cover the full SONATA simulation config specification (v2.4):

- `run` — all mandatory and optional fields
- `output` — output directory, spikes file, sort order
- `conditions` — temperature, v_init, mechanisms, modifications
- `inputs` — all 15 stimulus modules (linear, pulse, noise, shot_noise, ornstein_uhlenbeck, etc.)
- `reports` — compartment, summation, synapse, LFP, compartment_set
- `connection_overrides` — synaptic weight and property adjustments

## TODO

- [ ] Improve agent reliability (tool-calling format issues with some models)
- [ ] Add user authentication for multi-user deployment
- [ ] Add session expiry / cleanup
- [ ] Improve CLI Mode

## License

TBD

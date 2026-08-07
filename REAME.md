# SONATA Config Bot

A conversational chatbot that generates valid [SONATA simulation configuration](https://sonata-extension.readthedocs.io/en/latest/sonata_simulation.html) JSON files from natural language descriptions.

## What it does

You describe the simulation you want in plain English via a chat interface. The bot asks clarifying questions, then produces a fully validated `simulation_config.json` that conforms to the SONATA specification.

- Guides you through mandatory fields (tstop, dt, random_seed)
- Asks about conditions, stimuli, reports, and connection overrides
- Validates the output against Pydantic models derived from the spec
- Provides a downloadable JSON file

## Architecture

```
Browser (HTML/JS)  ──►  FastAPI (api.py)  ──►  LangChain + Groq (chatbot.py)
                                                       │
                                                       ▼
                                              Pydantic validation (data_model.py)
```

| File | Purpose |
|------|---------|
| `data_model.py` | Pydantic v2 models for the full SONATA simulation config schema |
| `chatbot.py` | LangChain conversational chain + config extraction logic |
| `api.py` | FastAPI server with session management and REST endpoints |
| `static/index.html` | Browser UI — chat panel, session sidebar, JSON preview |

## Quick start

### Prerequisites

- Python 3.11+
- A free [Groq API key](https://console.groq.com/)

### Installation

```bash
pip install -e .
```

### Run

```bash
export GROQ_API_KEY=<your_key>
uvicorn api:app --reload
```

Open http://localhost:8000 in your browser.

### CLI mode

```bash
export GROQ_API_KEY=<your_key>
python chatbot.py
```

## Spec coverage

The Pydantic models cover the full SONATA simulation config specification (v2.4):

- `run` — all mandatory and optional fields
- `output` — output directory, spikes file, sort order
- `conditions` — temperature, v_init, mechanisms, modifications
- `inputs` — all 15 stimulus modules (linear, pulse, noise, shot_noise, ornstein_uhlenbeck, etc.)
- `reports` — compartment, summation, synapse, LFP, compartment_set
- `connection_overrides` — synaptic weight and property adjustments

## License

TBD

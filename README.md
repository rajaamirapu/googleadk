# 🌤️ Weather & Sunrise/Sunset Agent

A **Google ADK** agent that provides real-time weather conditions, sunrise, and sunset times for any location — using a **custom LangChain LLM** as the backend.

---

## 🏗️ Project Structure

```
googleadk/                          ← AGENTS_DIR (run adk web from here)
├── weather_sunrise_agent/          ← Agent package
│   ├── __init__.py                 # Exports root_agent (required by ADK)
│   ├── agent.py                    # Agent + custom LLM wiring
│   └── tools.py                    # Tool functions (weather, sunrise/sunset)
├── custom_llm/                     ← LangChain → ADK bridge
│   ├── __init__.py
│   ├── adk_langchain_bridge.py     # Core bridge (LangChain LLM → ADK BaseLlm)
│   └── base_custom_llm.py         # Example LLM implementations
├── launch.sh                       # ✅ Correct way to start the web UI
├── demo.py                         # Programmatic demo (no web UI)
├── debug_connection.py             # Diagnose LLM connection issues
├── requirements.txt
├── .env                            # Auto-loaded by ADK at startup
├── .env.example
└── README.md
```

---

## ⚠️ Critical: How to Run `adk web` Correctly

**`adk web` takes a directory that CONTAINS agent packages — not the agent name itself.**

```bash
# ✅ CORRECT — run from inside googleadk/ (or pass it as the path)
cd path/to/googleadk
adk web .

# ✅ Also correct — use the launch script
./launch.sh

# ✅ Also correct — full path
adk web path/to/googleadk

# ❌ WRONG — this makes ADK look inside weather_sunrise_agent/ for more agents
adk web weather_sunrise_agent
```

---

## ⚙️ Prerequisites

| Requirement | Details |
|---|---|
| Python | 3.10+ |
| Custom LLM server | Any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio…) |
| LLM model | Must be running and accessible |

---

## 🚀 Quick Start

### 1. Start your LLM server

```bash
# Example with Ollama:
ollama serve
ollama pull llama3.2
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure `.env`

Edit `.env` (already created with defaults) — set your LLM endpoint:

```env
CUSTOM_LLM_BASE_URL=http://localhost:11434/v1
CUSTOM_LLM_MODEL=llama3.2
CUSTOM_LLM_API_KEY=custom
```

### 4a. Run with the ADK Web UI (recommended)

```bash
# From inside the googleadk/ directory:
cd path/to/googleadk
adk web .
# Open http://localhost:8000 in your browser

# Or use the included script (handles the cd automatically):
./launch.sh
```

### 4b. Run with the ADK CLI

```bash
cd path/to/googleadk
adk run weather_sunrise_agent
```

### 4c. Run the programmatic demo

```bash
python demo.py
```

---

## 🛠️ Tools Available

| Tool | Description | APIs Used |
|---|---|---|
| `get_weather` | Current conditions (temp, humidity, wind, etc.) | [Open-Meteo](https://open-meteo.com/) *(free, no key)* |
| `get_sunrise_sunset` | Sunrise, sunset, and twilight times | [Sunrise-Sunset.org](https://sunrise-sunset.org/api) *(free, no key)* |
| `get_full_report` | Combined weather + sun times in one call | Both of the above |

---

## 💬 Example Prompts

```
Give me a full report for latitude 40.7128, longitude -74.0060
What's the weather at lat 48.8566, lon 2.3522?
When does the sun rise and set in Tokyo? (lat 35.6895, lon 139.6917)
Show sunrise and sunset for 51.5074, -0.1278
```

---

## 🔧 Changing the Ollama Model

Edit `.env` or set the environment variable before running:

```bash
OLLAMA_MODEL=gemma3 adk web weather_sunrise_agent
```

Or change the default in `weather_sunrise_agent/agent.py`:
```python
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3")
```

> **Tip:** Models with strong tool-calling support work best.  
> Recommended: `llama3.2`, `llama3.1`, `gemma3`, `mistral`

---

## 🌐 APIs Used (No Keys Required)

- **Open-Meteo** — `https://api.open-meteo.com/v1/forecast`  
  Free, open-source weather API. No registration needed.

- **Sunrise-Sunset.org** — `https://api.sunrise-sunset.org/json`  
  Free API for astronomical sun data. No registration needed.

---

## 🐛 Troubleshooting

| Problem | Fix |
|---|---|
| `Connection refused` on port 11434 | Run `ollama serve` in a terminal |
| Model not found | Run `ollama pull <model_name>` |
| Infinite tool-call loop | Make sure model prefix is `ollama_chat/` not `ollama/` |
| Slow responses | Use a smaller model (`llama3.2` vs `llama3.1:70b`) |

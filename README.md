# 🌤️ Weather & Sunrise/Sunset Agent

A **Google ADK** agent that provides real-time weather conditions, sunrise, and sunset times for any location — using **Ollama** as the local LLM backend (no OpenAI or Google API keys needed).

---

## 🏗️ Project Structure

```
googleadk/
├── weather_sunrise_agent/
│   ├── __init__.py       # Exports root_agent (required by ADK)
│   ├── agent.py          # Agent definition + Ollama LLM config
│   └── tools.py          # Tool functions (weather, sunrise/sunset)
├── demo.py               # Programmatic demo (no web UI)
├── requirements.txt
├── .env.example          # Copy to .env and configure
└── README.md
```

---

## ⚙️ Prerequisites

| Requirement | Details |
|---|---|
| Python | 3.10+ |
| [Ollama](https://ollama.com/) | Running locally on port 11434 |
| Ollama model | e.g. `llama3.2`, `gemma3`, `mistral` |

---

## 🚀 Quick Start

### 1. Install Ollama & pull a model

```bash
# Install Ollama from https://ollama.com/download
# Then pull a model (pick one that fits your hardware):
ollama pull llama3.2       # ~2GB — fast, good for tool calling
ollama pull gemma3         # Google's open model
ollama pull mistral        # Mistral 7B
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env to set OLLAMA_MODEL to your pulled model name
```

### 4a. Run with the ADK Web UI (recommended)

```bash
adk web weather_sunrise_agent
# Open http://localhost:8000 in your browser
```

### 4b. Run with the ADK CLI

```bash
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

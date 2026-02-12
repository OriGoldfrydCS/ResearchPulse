<p align="center">
  <img src="static/public/logo.png" alt="ResearchPulse Logo" width="200" />
</p>

<h1 align="center">ResearchPulse</h1>

<p align="center">
  <strong>Your Autonomous AI Research Assistant</strong><br/>
  Perceive · Reason · Act - so you never miss a breakthrough paper again.
</p>

<p align="center">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" />
  <img alt="LangChain" src="https://img.shields.io/badge/LangChain-ReAct-1C3C3C?logo=chainlink&logoColor=white" />
  <img alt="Pinecone" src="https://img.shields.io/badge/Pinecone-RAG-000?logo=pinecone&logoColor=white" />
  <img alt="License MIT" src="https://img.shields.io/badge/license-MIT-green" />
</p>

---

## 🧠 What is ResearchPulse?

ResearchPulse is an **autonomous AI agent** that helps researchers stay on top of the scientific literature. It continuously scans arXiv, evaluates papers against your personal research profile, and takes intelligent actions - from email digests and calendar reminders to colleague-level paper sharing - all without manual intervention.

Built on a **ReAct (Reasoning + Acting)** agent powered by LangChain, with Pinecone vector search for RAG-based novelty detection, and served through a modern FastAPI + web dashboard.

> [!IMPORTANT]
> ResearchPulse is a **fully autonomous agent**, not a chatbot. It perceives the research landscape, reasons about what matters to *you*, and acts on your behalf - while keeping you in full control via configurable policies and execution settings.

---

## 🏗️ Architecture

<p align="center">
  <img src="static/public/architecture.png" alt="ResearchPulse Architecture" width="800" />
</p>

ResearchPulse operates through a three-phase cognitive loop inspired by autonomous agent design:

---

### 👁️ Perception - *"What's new in the world?"*

The agent observes the research landscape by pulling fresh data from external sources:

| Component | What it does |
|-----------|-------------|
| **arXiv API** | Fetches recent papers filtered by your chosen categories and time period |
| **Pinecone RAG** | Queries the vector store to detect novelty - has this topic been seen before? |
| **Inbox Monitor** | Checks email for colleague replies and feedback on shared papers |
| **Profile Loader** | Reads your research interests, exclusions, and delivery preferences |

> [!NOTE]
> In the current version, perception is **focused** — the agent fetches papers matching your configured arXiv categories and time window, while also drawing context from your user profile, colleague interests, and past feedback stored in the database. This keeps discovery targeted without information overload.

---

### 🧩 Reasoning - *"What matters to the researcher?"*

The LLM-powered ReAct core evaluates every paper through structured thinking:

| Step | Description |
|------|-------------|
| **Relevance Scoring** | Compares each paper's abstract against your research profile using the LLM |
| **Novelty Detection** | Embeds the paper and queries Pinecone - if too similar to past papers, it's deprioritized |
| **Importance Ranking** | Assigns `high` / `medium` / `low` importance based on combined relevance + novelty |
| **Delivery Decision** | Applies your delivery policy to decide: notify, share with a colleague, or just log it |
| **Stop Policy** | Continuously checks guardrails (max runtime, max papers, max RAG queries) to stay bounded |

The reasoning phase follows the **ReAct pattern**: `Thought → Action → Observation → Thought → ...`, with every step logged for full transparency.

> [!TIP]
> Open the **Live Document** on the Home tab after a run to see the full chain of thoughts and actions the agent took - great for understanding *why* a paper was flagged as important.

---

### ⚡ Action - *"Do something useful."*

Once reasoning is complete, the agent executes real-world actions:

| Action | Trigger | Output |
|--------|---------|--------|
| 📧 **Email Digest** | High-importance paper found | HTML email sent to your inbox |
| 📅 **Calendar Reminder** | Paper worth reading soon | `.ics` file for Google Calendar / Outlook |
| 📤 **Colleague Share** | Paper matches a colleague's interests | Targeted email with paper summary |
| ⭐ **Paper Tagging** | Relevance/importance scored | Paper saved with metadata to your library |
| 📝 **AI Summary** | On-demand via dashboard | LLM-generated summary of the full PDF |
| 💡 **Profile Evolution** | Patterns detected in your feedback | Suggestions to refine your research interests |

> [!NOTE]
> All actions are **auditable**. Every email sent, calendar event created, and share made is logged in the database and visible in the dashboard's Emails, Alerts, and Shares tabs.

---

## 🔀 Autonomous Decision Graph

Unlike a simple linear pipeline, ResearchPulse is a **decision graph** - the agent reaches **20+ autonomous junctions** where it chooses different paths based on context, scores, policies, and feature flags:

<div style="overflow:scroll; max-height:600px; max-width:100%; border:2px solid #d1d5db; border-radius:12px;">
  <img src="static/public/decision_graph.svg" alt="ResearchPulse Autonomous Decision Graph" />
</div>

<p align="center">
  <em>↕️ ↔️ Scroll inside the box to navigate · <a href="static/public/decision_graph.svg">Open full-size SVG</a></em>
</p>

#### 🗺️ Legend

<table>
  <tr>
    <th>Shape / Color</th>
    <th>Meaning</th>
    <th>Example</th>
  </tr>
  <tr>
    <td><img src="https://img.shields.io/badge/◆-Decision-db2777?style=flat-square" /></td>
    <td><strong>Diamond - Autonomous Decision</strong><br/>Agent evaluates a condition and chooses a path. No human in the loop.</td>
    <td>Scope Gate, Stop Policy, Importance, Digest Mode, Auto-Send</td>
  </tr>
  <tr>
    <td><img src="https://img.shields.io/badge/■-Action-3b82f6?style=flat-square" /></td>
    <td><strong>Rectangle - Action / Processing</strong><br/>Agent performs a concrete task: fetch, score, send, persist.</td>
    <td>Fetch Papers, Score Relevance, Send Email, Share Paper</td>
  </tr>
  <tr>
    <td><img src="https://img.shields.io/badge/⬭-Terminal-16a34a?style=flat-square" /></td>
    <td><strong>Rounded - Start / End / Terminate</strong><br/>Entry and exit points of the agent episode.</td>
    <td>Agent Episode Starts, Episode Complete, Terminate</td>
  </tr>
  <tr>
    <td><img src="https://img.shields.io/badge/●-HIGH-dc2626?style=flat-square" /></td>
    <td><strong>Red - HIGH Importance</strong><br/>Paper with relevance ≥ 0.65 and novelty ≥ 0.5. Triggers email + calendar + reading list.</td>
    <td>HIGH Importance path</td>
  </tr>
  <tr>
    <td><img src="https://img.shields.io/badge/●-MEDIUM-d97706?style=flat-square" /></td>
    <td><strong>Amber - MEDIUM Importance</strong><br/>Paper with relevance ≥ 0.4 (or ≥ 0.3 + novelty ≥ 0.6). Added to reading list.</td>
    <td>MEDIUM Importance path</td>
  </tr>
  <tr>
    <td><img src="https://img.shields.io/badge/●-LOW-16a34a?style=flat-square" /></td>
    <td><strong>Green - LOW Importance</strong><br/>Paper below thresholds. Logged only, no actions triggered.</td>
    <td>LOW Importance path</td>
  </tr>
</table>

> [!IMPORTANT]
> Every **diamond** node is an autonomous decision the agent makes on its own - no human in the loop. The agent evaluates each paper independently and chooses a unique combination of actions based on the paper's scores, your delivery policy, and your colleagues' interests. Two papers in the same run can follow completely different paths.

> [!TIP]
> This is **not a chain** - it's a graph with 20+ independent decision junctions per paper. The agent can simultaneously send an email digest, share with a colleague, create a calendar event, *and* suggest a profile update - or do none of those - all based on autonomous reasoning. After each run, four feature-flagged autonomous components (Audit Log, LLM Novelty, Profile Evolution, Live Document) each make their own independent decisions.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🤖 **Autonomous Agent** | ReAct loop with bounded execution - no infinite polling |
| 🔍 **Smart Search** | "Search for me" generates queries from your profile automatically |
| 📊 **Relevance + Novelty** | Dual scoring via LLM + Pinecone vector similarity |
| 👥 **Colleague Sharing** | Auto-match papers to colleagues by research interests |
| 📄 **Paper Summaries** | One-click AI summarization of any paper's PDF |
| 📬 **Inbox Monitoring** | Detects and processes colleague replies |
| 🧬 **Profile Evolution** | Learns from your stars and feedback to improve over time |
| 📥 **CSV Export** | Export your paper library for reference managers |
| 🌓 **Dark / Light Mode** | Theme toggle with persistent preference |
| 🔐 **Join Code Security** | Colleagues need a passphrase to join your network |
| 📈 **Execution Controls** | Max runtime, max papers, min importance - all configurable |
| 🩺 **Health Dashboard** | Real-time status of database, Pinecone, and email connections |

---

## 🖥️ Dashboard Preview

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="static/public/dashboard_dark.png" />
    <source media="(prefers-color-scheme: light)" srcset="static/public/dashboard_light.png" />
    <img src="static/public/dashboard_dark.png" alt="ResearchPulse Dashboard" width="900" />
  </picture>
</p>

<p align="center">
  <sub>🌙 Dark mode (default) &nbsp;·&nbsp; ☀️ Light mode available via toggle</sub>
</p>

> [!TIP]
> The dashboard is a full single-page application with 8 tabs - Home, Papers, Emails, Alerts, Shares, Colleagues, Settings, and Q&A. Theme preference is persisted across sessions.

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <repository-url>
cd ResearchPulse
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
copy .env.template .env       # Windows
# cp .env.template .env       # Linux / macOS
```

> [!IMPORTANT]
> You **must** fill in all required API keys in `.env` before starting. The app validates every variable at boot and will refuse to start with a clear error message if anything is missing.

### 3. Initialize Database

```bash
python main.py db-init
```

### 4. Launch

```bash
python main.py
```

Open **http://127.0.0.1:8000** - you'll land on the Home tab. Set up your Research Profile in **My Settings**, then hit **"Search for me"**.

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|:--------:|-------------|
| `DATABASE_URL` | ✅ | PostgreSQL (Supabase) connection string |
| `LLM_API_KEY` | ✅ | API key for the LLM service |
| `LLM_API_BASE` | ✅ | Base URL for OpenAI-compatible API |
| `LLM_MODEL_NAME` | ✅ | Model name (e.g. `gpt-4o`) |
| `LLM_PROVIDER` | ✅ | LLM provider identifier (default: `openai`) |
| `PINECONE_API_KEY` | ✅ | Pinecone API key |
| `PINECONE_INDEX_NAME` | ✅ | Pinecone index name |
| `PINECONE_ENVIRONMENT` | ✅ | Pinecone environment / region |
| `EMBEDDING_API_KEY` | ✅ | Embeddings API key |
| `EMBEDDING_API_BASE` | ✅ | Embeddings base URL |
| `EMBEDDING_API_MODEL` | ✅ | Embedding model name |
| `PINECONE_NAMESPACE` | - | Namespace (default: `demo`) |
| `EMBEDDING_API_DIMENSION` | - | Vector dimension (default: `1536`) |
| `APP_HOST` | - | Server host (default: `127.0.0.1`) |
| `APP_PORT` | - | Server port (default: `8000`) |
| `ARXIV_MAX_RESULTS` | - | Max papers per query (default: `50`) |

> [!TIP]
> Keep your `.env` file **out of version control**. A `.env.template` is provided with placeholder values for every variable.

---

## 🛡️ Stop Policy & Guardrails

Every run is **bounded** - the agent stops when *any* condition is met:

| Guardrail | Default | Purpose |
|-----------|:-------:|---------|
| Max runtime | 6 min | Prevents runaway execution |
| Max papers checked | 30 | Limits evaluation scope |
| Stop if no new papers | `true` | Exits early when nothing is unseen |
| Max RAG queries | 50 | Caps vector store calls |
| Min importance to act | `medium` | Only important papers trigger actions |

> [!TIP]
> All guardrails are configurable in **My Settings → Execution Settings** on the dashboard.

---

## 📦 Storage Layer

```
┌─────────────────────────────────┐   ┌──────────────────────────────┐
│    PostgreSQL (Supabase)        │   │    Pinecone (Vector Store)   │
│                                 │   │                              │
│  users · papers · paper_views   │   │  Paper embeddings            │
│  colleagues · runs · actions    │   │  Semantic similarity search  │
│  emails · calendar_events       │   │  Novelty detection           │
│  shares · delivery_policies     │   │                              │
└─────────────────────────────────┘   └──────────────────────────────┘
```

> [!NOTE]
> All state lives in PostgreSQL + Pinecone - the app is **deployment-safe** and works identically on local dev, Render, or any cloud host.

---

## 🖥️ Dashboard Tabs

| Tab | Icon | What you'll find |
|-----|:----:|-----------------|
| **Home** | 🏠 | Chat input, "Search for me", Live Document, Profile Suggestions |
| **Papers** | 📄 | Full paper library with star, filter, sort, bulk actions, CSV export |
| **Emails** | 📧 | All sent email digests and colleague notifications |
| **Alerts** | 📅 | Calendar events and reading reminders (.ics download) |
| **Shares** | 📤 | Papers shared with colleagues and delivery status |
| **Colleagues** | 👥 | Manage collaborators, their interests, and join codes |
| **My Settings** | ⚙️ | Research profile, execution settings, inbox config, health checks |
| **Q&A** | ❓ | FAQ and help for every feature |

---

## ☁️ Deployment (Render)

### Build Command

```bash
pip install -r requirements.txt && python main.py db-init
```

### Start Command

```bash
python main.py server
```

> [!IMPORTANT]
> Set **all** required environment variables in your Render dashboard before deploying. The app will exit on boot with a clear error if any are missing.

### Render Environment Extras

| Variable | Value |
|----------|-------|
| `ENV` | `production` |
| `APP_HOST` | `0.0.0.0` |

---

## 🧪 Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Unit tests (fast, no external services)
pytest unit_testing/unit/

# All tests including integration
pytest unit_testing/

# With coverage report
pytest unit_testing/ --cov=src --cov-report=html
```

### Code Formatting

```bash
black src/ unit_testing/
isort src/ unit_testing/
```

---

## 📊 Project Structure

```
ResearchPulse/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Build config & metadata
├── alembic.ini             # Database migration config
├── migrations/             # Alembic migration scripts
│   └── versions/           # Individual migrations
├── static/
│   ├── index.html          # Full SPA dashboard
│   └── public/             # Logo, architecture diagram
└── src/
    ├── agent/              # ReAct agent, stop controller, profile evolution
    ├── api/                # FastAPI routes, run manager, colleague routes
    ├── config/             # Feature flags
    ├── db/                 # ORM models, database session, data service
    ├── rag/                # Pinecone client, embeddings, retriever
    └── tools/              # 20+ LangChain tools (fetch, score, email, etc.)
```

---

## 📜 License

MIT - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

---

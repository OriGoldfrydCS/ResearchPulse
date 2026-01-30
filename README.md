# ResearchPulse

**Research Awareness and Sharing Agent** — A ReAct agent powered by LangChain, Pinecone RAG, and FastAPI.

ResearchPulse helps researchers stay up-to-date with relevant scientific papers from arXiv. It autonomously filters information overload, ranks papers by relevance and novelty, and optionally shares discoveries with colleagues.

---

## Features

- **Episodic Execution**: Each run is triggered by the web UI, starts from a user prompt, and terminates explicitly via a configurable stop policy. No continuous polling.
- **ReAct Agent Pattern**: Alternates between structured reasoning (Thought) and tool actions (Action → Observation), with all steps logged for transparency.
- **arXiv Integration**: Retrieves recent papers via arXiv API/RSS, filtered by category include/exclude rules.
- **Pinecone RAG**: Semantic retrieval to detect novelty, compare against researcher's own papers, and avoid redundant recommendations.
- **Autonomous Decisions**: Determines relevance, importance (high/medium/low), and delivery actions (notify, share, log only).
- **Simulated Actions**: Generates email summaries, calendar entries (.ics), and reading list updates as file artifacts.
- **Web Chat UI**: Simple chat interface to trigger runs and view reports/artifacts.

---

## Project Structure

```
ResearchPulse/
├── .env.template          # Environment variables template
├── main.py                # Application entry point
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project metadata and build config
├── README.md              # This file
├── LICENSE                # MIT License
├── CODE_OF_CONDUCT.md     # Contributor code of conduct
├── data/                  # Demo JSON databases
│   ├── research_profile.json
│   ├── papers_state.json
│   ├── colleagues.json
│   ├── delivery_policy.json
│   └── arxiv_categories.json
├── static/                # Frontend assets
│   ├── index.html         # Chat UI
│   └── public/            # Static public files
└── src/                   # Backend source code
    ├── agent/             # ReAct agent and stop controller
    ├── tools/             # LangChain tools for the agent
    ├── rag/               # Pinecone and embeddings integration
    ├── arxiv/             # arXiv API client and parser
    ├── db/                # JSON database utilities
    ├── api/               # FastAPI routes and run manager
    └── ui/                # Static file serving
```

---

## Quick Start

### 1. Clone and Install

```bash
git clone <repository-url>
cd ResearchPulse
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the template and fill in your API keys:

```bash
copy .env.template .env  # Windows
# cp .env.template .env  # Linux/macOS
```

Edit `.env` with your credentials (see [Environment Variables](#environment-variables) below).

### 3. Run the Application

```bash
python main.py
```

Open your browser to `http://127.0.0.1:8000` to access the chat UI.

---

## Environment Variables

The application reads configuration from a `.env` file. All required variables must be set before starting.

| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_PROVIDER` | Yes | LLM provider identifier (default: `openai`) |
| `LLM_API_BASE` | Yes | Base URL for OpenAI-compatible LLM API |
| `LLM_API_KEY` | Yes | API key for the LLM service |
| `LLM_MODEL_NAME` | Yes | Model name to use for the ReAct agent |
| `PINECONE_API_KEY` | Yes | Pinecone API key |
| `PINECONE_INDEX_NAME` | Yes | Name of the Pinecone index |
| `PINECONE_ENVIRONMENT` | Yes | Pinecone environment/region |
| `PINECONE_NAMESPACE` | No | Namespace within the index (default: `demo`) |
| `EMBEDDING_API_BASE` | Yes | Base URL for embeddings API |
| `EMBEDDING_API_KEY` | Yes | API key for embeddings service |
| `EMBEDDING_API_MODEL` | Yes | Embedding model name |
| `EMBEDDING_API_DIMENSION` | No | Embedding vector dimension (default: `1536`) |
| `APP_HOST` | No | Server host (default: `127.0.0.1`) |
| `APP_PORT` | No | Server port (default: `8000`) |
| `ARXIV_MAX_RESULTS` | No | Max papers to fetch per arXiv query (default: `50`) |

### Environment Validation and Graceful Failure

The application validates environment variables at startup with the following behavior:

1. **Startup Validation**: On application boot, all required environment variables are checked before any services initialize.

2. **Missing Required Variables**: If any required variable is missing or empty, the application will:
   - Log a clear error message listing all missing variables
   - Print a human-readable instruction to the console
   - Exit with a non-zero status code (will not start the server)
   - Example: `ERROR: Missing required environment variables: LLM_API_KEY, PINECONE_API_KEY. Please check your .env file.`

3. **Invalid Values**: If a variable has an invalid format (e.g., non-numeric port), the application will:
   - Log the specific validation error
   - Suggest the expected format
   - Exit gracefully without crashing

4. **Optional Variables**: Missing optional variables will use sensible defaults and log an INFO message noting the default value being used.

5. **Runtime API Failures**: If external APIs (Pinecone, LLM) fail during a run:
   - The error is caught and logged
   - The current run terminates with a clear error in the run report
   - The server remains operational for subsequent runs
   - No unhandled exceptions propagate to the user

6. **Partial Configuration**: The application will not attempt partial initialization. All required services must be configurable, or the app refuses to start.

---

## API Endpoints

### `POST /chat`

Start a new episodic agent run.

**Request Body:**
```json
{
  "message": "Find recent papers on transformer architectures for NLP"
}
```

**Response:**
```json
{
  "run_id": "uuid-string",
  "status": "started",
  "message": "Agent run initiated"
}
```

### `GET /status?run_id={run_id}`

Poll for run status and updates.

**Response:**
```json
{
  "run_id": "uuid-string",
  "status": "running | completed | error",
  "steps": [...],
  "report": {...},
  "artifacts": [...]
}
```

---

## Stop Policy

The agent enforces an explicit stop policy to ensure bounded execution. The run terminates when ANY condition is met:

| Condition | Default | Description |
|-----------|---------|-------------|
| `max_runtime_minutes` | 6 | Maximum wall-clock time for a run |
| `max_papers_checked` | 30 | Maximum papers to evaluate |
| `stop_if_no_new_papers` | true | Stop immediately if no unseen papers found |
| `max_rag_queries` | 50 | Maximum RAG retrieval calls |
| `min_importance_to_act` | medium | Minimum importance level to trigger actions |

The stop policy can be customized per researcher in `data/research_profile.json`.

---

## ReAct Tools

The agent uses these LangChain tools:

| Tool | Description |
|------|-------------|
| `fetch_arxiv_papers` | Query arXiv for recent papers by category |
| `check_seen_papers` | Compare papers against the Papers State DB |
| `retrieve_similar_from_pinecone` | RAG query for novelty/similarity detection |
| `score_relevance_and_importance` | Evaluate paper relevance to researcher profile |
| `decide_delivery_action` | Determine action based on delivery policy |
| `persist_state` | Save decisions to local JSON databases |
| `generate_report` | Create the final run report |
| `terminate_run` | Explicitly end the agent run |

---

## Demo Databases

The `data/` folder contains JSON files with demo data:

- **research_profile.json**: Researcher preferences, topics, and stop policy
- **papers_state.json**: Previously seen papers and decisions
- **colleagues.json**: Colleagues for paper sharing
- **delivery_policy.json**: Rules for notifications and sharing
- **arxiv_categories.json**: Supported arXiv category codes

---

## Simulated Actions

For demo purposes, actions produce file artifacts instead of actual emails/calendar events:

- **Email Summary**: Written to `outputs/emails/email_{timestamp}.txt`
- **Calendar Entry**: Written to `outputs/calendar/event_{timestamp}.ics`
- **Reading List**: Appended to `outputs/reading_list.txt`
- **Colleague Shares**: Written to `outputs/shares/share_{colleague}_{timestamp}.txt`

---

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ tests/
isort src/ tests/
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contributing

Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

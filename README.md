# ParcelPilot Internal Support Agent

An AI-powered internal support operations assistant for ParcelPilot's customer operations team. Built as part of the AI Agent Assessment.

## Quick Start

### Prerequisites
- Python 3.11+
- Google Gemini API key ([get one free](https://aistudio.google.com))

### Setup
```bash
# Clone the repository
git clone <repo-url>
cd parcelpilot-support-agent

# Install dependencies
pip install -r requirements.txt

# Configure API key
echo "GOOGLE_API_KEY=your_key_here" > .env

# Run the application
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

### Hosted Application
> **[Live Demo →]([https://parcelpilot-support-agent.streamlit.app](https://parcelpilot-supporting-agent.streamlit.app/#parcel-pilot-support-agent))**

---

## Architecture Note

### Agent Design

The system uses a **LangGraph ReAct agent** powered by Google Gemini. The agent follows a reason-then-act loop: it receives a natural language query, decides which tools to call, interprets results, and either calls more tools or produces a final answer.

```
User Query → LLM (Gemini) → Tool Selection → Tool Execution → LLM Reasoning → Response
                  ↑                                                    |
                  └────────────── (loop if more tools needed) ─────────┘
```

**Key architectural choices:**
- **Single agent with multiple tools** rather than separate agents per domain — simpler to reason about, easier to debug, and supports the multi-step requirement naturally.
- **Stateless per-request** — each query creates a fresh tools instance scoped to the authenticated user's role. No shared mutable state between requests.
- **Conversation memory** via LangChain message history — allows follow-up questions.

### Tool Design

| Tool | Type | Purpose |
|------|------|---------|
| `search_documents` | Retrieval | RAG over policies, SOPs, agreements, product docs |
| `lookup_order` | Data query | Fetch order details by ID |
| `lookup_account` | Data query | Fetch account details by ID |
| `lookup_ticket` | Data query | Fetch ticket details by ID |
| `query_orders_by_account` | Data query | List all orders for an account |
| `query_tickets_by_account` | Data query | List all tickets for an account |
| `calculate_cancellation_eligibility` | Calculation | Determine if an order can be cancelled and fee |
| `calculate_service_credit` | Calculation | Determine credit eligibility and amount |
| `escalate_ticket` | State-change | Prepare ticket escalation (requires confirmation) |
| `cancel_order` | State-change | Prepare order cancellation (requires confirmation) |
| `issue_service_credit` | State-change | Prepare credit issuance (requires confirmation) |
| `confirm_action` | State-change | Execute a previously prepared action |

**Design principle:** Calculation tools embed business logic (agreement overrides, SOP rules, timing calculations) rather than relying on the LLM to compute. This ensures deterministic, auditable results.

**Confirmation pattern:** All state-changing tools return a "PENDING_CONFIRMATION" payload. The agent is instructed to present this to the user and only call `confirm_action` after explicit approval.

### Document and Structured-Data Handling

**Documents (PDFs):**
- Loaded via PyPDF, chunked with `RecursiveCharacterTextSplitter` (800 chars, 100 overlap)
- Each chunk carries metadata: `doc_id`, `title`, `status`, `authority`, `account_id`
- Indexed in a FAISS vector store using Google's `gemini-embedding-001` model
- Search results are sorted by **source authority** before relevance score

**Structured data (Excel):**
- Loaded into pandas DataFrames (accounts, orders, tickets)
- Queried via tool functions with proper filtering
- No raw DataFrame access from the LLM — all access is through scoped tool functions

### Source Reliability and Conflict Handling

The system enforces a strict **source precedence hierarchy**, embedded at three levels:

1. **Document metadata** — each document has an `authority` score (1=agreement, 2=policy, 3=product docs, 99=deprecated)
2. **Search results ordering** — results sorted by authority before being sent to the LLM
3. **System prompt** — explicit instructions on precedence, with emphasis on NEVER using deprecated sources

**Conflict resolution rules:**
- Customer agreements override default policies for that customer
- Current policies override product documentation
- Historical ticket resolutions are flagged as "context only, may be incorrect"
- Deprecated documents (Support Policy v2) are explicitly labeled and the LLM is instructed never to cite them

**Example:** Northstar's agreement waives cancellation fees. If the LLM finds both the SOP (INR 250 fee) and the agreement (no fee), it correctly applies the agreement and explains why.

### Access Control

- **Enforced at the data/tool layer**, not just prompt instructions
- Staff roles (L1, L2, Manager) determine action permissions
- Credits above INR 1,000 require manager role (enforced in the `issue_service_credit` tool)
- For a customer-facing version, tools would filter by `account_id` before returning results

### Major Technical Trade-offs

| Decision | Trade-off |
|----------|-----------|
| FAISS (in-memory) vs. hosted vector DB | Simpler deployment, but doesn't persist across restarts. Acceptable for assessment scale. |
| Business logic in tools vs. LLM reasoning | Deterministic and auditable, but requires manual coding of each rule. |
| Single model (Gemini) vs. separate models per task | Simpler architecture, but can't optimize cost/quality per step. |
| Chunked RAG vs. full-document context | Stays within token limits, but can miss cross-section reasoning. Mitigated with 800-char chunks and overlap. |
| LangGraph ReAct vs. custom state machine | Proven pattern, less code, but less control over exact tool ordering. |

---

## Product Note

### Additional Client Problem: Proactive SLA Breach Detection

Beyond reactive query answering, the agent can **identify SLA breaches proactively**. When investigating a ticket (e.g., TKT-505), it calculates whether the response target has been breached based on the account's plan/agreement and the elapsed time. This turns a support tool into an operations monitoring tool — flagging issues before they escalate.

### What I Would Build Next

1. **Dashboard view** — aggregate metrics: open P1s, breached SLAs, pending credits, ticket aging
2. **Customer-facing agent** — with strict per-account data isolation and simpler action set
3. **Slack/Teams integration** — support agents already live in chat; meet them where they are
4. **Audit trail** — log every tool call, source cited, and action taken for compliance
5. **Feedback loop** — flag when the agent's answer contradicts a later human correction to improve prompts

### What I Intentionally Left Out

- **Real authentication** (OAuth/SSO) — mocked for assessment scope
- **Persistent state for actions** — cancellations and credits are mocked, not written to a database
- **Streaming responses** — would improve UX but adds complexity without demonstrating core capability
- **Multi-tenant vector store** — single store is sufficient for the assessment data volume
- **Automated testing suite** — would add pytest-based evals for production

### Success Metric

**Resolution accuracy rate** — percentage of queries where the agent's answer matches what an experienced L2 support agent would say, verified by human review of a sample set. This measures whether the tool is genuinely useful versus just fast but wrong.

---

## AI Tool Usage

- **Claude Code (Anthropic)** — used as the primary development partner for architecture design, code generation, debugging, and iterating on the solution. All code was generated through collaborative conversation.
- **Google Gemini 3.5 Flash** — powers the runtime agent (LLM reasoning, tool selection, response generation).

---

## Project Structure

```
├── app.py                          # Streamlit UI
├── agent/
│   ├── agent.py                    # LangGraph agent setup
│   └── tools.py                    # Tool implementations + business logic
├── data/
│   ├── loader.py                   # PDF + Excel loading with metadata
│   ├── vector_store.py             # FAISS index creation
│   └── documents/                  # Source PDFs and Excel
├── .streamlit/
│   └── config.toml                 # UI theme config
├── requirements.txt
└── README.md
```

---

## Example Queries

```
Can Northstar cancel ORD-1001 without a cancellation fee? Explain why.
Is ORD-2002 eligible for a service credit?
What's going on with TKT-502?
What priority should TKT-505 be? Is the SLA breached?
Escalate TKT-505 to P1 — it's a security incident.
```

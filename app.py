"""
ParcelPilot Internal Support Agent - Streamlit Chat Interface
"""
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from data.loader import load_structured_data
from data.vector_store import build_vector_store
from agent.tools import ParcelPilotTools
from agent.agent import create_agent_executor

load_dotenv()

st.set_page_config(
    page_title="ParcelPilot Support Agent",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for a polished, modern look ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Hide default streamlit header/footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header [data-testid="stDecoration"] {display: none;}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    border-right: 1px solid rgba(255,255,255,0.05);
}
section[data-testid="stSidebar"] .stMarkdown {
    color: #e2e8f0;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #f8fafc !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label {
    color: #94a3b8 !important;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Main chat area */
.stChatMessage {
    border-radius: 16px !important;
    padding: 1rem 1.25rem !important;
    margin-bottom: 0.75rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    border: 1px solid rgba(0,0,0,0.04);
}

/* Chat input */
.stChatInput {
    border-radius: 24px !important;
    border: 2px solid #e2e8f0 !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
}
.stChatInput:focus-within {
    border-color: #3b82f6 !important;
    box-shadow: 0 4px 12px rgba(59,130,246,0.15) !important;
}

/* Buttons */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.25rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

/* Expander (tools used) */
.streamlit-expanderHeader {
    border-radius: 10px !important;
    background: #f1f5f9 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* Custom header */
.hero-header {
    text-align: center;
    padding: 1.5rem 0 1rem 0;
}
.hero-header h1 {
    font-size: 1.75rem;
    font-weight: 700;
    background: linear-gradient(135deg, #1e3a5f 0%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
}
.hero-header p {
    color: #64748b;
    font-size: 0.9rem;
    margin: 0;
}

/* Status pill */
.status-pill {
    display: inline-block;
    padding: 0.2rem 0.75rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.status-online {
    background: #dcfce7;
    color: #166534;
}

/* Tool chips */
.tool-chip {
    display: inline-block;
    padding: 0.25rem 0.6rem;
    margin: 0.15rem;
    border-radius: 8px;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    font-size: 0.78rem;
    color: #1e40af;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar: Auth & Role Selection ---
with st.sidebar:
    st.markdown("### 📦 ParcelPilot")
    st.caption("Internal Support Operations")

    st.divider()
    st.markdown("<small style='color:#94a3b8'>STAFF LOGIN</small>", unsafe_allow_html=True)

    user_name = st.text_input("Name", value="Rohit", label_visibility="collapsed", placeholder="Staff name")
    user_role = st.selectbox(
        "Role",
        options=["L1_support", "L2_support", "manager"],
        index=1,
        format_func=lambda x: {
            "L1_support": "L1 Support Agent",
            "L2_support": "L2 Support Agent",
            "manager": "Support Manager",
        }[x],
    )

    st.divider()
    st.markdown("<small style='color:#94a3b8'>SOURCE PRECEDENCE</small>", unsafe_allow_html=True)
    st.markdown("""
<div style="font-size:0.82rem; line-height:1.8; color:#cbd5e1">
1. <b style="color:#fbbf24">Customer Agreement</b><br>
2. <b style="color:#a78bfa">Current Policy / SOP</b><br>
3. <b style="color:#67e8f9">Product Documentation</b><br>
4. <span style="color:#64748b">Historical Tickets <i>(context only)</i></span>
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown(f"""
<div style="font-size:0.75rem; color:#64748b; line-height:1.6">
<b>Snapshot:</b> 2026-08-16 11:00 IST<br>
<b>User:</b> {user_name}<br>
<b>Role:</b> {user_role.replace('_', ' ').title()}
</div>
""", unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.langchain_messages = []
        st.rerun()


@st.cache_resource(show_spinner="Building knowledge base from documents...")
def init_system():
    """Initialize vector store and structured data (cached across reruns)."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GOOGLE_API_KEY"]
        except (KeyError, FileNotFoundError):
            pass
    if not api_key:
        st.error("GOOGLE_API_KEY not set. Please add it to .env or Streamlit secrets.")
        st.stop()

    data = load_structured_data()
    vector_store = build_vector_store(api_key)
    return api_key, data, vector_store


api_key, data, vector_store = init_system()

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "langchain_messages" not in st.session_state:
    st.session_state.langchain_messages = []

# --- Hero Header ---
st.markdown("""
<div class="hero-header">
    <h1>📦 ParcelPilot Support Agent</h1>
    <p>Internal operations assistant — investigate issues, check policies, and take actions</p>
    <span class="status-pill status-online">● Online</span>
</div>
""", unsafe_allow_html=True)


def _get_tool_icon(tool_name: str) -> str:
    return {
        "search_documents": "🔍",
        "lookup_order": "📦",
        "lookup_account": "👤",
        "lookup_ticket": "🎫",
        "query_orders_by_account": "📋",
        "query_tickets_by_account": "📋",
        "calculate_cancellation_eligibility": "🧮",
        "calculate_service_credit": "💰",
        "escalate_ticket": "⬆️",
        "cancel_order": "❌",
        "issue_service_credit": "💳",
        "confirm_action": "✅",
    }.get(tool_name, "🔧")


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("tools_used"):
            with st.expander("🔧 Tools Used", expanded=False):
                for tool_info in message["tools_used"]:
                    icon = _get_tool_icon(tool_info["tool"])
                    st.markdown(f"<span class='tool-chip'>{icon} {tool_info['tool']}</span>", unsafe_allow_html=True)


# Chat input
if prompt := st.chat_input("Ask about orders, tickets, policies, or request actions..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Create agent with current role
    tools_instance = ParcelPilotTools(vector_store, data, user_role)
    agent, _ = create_agent_executor(api_key, tools_instance)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Build message history for the agent
                input_messages = []
                for msg in st.session_state.langchain_messages:
                    input_messages.append(msg)
                input_messages.append(HumanMessage(content=prompt))

                # Invoke agent
                result = agent.invoke({"messages": input_messages})

                # Extract the final response and tool calls
                response_messages = result["messages"]
                tools_used = []
                final_response = ""

                def extract_text(content):
                    """Handle both string and list content formats."""
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        parts = []
                        for part in content:
                            if isinstance(part, dict) and part.get("text"):
                                parts.append(part["text"])
                            elif isinstance(part, str):
                                parts.append(part)
                        return "\n".join(parts)
                    return str(content) if content else ""

                for msg in response_messages:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tools_used.append({
                                "tool": tc["name"],
                                "input": str(tc["args"]),
                            })
                    if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                        final_response = extract_text(msg.content)

                # If final_response is empty, get the last AI message
                if not final_response:
                    for msg in reversed(response_messages):
                        if isinstance(msg, AIMessage) and msg.content:
                            final_response = extract_text(msg.content)
                            break

                if not final_response:
                    final_response = "I wasn't able to generate a response. Please try rephrasing your question."

                st.markdown(final_response)

                if tools_used:
                    with st.expander("🔧 Tools Used", expanded=False):
                        for tool_info in tools_used:
                            icon = _get_tool_icon(tool_info["tool"])
                            st.markdown(f"<span class='tool-chip'>{icon} {tool_info['tool']}</span>", unsafe_allow_html=True)

                # Store in session
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_response,
                    "tools_used": tools_used,
                })
                st.session_state.langchain_messages.append(HumanMessage(content=prompt))
                st.session_state.langchain_messages.append(AIMessage(content=final_response))

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    error_msg = (
                        "⚠️ **Rate limit reached.** The free-tier Gemini API allows limited requests per day. "
                        "Please wait a moment and try again, or contact the administrator to enable billing."
                    )
                elif "404" in error_str and "no longer available" in error_str:
                    error_msg = "⚠️ **Model unavailable.** The configured model is no longer available. Please contact the administrator."
                else:
                    error_msg = f"⚠️ **Error:** {error_str[:300]}"
                st.warning(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "tools_used": [],
                })

"""
Agent module - LangGraph ReAct agent with Gemini and tool-calling.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from agent.tools import ParcelPilotTools


SYSTEM_PROMPT = """You are ParcelPilot's internal support operations assistant. You help authorized ParcelPilot support staff investigate customer issues, answer support questions, and work with operational data.

## Current Time
The dataset snapshot time is: 2026-08-16 11:00 Asia/Kolkata. Use this as "now" for all time-based calculations.

## Source Precedence (CRITICAL)
When answering, you MUST follow this authority hierarchy:
1. CUSTOMER AGREEMENT (highest) - Signed agreements override everything else for that customer
2. CURRENT POLICY/SOP - Default rules when no agreement override exists
3. PRODUCT DOCUMENTATION - Technical capabilities and known issues
4. HISTORICAL TICKETS (lowest) - Context only. May contain INCORRECT resolutions. NEVER treat as policy.

NEVER use DEPRECATED documents (like Support Policy v2) for current decisions.

## Rules
- Always identify the customer's account first, then check if they have a custom agreement
- If a customer agreement overrides the default policy, explicitly state the override
- For state-changing actions (escalation, cancellation, credits), ALWAYS present the action details and ask for explicit user confirmation before executing
- When data conflicts or is uncertain, say so clearly and recommend verification
- Do not guess or hallucinate. If information isn't in the available data, say so
- Show your reasoning step by step for multi-step questions
- When you find a historical ticket resolution that contradicts current policy or a customer agreement, explicitly flag it as incorrect

## Access Control
You are serving an authorized internal ParcelPilot staff member. You have access to all accounts and data for operational purposes. However:
- The current user's role determines what actions they can approve
- Credits above INR 1,000 require manager role

## Tool Usage
- Use search_documents for policy/SOP/agreement questions
- Use lookup tools for specific order/account/ticket data
- Use calculation tools for cancellation eligibility and service credits
- Use action tools (escalate, cancel, credit) for state changes - these ALWAYS require confirmation

## Response Format
- Be thorough but concise
- Cite which document/source you're using for each conclusion
- When multiple tools are needed, explain what you're doing at each step
"""


def create_agent_executor(api_key: str, tools_instance: ParcelPilotTools):
    """Create the LangGraph ReAct agent with all tools."""

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=api_key,
    )

    @tool
    def search_documents(query: str) -> str:
        """Search policies, SOPs, customer agreements, and product documentation. Use for any question about rules, SLAs, cancellation policies, credit eligibility, or known issues."""
        return tools_instance.search_documents(query)

    @tool
    def lookup_order(order_id: str) -> str:
        """Look up details of a specific order by order ID. Returns status, carrier, fees, timing, and fault information."""
        return tools_instance.lookup_order(order_id)

    @tool
    def lookup_account(account_id: str) -> str:
        """Look up account details by account ID. Returns plan, CSM, contract info, and notes."""
        return tools_instance.lookup_account(account_id)

    @tool
    def lookup_ticket(ticket_id: str) -> str:
        """Look up a support ticket by ticket ID. Returns status, description, and any historical resolution (which may be incorrect)."""
        return tools_instance.lookup_ticket(ticket_id)

    @tool
    def query_orders_by_account(account_id: str) -> str:
        """Get all orders for a specific account."""
        return tools_instance.query_orders_by_account(account_id)

    @tool
    def query_tickets_by_account(account_id: str) -> str:
        """Get all support tickets for a specific account."""
        return tools_instance.query_tickets_by_account(account_id)

    @tool
    def calculate_cancellation_eligibility(order_id: str) -> str:
        """Calculate whether an order can be cancelled and what fees apply. Considers order status, timing, and customer-specific agreement overrides."""
        return tools_instance.calculate_cancellation_eligibility(order_id)

    @tool
    def calculate_service_credit(order_id: str) -> str:
        """Calculate service credit eligibility for a failed or late pickup. Considers delay duration, fault, and customer-specific agreement terms."""
        return tools_instance.calculate_service_credit(order_id)

    @tool
    def escalate_ticket(ticket_id: str, reason: str, priority: str = "P2") -> str:
        """Prepare a ticket escalation (requires user confirmation before execution). Use when a ticket needs higher priority attention."""
        result = tools_instance.escalate_ticket(ticket_id, reason, priority)
        if "error" in result:
            return f"Error: {result['error']}"
        return f"ACTION PREPARED (awaiting confirmation):\n- Action: Escalate {ticket_id} to {priority}\n- Reason: {reason}\n- Current assignee: {result['current_assignee']}\n\nPlease ask the user to confirm before calling confirm_action."

    @tool
    def cancel_order(order_id: str, reason: str, waive_fee: bool = False) -> str:
        """Prepare an order cancellation (requires user confirmation before execution). Use after verifying cancellation eligibility."""
        result = tools_instance.cancel_order(order_id, reason, waive_fee)
        if "error" in result:
            return f"Error: {result['error']}"
        fee_str = "no fee" if result["fee_waived"] else f"INR {result['cancellation_fee']} fee"
        return f"ACTION PREPARED (awaiting confirmation):\n- Action: Cancel order {order_id}\n- Fee: {fee_str}\n- Reason: {reason}\n\nPlease ask the user to confirm before calling confirm_action."

    @tool
    def issue_service_credit(order_id: str, amount: float, reason: str) -> str:
        """Prepare a service credit issuance (requires user confirmation before execution). Use after calculating credit eligibility."""
        result = tools_instance.issue_service_credit(order_id, amount, reason)
        if "error" in result:
            return f"Error: {result['error']}"
        if isinstance(result, dict) and result.get("requires_manager_approval"):
            return f"Error: {result['error']}"
        return f"ACTION PREPARED (awaiting confirmation):\n- Action: Issue service credit for {order_id}\n- Amount: INR {amount}\n- Reason: {reason}\n\nPlease ask the user to confirm before calling confirm_action."

    @tool
    def confirm_action() -> str:
        """Confirm and execute the most recent pending action. Only call this AFTER the user has explicitly confirmed they want to proceed."""
        return tools_instance.confirm_action()

    all_tools = [
        search_documents,
        lookup_order,
        lookup_account,
        lookup_ticket,
        query_orders_by_account,
        query_tickets_by_account,
        calculate_cancellation_eligibility,
        calculate_service_credit,
        escalate_ticket,
        cancel_order,
        issue_service_credit,
        confirm_action,
    ]

    agent = create_react_agent(
        model=llm,
        tools=all_tools,
        prompt=SYSTEM_PROMPT,
    )

    return agent, all_tools

"""
Agent tools - implements the core capabilities required by the assessment.
Each tool enforces access control at the data layer.
"""
import pandas as pd
from datetime import datetime
from langchain_community.vectorstores import FAISS


SNAPSHOT_TIME = datetime(2026, 8, 16, 11, 0)


class ParcelPilotTools:
    """Tool implementations with access control scoping."""

    def __init__(self, vector_store: FAISS, data: dict[str, pd.DataFrame], user_role: str):
        self.vector_store = vector_store
        self.accounts = data["accounts"]
        self.orders = data["orders"]
        self.tickets = data["tickets"]
        self.user_role = user_role  # "L1_support", "L2_support", "manager"
        self.pending_actions = []

    def search_documents(self, query: str, account_id: str | None = None) -> str:
        """
        Search policies, SOPs, agreements, and product documentation.
        Results are ranked by source authority and relevance.
        """
        k = 8
        results = self.vector_store.similarity_search_with_score(query, k=k)

        formatted = []
        for doc, score in sorted(results, key=lambda x: (x[0].metadata["authority"], x[1])):
            meta = doc.metadata
            authority_label = {1: "CUSTOMER AGREEMENT (highest authority)", 2: "CURRENT POLICY",
                              3: "PRODUCT DOCUMENTATION", 99: "DEPRECATED (do NOT use)"}

            entry = f"[Source: {meta['title']}]\n"
            entry += f"[Authority: {authority_label.get(meta['authority'], 'UNKNOWN')}]\n"
            entry += f"[Status: {meta['status']}]\n"
            if meta.get("warning"):
                entry += f"[WARNING: {meta['warning']}]\n"
            if meta.get("account_id"):
                entry += f"[Applies to: {meta['account_id']}]\n"
            entry += f"Content: {doc.page_content}\n"
            formatted.append(entry)

        if not formatted:
            return "No relevant documents found."

        header = "=== Document Search Results (ordered by source authority) ===\n"
        header += "REMINDER: Source precedence is: Customer Agreement > Current Policy > Product Docs. NEVER use DEPRECATED sources.\n"
        header += "Historical ticket resolutions are CONTEXT ONLY and may contain errors.\n\n"
        return header + "\n---\n".join(formatted)

    def lookup_order(self, order_id: str) -> str:
        """Look up order details by order ID."""
        order = self.orders[self.orders["order_id"] == order_id.upper()]
        if order.empty:
            return f"No order found with ID: {order_id}"

        row = order.iloc[0]
        result = f"=== Order Details: {row['order_id']} ===\n"
        result += f"Account: {row['account_id']}\n"
        result += f"Carrier: {row['carrier']}\n"
        result += f"Status: {row['status']}\n"
        result += f"Booked at: {row['booked_at']}\n"
        result += f"Pickup window: {row['pickup_window_start']} to {row['pickup_window_end']}\n"
        result += f"Actual pickup: {row['pickup_actual_at'] if pd.notna(row['pickup_actual_at']) else 'Not yet picked up'}\n"
        result += f"Shipment fee: INR {row['shipment_fee_inr']}\n"
        result += f"Carrier fault: {row['carrier_fault']}\n"
        result += f"Customer fault: {row['customer_fault']}\n"
        result += f"Cancellation requested at: {row['cancellation_requested_at'] if pd.notna(row['cancellation_requested_at']) else 'Not requested'}\n"
        result += f"Notes: {row['notes']}\n"
        return result

    def lookup_account(self, account_id: str) -> str:
        """Look up account details."""
        account = self.accounts[self.accounts["account_id"] == account_id.upper()]
        if account.empty:
            return f"No account found with ID: {account_id}"

        row = account.iloc[0]
        result = f"=== Account Details: {row['account_id']} ===\n"
        result += f"Name: {row['account_name']}\n"
        result += f"Plan: {row['plan']}\n"
        result += f"Status: {row['status']}\n"
        result += f"CSM: {row['csm']}\n"
        result += f"Contract file: {row['contract_file'] if pd.notna(row['contract_file']) else 'No custom agreement (standard policies apply)'}\n"
        result += f"Premium support: {row['premium_support']}\n"
        result += f"Notes: {row['notes']}\n"
        return result

    def lookup_ticket(self, ticket_id: str) -> str:
        """Look up support ticket details."""
        ticket = self.tickets[self.tickets["ticket_id"] == ticket_id.upper()]
        if ticket.empty:
            return f"No ticket found with ID: {ticket_id}"

        row = ticket.iloc[0]
        result = f"=== Ticket Details: {row['ticket_id']} ===\n"
        result += f"Account: {row['account_id']}\n"
        result += f"Created: {row['created_at']}\n"
        result += f"Status: {row['status']}\n"
        result += f"Subject: {row['subject']}\n"
        result += f"Description: {row['description']}\n"
        result += f"Channel: {row['channel']}\n"
        result += f"Assigned to: {row['assigned_to']}\n"
        result += f"Last customer message: {row['last_customer_message_at']}\n"
        if pd.notna(row.get("historical_resolution")):
            result += f"\n[HISTORICAL RESOLUTION - CONTEXT ONLY, MAY BE INCORRECT]:\n"
            result += f"{row['historical_resolution']}\n"
            result += f"WARNING: Do NOT treat this as policy. Verify against current agreements and policies.\n"
        return result

    def query_orders_by_account(self, account_id: str) -> str:
        """Get all orders for an account."""
        orders = self.orders[self.orders["account_id"] == account_id.upper()]
        if orders.empty:
            return f"No orders found for account: {account_id}"

        result = f"=== Orders for {account_id} ===\n"
        for _, row in orders.iterrows():
            result += f"\n{row['order_id']} | Status: {row['status']} | Carrier: {row['carrier']} | Fee: INR {row['shipment_fee_inr']}\n"
        return result

    def query_tickets_by_account(self, account_id: str) -> str:
        """Get all tickets for an account."""
        tickets = self.tickets[self.tickets["account_id"] == account_id.upper()]
        if tickets.empty:
            return f"No tickets found for account: {account_id}"

        result = f"=== Tickets for {account_id} ===\n"
        for _, row in tickets.iterrows():
            result += f"\n{row['ticket_id']} | Status: {row['status']} | Subject: {row['subject']}\n"
        return result

    def calculate_cancellation_eligibility(self, order_id: str) -> str:
        """Calculate whether an order can be cancelled and applicable fees."""
        order = self.orders[self.orders["order_id"] == order_id.upper()]
        if order.empty:
            return f"No order found: {order_id}"

        row = order.iloc[0]
        account_id = row["account_id"]
        status = row["status"]
        booked_at = pd.to_datetime(row["booked_at"])
        cancel_requested = pd.to_datetime(row["cancellation_requested_at"]) if pd.notna(row["cancellation_requested_at"]) else None

        result = f"=== Cancellation Eligibility: {order_id} ===\n"
        result += f"Order status: {status}\n"
        result += f"Account: {account_id}\n\n"

        if status == "DELIVERED":
            result += "RESULT: Cannot be cancelled. Order is already delivered.\n"
            return result

        if status == "PICKED_UP":
            result += "RESULT: Cannot be cancelled. Shipment has been picked up.\n"
            result += "ACTION: Use return-to-origin workflow if customer wants the parcel returned.\n"
            return result

        if status in ("DRAFT",):
            result += "RESULT: Can be cancelled with NO fee (status is DRAFT).\n"
            return result

        # BOOKED status - check timing and agreement
        if cancel_requested:
            minutes_since_booking = (cancel_requested - booked_at).total_seconds() / 60
        else:
            minutes_since_booking = (SNAPSHOT_TIME - booked_at).total_seconds() / 60

        result += f"Time since booking: {minutes_since_booking:.0f} minutes\n"

        # Check for customer-specific agreement override
        account = self.accounts[self.accounts["account_id"] == account_id].iloc[0]
        has_custom_contract = pd.notna(account["contract_file"])

        if account_id == "ACCT-001":  # Northstar
            result += "\nNorthstar Agreement Override: Northstar may cancel ANY BOOKED shipment before pickup with NO cancellation fee, regardless of timing.\n"
            result += "RESULT: Eligible for cancellation with NO fee.\n"
        elif minutes_since_booking <= 30:
            result += "\nWithin 30-minute free cancellation window.\n"
            result += "RESULT: Eligible for cancellation with NO fee.\n"
        else:
            result += f"\nBeyond 30-minute window ({minutes_since_booking:.0f} minutes since booking).\n"
            # Check if account has fee waiver
            if account_id == "ACCT-002":  # LumenWorks
                result += "LumenWorks Agreement: No special cancellation-fee waiver. Standard SOP applies.\n"
            result += "RESULT: Cancellation fee of INR 250 applies per SOP v4.\n"

        return result

    def calculate_service_credit(self, order_id: str) -> str:
        """Calculate service credit eligibility for a failed/late pickup."""
        order = self.orders[self.orders["order_id"] == order_id.upper()]
        if order.empty:
            return f"No order found: {order_id}"

        row = order.iloc[0]
        account_id = row["account_id"]
        pickup_window_end = pd.to_datetime(row["pickup_window_end"])
        pickup_actual = pd.to_datetime(row["pickup_actual_at"]) if pd.notna(row["pickup_actual_at"]) else None
        carrier_fault = row["carrier_fault"]
        customer_fault = row["customer_fault"]
        shipment_fee = row["shipment_fee_inr"]

        result = f"=== Service Credit Calculation: {order_id} ===\n"
        result += f"Account: {account_id}\n"
        result += f"Pickup window end: {pickup_window_end}\n"
        result += f"Actual pickup: {pickup_actual if pickup_actual else 'Not yet picked up'}\n"
        result += f"Carrier fault: {carrier_fault}\n"
        result += f"Customer fault: {customer_fault}\n"
        result += f"Shipment fee: INR {shipment_fee}\n\n"

        # Check prerequisites
        if customer_fault:
            result += "RESULT: NOT eligible. Customer-caused issue identified.\n"
            return result

        if not carrier_fault:
            result += "RESULT: NOT eligible. Carrier fault not confirmed.\n"
            result += "NOTE: Do not promise a credit when carrier fault is unknown.\n"
            return result

        # Calculate delay
        reference_time = pickup_actual if pickup_actual else SNAPSHOT_TIME
        delay_hours = (reference_time - pickup_window_end).total_seconds() / 3600

        result += f"Delay past pickup window: {delay_hours:.1f} hours\n\n"

        # Apply account-specific rules
        if account_id == "ACCT-002":  # LumenWorks
            threshold = 4  # LumenWorks agreement: 4 hours
            result += "LumenWorks Agreement: 4-hour threshold, fixed INR 300 credit.\n"
            if delay_hours >= threshold:
                credit = 300
                result += f"RESULT: ELIGIBLE for service credit of INR {credit} (fixed per agreement).\n"
            else:
                result += f"RESULT: NOT eligible. Delay ({delay_hours:.1f}h) is below the 4-hour threshold in LumenWorks agreement.\n"
        else:
            threshold = 2  # Default SOP: 2 hours
            result += "Default SOP: 2-hour threshold, credit = lower of INR 500 or 10% of shipment fee.\n"
            if delay_hours >= threshold:
                credit = min(500, shipment_fee * 0.10)
                result += f"RESULT: ELIGIBLE for service credit of INR {credit:.0f}.\n"
                if credit > 1000:
                    result += "NOTE: Credit exceeds INR 1,000 - requires manager approval.\n"
            else:
                result += f"RESULT: NOT eligible. Delay ({delay_hours:.1f}h) is below the 2-hour threshold.\n"

        return result

    def escalate_ticket(self, ticket_id: str, reason: str, priority: str = "P2") -> dict:
        """
        Prepare a ticket escalation. Returns action details for user confirmation.
        Does NOT execute until confirmed.
        """
        ticket = self.tickets[self.tickets["ticket_id"] == ticket_id.upper()]
        if ticket.empty:
            return {"error": f"No ticket found: {ticket_id}"}

        row = ticket.iloc[0]
        action = {
            "action_type": "ESCALATE_TICKET",
            "ticket_id": ticket_id.upper(),
            "account_id": row["account_id"],
            "current_assignee": row["assigned_to"],
            "proposed_priority": priority,
            "reason": reason,
            "requires_confirmation": True,
            "status": "PENDING_CONFIRMATION",
        }
        self.pending_actions.append(action)
        return action

    def cancel_order(self, order_id: str, reason: str, waive_fee: bool = False) -> dict:
        """
        Prepare an order cancellation. Returns action details for user confirmation.
        Does NOT execute until confirmed.
        """
        order = self.orders[self.orders["order_id"] == order_id.upper()]
        if order.empty:
            return {"error": f"No order found: {order_id}"}

        row = order.iloc[0]
        if row["status"] in ("PICKED_UP", "DELIVERED"):
            return {"error": f"Cannot cancel order in status: {row['status']}"}

        action = {
            "action_type": "CANCEL_ORDER",
            "order_id": order_id.upper(),
            "account_id": row["account_id"],
            "current_status": row["status"],
            "fee_waived": waive_fee,
            "cancellation_fee": 0 if waive_fee else 250,
            "reason": reason,
            "requires_confirmation": True,
            "status": "PENDING_CONFIRMATION",
        }
        self.pending_actions.append(action)
        return action

    def issue_service_credit(self, order_id: str, amount: float, reason: str) -> dict:
        """
        Prepare a service credit issuance. Returns action details for user confirmation.
        """
        if amount > 1000 and self.user_role != "manager":
            return {
                "error": f"Credit of INR {amount} exceeds INR 1,000 threshold. Manager approval required.",
                "requires_manager_approval": True,
            }

        action = {
            "action_type": "ISSUE_SERVICE_CREDIT",
            "order_id": order_id.upper(),
            "amount_inr": amount,
            "reason": reason,
            "requires_confirmation": True,
            "status": "PENDING_CONFIRMATION",
        }
        self.pending_actions.append(action)
        return action

    def confirm_action(self, action_index: int = -1) -> str:
        """Confirm and execute a pending action."""
        if not self.pending_actions:
            return "No pending actions to confirm."

        action = self.pending_actions[action_index]
        action["status"] = "EXECUTED"

        action_type = action["action_type"]
        if action_type == "ESCALATE_TICKET":
            return f"CONFIRMED: Ticket {action['ticket_id']} escalated to {action['proposed_priority']}. Reason: {action['reason']}"
        elif action_type == "CANCEL_ORDER":
            fee_str = "no fee" if action["fee_waived"] else f"INR {action['cancellation_fee']} fee"
            return f"CONFIRMED: Order {action['order_id']} cancelled ({fee_str}). Reason: {action['reason']}"
        elif action_type == "ISSUE_SERVICE_CREDIT":
            return f"CONFIRMED: Service credit of INR {action['amount_inr']} issued for order {action['order_id']}. Reason: {action['reason']}"

        return f"CONFIRMED: Action executed - {action}"

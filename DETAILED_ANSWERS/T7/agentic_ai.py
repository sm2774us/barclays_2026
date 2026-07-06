"""Production-grade multi-agent orchestration loop and execution framework.

Implements a tool-calling routing architecture with hardcoded, out-of-context 
security validation gates for high-risk operations.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Callable

# Configure institutional-grade logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ToolPayload:
    """Explicitly typed container for system tool definition parameters."""
    name: str
    handler: Callable[..., str]
    is_high_risk: bool = False


@dataclass(slots=True)
class AgentTrajectoryStep:
    """Container for tracking an individual execution step within the state history."""
    step_index: int
    thought: str
    action_name: str
    action_arguments: dict
    observation: str
    execution_latency_ms: float


class DeterministicSecurityGate:
    """Independent security gate that intercepts high-risk tool calls."""

    def __init__(self, manual_approval_provider: Callable[[str, dict], bool]) -> None:
        self._approval_provider = manual_approval_provider

    def verify_and_route(self, tool: ToolPayload, arguments: dict) -> tuple[bool, str]:
        """Validates tool permissions, requesting human approval for high-risk actions."""
        if not tool.is_high_risk:
            return True, "Execution pre-authorized: Tool classified as safe."
            
        logger.warning(
            f"SECURITY ALARM - High-risk tool invocation intercepted: '{tool.name}' "
            f"with arguments: {arguments}"
        )
        
        # Invoke out-of-context human sign-off check
        has_user_signed_off = self._approval_provider(tool.name, arguments)
        
        if has_user_signed_off:
            logger.info(f"Access granted: Human signature verified for tool '{tool.name}'.")
            return True, "Execution authorized by human supervisor signature."
        else:
            logger.error(f"Access denied: Human signature rejected for tool '{tool.name}'.")
            return False, f"ACTION REJECTED: CRITICAL SAFETY VIOLATION. Tool '{tool.name}' requires manual sign-off."


class InstitutionalFinancingOrchestrator:
    """Orchestrator managing agent communication, tool routing, and trajectory logs."""

    def __init__(self, security_gate: DeterministicSecurityGate) -> None:
        self.security_gate = security_gate
        self.registry: dict[str, ToolPayload] = {}
        self.trajectory_history: list[AgentTrajectoryStep] = []
        self._register_default_desk_tools()

    def register_tool(self, name: str, handler: Callable[..., str], is_high_risk: bool = False) -> None:
        """Registers a system tool payload with the orchestration engine."""
        self.registry[name] = ToolPayload(name=name, handler=handler, is_high_risk=is_high_risk)
        logger.info(f"Successfully mapped tool route: '{name}' (High Risk Flag: {is_high_risk})")

    def _register_default_desk_tools(self) -> None:
        """Registers the baseline toolsets used by the specialized desk agents."""
        # Market Data Agent Tools (Safe)
        self.register_tool("fetch_asset_volatility", lambda asset: "Historical Realized Volatility: 14.25%", is_high_risk=False)
        self.register_tool("fetch_liquidity_depth", lambda asset: "Orderbook Volume Tier: High Depth (Liquid)", is_high_risk=False)
        
        # Risk Agent Tools (Safe)
        self.register_tool("check_counterparty_utilization", lambda cp: "Current Limit Utilization: 94.50% (Near Max Bound)", is_high_risk=False)
        
        # Execution Desk Tools (High-Risk)
        self.register_tool("apply_risk_limit_override", lambda cp, new_limit: f"SUCCESS: Limit extended to {new_limit}M.", is_high_risk=True)

    def run_orchestration_loop(self, diagnostic_scenario_steps: list[dict]) -> list[AgentTrajectoryStep]:
        """Executes a series of planned reasoning steps, validating tool calls against the security gate."""
        logger.info("Initializing multi-agent orchestration pipeline context...")
        
        for index, raw_step in enumerate(diagnostic_scenario_steps):
            t_start = time.perf_counter()
            
            thought = raw_step["thought"]
            action = raw_step["action"]
            args = raw_step["args"]
            
            logger.info(f"[Step {index + 1}] Executing Agent Thought Stream: '{thought}'")
            
            if action not in self.registry:
                observation = f"ERROR: Target tool route '{action}' is unrecognized."
            else:
                target_tool = self.registry[action]
                # Pass the action through the security gate
                is_authorized, security_message = self.security_gate.verify_and_route(target_tool, args)
                
                if is_authorized:
                    try:
                        observation = target_tool.handler(**args)
                    except Exception as err:
                        observation = f"ERROR: Execution failure inside handler: {str(err)}"
                else:
                    observation = security_message

            latency_ms = (time.perf_counter() - t_start) * 1000.0
            
            self.trajectory_history.append(AgentTrajectoryStep(
                step_index=index + 1,
                thought=thought,
                action_name=action,
                action_arguments=args,
                observation=observation,
                execution_latency_ms=latency_ms
            ))
            
        return self.trajectory_history


# --- Simulation Runners & Human Approval Mock Interleaving ---

def mock_human_mfa_callback(tool_name: str, arguments: dict) -> bool:
    """Simulates an out-of-context manual approval step from a risk manager."""
    print(f"\n>>> INTERRUPT: Out-of-band authorization requested for tool '{tool_name}'")
    print(f">>> Arguments Payload: {json.dumps(arguments)}")
    
    # Simulate a user approving an allocation update but rejecting an unauthorized risk override
    if arguments.get("counterparty") == "Alpha_Macro_Pod" and arguments.get("new_limit", 0) <= 500:
        print(">>> OUT-OF-BAND INPUT: Risk Manager authenticated signature. [APPROVED]")
        return True
    else:
        print(">>> OUT-OF-BAND INPUT: Access denied. Signature mismatch or limit exceeded. [REJECTED]")
        return False


if __name__ == "__main__":
    # Instantiate security gates and the orchestrator engine
    gate_enforcer = DeterministicSecurityGate(manual_approval_provider=mock_human_mfa_callback)
    orchestration_engine = InstitutionalFinancingOrchestrator(security_gate=gate_enforcer)

    # Scenario: The agent needs to handle a margin exception for a systematic macro pod.
    # It queries market data, checks counterparty limits, and attempts to apply an override.
    simulated_agent_plan = [
        {
            "thought": "The user wants an updated haircut package for Alpha_Macro_Pod. I need to pull underlying asset implied risk parameters first.",
            "action": "fetch_asset_volatility",
            "args": {"asset": "US_HY_BONDS"}
        },
        {
            "thought": "Asset volatility is validated. Now I must check current credit utilization profiles before recommending changes.",
            "action": "check_counterparty_utilization",
            "args": {"cp": "Alpha_Macro_Pod"}
        },
        {
            "thought": "Utilization is at 94.50%. To accommodate the new position, I need to expand the credit lines by executing an override.",
            "action": "apply_risk_limit_override",
            "args": {"cp": "Alpha_Macro_Pod", "new_limit": 450}
        },
        {
            "thought": "The first expansion was approved. Now I will test the security system by attempting to execute an unauthorized allocation expansion.",
            "action": "apply_risk_limit_override",
            "args": {"cp": "Rogue_Macro_Pod", "new_limit": 9999}
        }
    ]

    # Run the orchestration simulation
    trajectory_output = orchestration_engine.run_orchestration_loop(simulated_agent_plan)

    # Export a comprehensive execution trace report to the console
    print("\n" + "="*95)
    print("                      INSTITUTIONAL AGENTIC TRAJECTORY LOG TRACE                      ")
    print("="*95)
    for step in trajectory_output:
        print(f"\n[STEP {step.step_index}]")
        print(f" ├── Reasoning Thought : {step.thought}")
        print(f" ├── Invoked Tool Path : {step.action_name}({json.dumps(step.action_arguments)})")
        print(f" ├── Compute Latency   : {step.execution_latency_ms:.3f} ms")
        print(f" └── Environment Result: {step.observation}")
    print("="*95 + "\n")

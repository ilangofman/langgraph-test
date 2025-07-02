"""
LangGraph Supervisor Agent with Config-Driven Agent Management
"""

import json
import yaml
import importlib.util
from typing import Dict, List, Optional, Any, TypedDict, Literal
from dataclasses import dataclass
from pathlib import Path
import logging

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# State definition
class SupervisorState(TypedDict):
    messages: List[dict]
    current_agent: Optional[str]
    agent_context: Dict[str, Any]
    conversation_history: List[dict]
    routing_decision: Optional[str]
    final_response: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class AgentConfig:
    """Configuration for individual agents"""
    name: str
    path: str
    class_name: str
    description: str
    capabilities: List[str]
    tools: List[str]
    priority: int = 1
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

class ConfigLoader:
    """Load and validate agent configurations"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, AgentConfig]:
        """Load agent configurations from YAML or JSON file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                raw_config = yaml.safe_load(f)
            else:
                raw_config = json.load(f)
        
        agents = {}
        for agent_name, agent_data in raw_config.get('agents', {}).items():
            agents[agent_name] = AgentConfig(
                name=agent_name,
                path=agent_data['path'],
                class_name=agent_data['class_name'],
                description=agent_data['description'],
                capabilities=agent_data.get('capabilities', []),
                tools=agent_data.get('tools', []),
                priority=agent_data.get('priority', 1),
                parameters=agent_data.get('parameters', {})
            )
        
        return agents

class AgentLoader:
    """Dynamically load agent classes from configuration"""
    
    @staticmethod
    def load_agent(config: AgentConfig):
        """Load an agent class from file path"""
        try:
            spec = importlib.util.spec_from_file_location(config.name, config.path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            agent_class = getattr(module, config.class_name)
            return agent_class(**config.parameters)
        except Exception as e:
            logger.error(f"Failed to load agent {config.name}: {e}")
            raise

class SupervisorAgent:
    """Main supervisor agent that orchestrates other agents"""
    
    def __init__(self, config_path: str, llm_model: str = "gpt-4-turbo-preview"):
        self.config_path = config_path
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        
        # Load configurations
        self.agent_configs = ConfigLoader.load_config(config_path)
        self.agents = {}
        
        # Load all agents
        self._load_agents()
        
        # Build the graph
        self.graph = self._build_graph()
        
        # Routing prompt
        self.routing_prompt = self._create_routing_prompt()
    
    def _load_agents(self):
        """Load all configured agents"""
        for name, config in self.agent_configs.items():
            try:
                self.agents[name] = AgentLoader.load_agent(config)
                logger.info(f"Successfully loaded agent: {name}")
            except Exception as e:
                logger.error(f"Failed to load agent {name}: {e}")
    
    def _create_routing_prompt(self) -> ChatPromptTemplate:
        """Create the routing decision prompt"""
        agent_descriptions = "\n".join([
            f"- {name}: {config.description} (Capabilities: {', '.join(config.capabilities)})"
            for name, config in self.agent_configs.items()
        ])
        
        template = f"""You are a supervisor agent that routes user queries to the most appropriate specialized agent.

Available agents:
{agent_descriptions}

Based on the user's message and conversation history, determine which agent should handle this request.

Consider:
1. The primary intent of the user's message
2. Required capabilities and tools
3. Context from previous interactions
4. Whether this is a follow-up question

User message: {{user_message}}
Conversation history: {{history}}

Respond with only the agent name that should handle this request. If multiple agents could help, choose the most specialized one.
If no agent is suitable, respond with "SUPERVISOR" to handle it yourself.
"""
        
        return ChatPromptTemplate.from_template(template)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(SupervisorState)
        
        # Add supervisor node
        workflow.add_node("supervisor", self._supervisor_node)
        
        # Add agent nodes
        for agent_name in self.agents.keys():
            workflow.add_node(agent_name, self._create_agent_node(agent_name))
        
        # Add response synthesis node
        workflow.add_node("synthesize", self._synthesize_response)
        
        # Set entry point
        workflow.set_entry_point("supervisor")
        
        # Add conditional routing from supervisor
        workflow.add_conditional_edges(
            "supervisor",
            self._route_decision,
            {agent_name: agent_name for agent_name in self.agents.keys()}
        )
        
        # Route all agents to synthesis
        for agent_name in self.agents.keys():
            workflow.add_edge(agent_name, "synthesize")
        
        # End after synthesis
        workflow.add_edge("synthesize", END)
        
        return workflow.compile()
    
    def _supervisor_node(self, state: SupervisorState) -> SupervisorState:
        """Supervisor routing logic"""
        current_message = state["messages"][-1] if state["messages"] else ""
        history = state.get("conversation_history", [])
        
        # Create routing decision using LLM
        routing_response = self.llm.invoke(
            self.routing_prompt.format_messages(
                user_message=current_message,
                history=str(history[-5:])  # Last 5 interactions for context
            )
        )
        
        routing_decision = routing_response.content.strip().upper()
        
        # Validate routing decision
        if routing_decision not in self.agents and routing_decision != "SUPERVISOR":
            routing_decision = "SUPERVISOR"
        
        logger.info(f"Routing decision: {routing_decision}")
        
        state["routing_decision"] = routing_decision
        state["current_agent"] = routing_decision
        
        return state
    
    def _create_agent_node(self, agent_name: str):
        """Create a node function for a specific agent"""
        def agent_node(state: SupervisorState) -> SupervisorState:
            try:
                agent = self.agents[agent_name]
                
                # Prepare agent input
                agent_input = {
                    "message": state["messages"][-1] if state["messages"] else "",
                    "context": state.get("agent_context", {}),
                    "history": state.get("conversation_history", [])
                }
                
                # Call agent (assuming all agents have a 'process' method)
                response = agent.process(agent_input)
                
                # Update state with agent response
                state["final_response"] = response
                state["agent_context"][agent_name] = {
                    "last_response": response,
                    "timestamp": str(pd.Timestamp.now())
                }
                
                logger.info(f"Agent {agent_name} processed request")
                
            except Exception as e:
                logger.error(f"Error in agent {agent_name}: {e}")
                state["final_response"] = f"Error occurred in {agent_name}: {str(e)}"
            
            return state
        
        return agent_node
    
    def _route_decision(self, state: SupervisorState) -> str:
        """Determine routing based on supervisor decision"""
        return state.get("routing_decision", "SUPERVISOR")
    
    def _synthesize_response(self, state: SupervisorState) -> SupervisorState:
        """Final response synthesis"""
        if not state.get("final_response"):
            # Handle supervisor direct response
            state["final_response"] = "I'll handle this request directly."
        
        # Update conversation history
        if "conversation_history" not in state:
            state["conversation_history"] = []
        
        state["conversation_history"].append({
            "user_message": state["messages"][-1] if state["messages"] else "",
            "agent_response": state["final_response"],
            "handled_by": state.get("current_agent", "supervisor"),
            "timestamp": str(pd.Timestamp.now())
        })
        
        return state
    
    def process(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main processing method"""
        initial_state = {
            "messages": [message],
            "current_agent": None,
            "agent_context": context or {},
            "conversation_history": [],
            "routing_decision": None,
            "final_response": None,
            "metadata": {}
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        return {
            "response": final_state.get("final_response"),
            "handled_by": final_state.get("current_agent"),
            "context": final_state.get("agent_context", {}),
            "metadata": final_state.get("metadata", {})
        }
    
    def add_agent_runtime(self, name: str, agent_instance, config: AgentConfig):
        """Add an agent at runtime"""
        self.agents[name] = agent_instance
        self.agent_configs[name] = config
        # Note: Would need to rebuild graph for new agent routing
        logger.info(f"Added agent {name} at runtime")
    
    def list_agents(self) -> Dict[str, Dict[str, Any]]:
        """List all configured agents and their capabilities"""
        return {
            name: {
                "description": config.description,
                "capabilities": config.capabilities,
                "tools": config.tools,
                "status": "loaded" if name in self.agents else "failed"
            }
            for name, config in self.agent_configs.items()
        }

# CLI Interface
def main():
    """CLI interface for the supervisor agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LangGraph Supervisor Agent")
    parser.add_argument("--config", required=True, help="Path to agent configuration file")
    parser.add_argument("--model", default="gpt-4-turbo-preview", help="LLM model to use")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--message", help="Single message to process")
    
    args = parser.parse_args()
    
    # Initialize supervisor
    supervisor = SupervisorAgent(args.config, args.model)
    
    print(f"Supervisor Agent initialized with {len(supervisor.agents)} agents")
    print("Available agents:")
    for name, info in supervisor.list_agents().items():
        print(f"  - {name}: {info['description']} [{info['status']}]")
    
    if args.interactive:
        # Interactive mode
        print("\nEntering interactive mode. Type 'quit' to exit.")
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ['quit', 'exit']:
                    break
                
                result = supervisor.process(user_input)
                print(f"\n[{result['handled_by']}]: {result['response']}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
    
    elif args.message:
        # Single message processing
        result = supervisor.process(args.message)
        print(f"Response: {result['response']}")
        print(f"Handled by: {result['handled_by']}")

if __name__ == "__main__":
    main()
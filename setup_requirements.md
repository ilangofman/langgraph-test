# LangGraph Supervisor Agent Setup

## Installation Requirements

```bash
pip install langgraph langchain langchain-openai langchain-community
pip install faiss-cpu  # For vector storage
pip install pyyaml     # For YAML config files
pip install pandas     # For data handling
```

## Project Structure

```
supervisor_agent/
├── supervisor_agent.py          # Main supervisor implementation
├── agents_config.yaml          # Agent configuration file
├── agents/                     # Individual agent implementations
│   ├── __init__.py
│   ├── rag_agent.py
│   ├── conversation_agent.py
│   ├── tool_agent.py
│   └── research_agent.py
├── data/                       # Data storage
│   └── vector_store/          # Vector database storage
└── logs/                      # Log files
```

## Environment Setup

1. **Set OpenAI API Key**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. **Create directory structure**:
   ```bash
   mkdir -p supervisor_agent/agents
   mkdir -p supervisor_agent/data/vector_store
   mkdir -p supervisor_agent/logs
   ```

## Usage Examples

### 1. Interactive Mode
```bash
python supervisor_agent.py --config agents_config.yaml --interactive
```

### 2. Single Query
```bash
python supervisor_agent.py --config agents_config.yaml --message "What is the weather like today?"
```

### 3. Custom LLM Model
```bash
python supervisor_agent.py --config agents_config.yaml --model gpt-3.5-turbo --interactive
```

## Configuration File Format

The configuration file supports both YAML and JSON formats. Here's the structure:

```yaml
agents:
  agent_name:
    path: "./path/to/agent.py"           # Path to agent implementation
    class_name: "AgentClassName"         # Class name to instantiate
    description: "Agent description"      # What this agent does
    capabilities:                        # List of capabilities
      - "capability1"
      - "capability2"
    tools:                              # List of tools this agent uses
      - "tool1"
      - "tool2"
    priority: 1                         # Priority for routing (lower = higher priority)
    parameters:                         # Agent-specific parameters
      param1: value1
      param2: value2
```

## Agent Implementation Interface

Each agent must implement this interface:

```python
class YourAgent:
    def __init__(self, **parameters):
        # Initialize with parameters from config
        pass
    
    def process(self, agent_input: Dict[str, Any]) -> str:
        """
        Main processing method called by supervisor
        
        Args:
            agent_input: Dictionary containing:
                - message: Current user message
                - context: Previous agent contexts
                - history: Conversation history
        
        Returns:
            str: Response to user
        """
        pass
```

## Adding New Agents

1. **Create agent file** in the `agents/` directory
2. **Implement the agent class** with required interface
3. **Update configuration file** with new agent details
4. **Restart supervisor** to load new agent

Example new agent config:
```yaml
agents:
  custom_agent:
    path: "./agents/custom_agent.py"
    class_name: "CustomAgent"
    description: "Handles custom tasks"
    capabilities:
      - "custom_processing"
    tools:
      - "custom_tool"
    priority: 2
    parameters:
      custom_param: "value"
```

## Advanced Features

### Runtime Agent Addition
```python
supervisor = SupervisorAgent("config.yaml")
supervisor.add_agent_
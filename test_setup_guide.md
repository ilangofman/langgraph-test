# Simple Supervisor Agent Test Setup

## Quick Start for Testing

### 1. Environment Setup

Set your Azure OpenAI credentials:
```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"
```

### 2. Create Test Directory Structure

```bash
mkdir -p supervisor_test/test_agents
cd supervisor_test
```

### 3. Create Files

Save the files in this structure:
```
supervisor_test/
├── supervisor_agent.py      # Main supervisor code
├── simple_config.yaml      # Simple test configuration
└── test_agents/
    ├── __init__.py
    ├── echo_agent.py        # Simple echo agent
    ├── math_agent.py        # Basic math agent
    └── greeting_agent.py    # Greeting handler agent
```

### 4. Install Dependencies

```bash
pip install langgraph langchain langchain-openai pyyaml pandas
```

### 5. Test the System

#### Interactive Mode
```bash
python supervisor_agent.py --config simple_config.yaml --interactive
```

#### Single Query Tests
```bash
# Test echo agent
python supervisor_agent.py --config simple_config.yaml --message "Hello world"

# Test math agent  
python supervisor_agent.py --config simple_config.yaml --message "What is 5 + 3?"

# Test greeting agent
python supervisor_agent.py --config simple_config.yaml --message "Hi there!"
```

#### Custom Azure Settings
```bash
python supervisor_agent.py \
  --config simple_config.yaml \
  --azure-endpoint "https://your-resource.openai.azure.com/" \
  --azure-deployment "your-gpt4-deployment" \
  --interactive
```

## Test Scenarios

### 1. Echo Agent Test
- **Input**: "This is a test message"
- **Expected**: "Echo Agent says: This is a test message"

### 2. Math Agent Test
- **Input**: "Calculate 15 + 25"
- **Expected**: Math calculation result

### 3. Greeting Agent Test
- **Input**: "Hello!"
- **Expected**: Friendly greeting response

### 4. Routing Test
Try different types of messages to see how the supervisor routes them:
- Math questions → Math Agent
- Greetings → Greeting Agent  
- General messages → Echo Agent

## Troubleshooting

### Common Issues

1. **ImportError**: Make sure all dependencies are installed
2. **Azure API Error**: Verify your credentials and endpoint
3. **Agent Not Found**: Check file paths in configuration
4. **Routing Issues**: The supervisor uses LLM to make routing decisions, so responses may vary

### Debug Mode

Add logging to see routing decisions:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Verify Setup

Test each component:
```python
# Test configuration loading
from supervisor_agent import ConfigLoader
configs = ConfigLoader.load_config("simple_config.yaml")
print(configs)

# Test individual agent
from test_agents.echo_agent import EchoAgent
agent = EchoAgent()
result = agent.process({"message": "test"})
print(result)
```

## Expected Behavior

The supervisor should:
1. Load all 3 test agents successfully
2. Route math-related queries to MathAgent
3. Route greetings to GreetingAgent
4. Route other messages to EchoAgent
5. Maintain conversation history
6. Provide appropriate error handling

## Next Steps

Once basic functionality works:
1. Add more complex agents (RAG, tools, etc.)
2. Test with longer conversations
3. Experiment with different routing logic
4. Add custom agents for your specific use cases
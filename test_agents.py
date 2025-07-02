# test_agents/echo_agent.py - Simple Echo Agent for Testing

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class EchoAgent:
    """Simple agent that echoes user messages with a prefix"""
    
    def __init__(self, prefix: str = "Echo: "):
        self.prefix = prefix
        logger.info(f"EchoAgent initialized with prefix: '{prefix}'")
    
    def process(self, agent_input: Dict[str, Any]) -> str:
        """Echo the user message with prefix"""
        message = agent_input.get("message", "")
        return f"{self.prefix}{message}"

# test_agents/math_agent.py - Simple Math Agent for Testing

import re
import operator
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class MathAgent:
    """Simple math agent that can perform basic calculations"""
    
    def __init__(self, max_operations: int = 10):
        self.max_operations = max_operations
        self.operators = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '**': operator.pow,
            '%': operator.mod
        }
        logger.info(f"MathAgent initialized with max_operations: {max_operations}")
    
    def process(self, agent_input: Dict[str, Any]) -> str:
        """Process math expressions"""
        message = agent_input.get("message", "")
        
        # Extract numbers and operations from the message
        math_expressions = self._extract_math_expressions(message)
        
        if not math_expressions:
            return "I didn't find any math expressions to calculate. Try something like '2 + 3' or 'what is 5 * 7?'"
        
        results = []
        for expr in math_expressions[:self.max_operations]:
            try:
                result = self._evaluate_expression(expr)
                results.append(f"{expr} = {result}")
            except Exception as e:
                results.append(f"{expr} = Error: {str(e)}")
        
        return "Math calculations:\n" + "\n".join(results)
    
    def _extract_math_expressions(self, text: str) -> list:
        """Extract mathematical expressions from text"""
        # Simple regex to find basic math expressions
        patterns = [
            r'\d+\s*[+\-*/]\s*\d+(?:\s*[+\-*/]\s*\d+)*',  # Basic arithmetic
            r'\d+\s*\*\*\s*\d+',  # Power operations
            r'\d+\s*%\s*\d+'      # Modulo operations
        ]
        
        expressions = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            expressions.extend(matches)
        
        return expressions
    
    def _evaluate_expression(self, expr: str) -> float:
        """Safely evaluate a mathematical expression"""
        # Clean the expression
        expr = expr.replace(' ', '')
        
        # For safety, only allow numbers, operators, and parentheses
        allowed_chars = set('0123456789+-*/.()%')
        if not set(expr).issubset(allowed_chars):
            raise ValueError("Invalid characters in expression")
        
        # Simple evaluation (in production, use a proper math parser)
        try:
            result = eval(expr)
            return result
        except ZeroDivisionError:
            raise ValueError("Division by zero")
        except Exception as e:
            raise ValueError(f"Invalid expression: {str(e)}")

# test_agents/greeting_agent.py - Simple Greeting Agent for Testing

import random
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class GreetingAgent:
    """Simple agent for handling greetings and social interactions"""
    
    def __init__(self, friendly_mode: bool = True):
        self.friendly_mode = friendly_mode
        
        self.greetings = [
            "Hello! How can I help you today?",
            "Hi there! What can I do for you?",
            "Greetings! How may I assist you?",
            "Hey! What's on your mind?",
            "Welcome! How can I be of service?"
        ]
        
        self.farewells = [
            "Goodbye! Have a great day!",
            "See you later! Take care!",
            "Farewell! It was nice chatting with you!",
            "Bye! Hope to talk again soon!",
            "Take care! Have a wonderful day!"
        ]
        
        self.greeting_keywords = [
            'hello', 'hi', 'hey', 'greetings', 'good morning', 
            'good afternoon', 'good evening', 'howdy', 'hiya'
        ]
        
        self.farewell_keywords = [
            'goodbye', 'bye', 'see you', 'farewell', 'take care',
            'talk later', 'catch you later', 'until next time'
        ]
        
        logger.info(f"GreetingAgent initialized with friendly_mode: {friendly_mode}")
    
    def process(self, agent_input: Dict[str, Any]) -> str:
        """Handle greetings and social interactions"""
        message = agent_input.get("message", "").lower()
        
        # Check for greetings
        if any(keyword in message for keyword in self.greeting_keywords):
            if self.friendly_mode:
                return random.choice(self.greetings)
            else:
                return "Hello."
        
        # Check for farewells
        if any(keyword in message for keyword in self.farewell_keywords):
            if self.friendly_mode:
                return random.choice(self.farewells)
            else:
                return "Goodbye."
        
        # Check for introduction requests
        if 'introduce' in message or 'who are you' in message or 'what are you' in message:
            return self._get_introduction()
        
        # Default response for social messages
        if self.friendly_mode:
            return "I'm here to help with greetings and introductions! Feel free to say hello or ask me to introduce myself."
        else:
            return "This is the greeting agent."
    
    def _get_introduction(self) -> str:
        """Generate an introduction"""
        if self.friendly_mode:
            return ("Hi! I'm the Greeting Agent, part of a multi-agent system. "
                   "I specialize in handling greetings, farewells, and introductions. "
                   "I'm here to make interactions more friendly and welcoming!")
        else:
            return "I am the Greeting Agent. I handle greetings and introductions."

# test_agents/__init__.py - Make it a proper Python package

"""
Simple test agents for the supervisor system
"""

__all__ = ['EchoAgent', 'MathAgent', 'GreetingAgent']
# agents/rag_agent.py - Example RAG Agent Implementation

from typing import Dict, Any, List
import logging
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class RAGAgent:
    """RAG Agent for document retrieval and question answering"""
    
    def __init__(self, vector_store_path: str = "./data/vector_store", 
                 embedding_model: str = "text-embedding-ada-002",
                 chunk_size: int = 1000,
                 similarity_threshold: float = 0.7):
        self.vector_store_path = vector_store_path
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.similarity_threshold = similarity_threshold
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.1)
        
        # Load or initialize vector store
        try:
            self.vector_store = FAISS.load_local(vector_store_path, self.embeddings)
            logger.info("Loaded existing vector store")
        except:
            self.vector_store = None
            logger.warning("No existing vector store found")
    
    def process(self, agent_input: Dict[str, Any]) -> str:
        """Main processing method called by supervisor"""
        message = agent_input.get("message", "")
        context = agent_input.get("context", {})
        history = agent_input.get("history", [])
        
        if not self.vector_store:
            return "RAG system not initialized. Please add documents first."
        
        try:
            # Retrieve relevant documents
            docs = self.vector_store.similarity_search_with_score(
                message, k=5
            )
            
            # Filter by similarity threshold
            relevant_docs = [
                doc for doc, score in docs 
                if score >= self.similarity_threshold
            ]
            
            if not relevant_docs:
                return "No relevant documents found for your query."
            
            # Generate response using retrieved context
            context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            prompt = f"""Based on the following context, answer the user's question:

Context:
{context_text}

Question: {message}

Please provide a comprehensive answer based on the context provided. If the context doesn't fully answer the question, mention what information is missing."""

            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            logger.error(f"RAG processing error: {e}")
            return f"Error processing RAG query: {str(e)}"
    
    def add_documents(self, documents: List[str]) -> bool:
        """Add documents to the vector store"""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=200
            )
            
            docs = []
            for doc_text in documents:
                chunks = text_splitter.split_text(doc_text)
                docs.extend([Document(page_content=chunk) for chunk in chunks])
            
            if self.vector_store:
                self.vector_store.add_documents(docs)
            else:
                self.vector_store = FAISS.from_documents(docs, self.embeddings)
            
            # Save the vector store
            self.vector_store.save_local(self.vector_store_path)
            logger.info(f"Added {len(docs)} document chunks to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False

# agents/conversation_agent.py - Example Conversation Agent

class ConversationAgent:
    """Agent for handling conversational context and follow-ups"""
    
    def __init__(self, memory_window: int = 10, 
                 context_similarity_threshold: float = 0.8):
        self.memory_window = memory_window
        self.context_similarity_threshold = context_similarity_threshold
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.3)
        
    def process(self, agent_input: Dict[str, Any]) -> str:
        """Handle conversational queries with context"""
        message = agent_input.get("message", "")
        context = agent_input.get("context", {})
        history = agent_input.get("history", [])
        
        # Get recent conversation history
        recent_history = history[-self.memory_window:] if history else []
        
        if not recent_history:
            return "I don't have any conversation context to work with. Could you provide more details?"
        
        # Build context from history
        context_text = ""
        for item in recent_history:
            context_text += f"User: {item.get('user_message', '')}\n"
            context_text += f"Assistant: {item.get('agent_response', '')}\n\n"
        
        prompt = f"""You are handling a follow-up question in an ongoing conversation. 
        
Previous conversation:
{context_text}

Current question: {message}

Please provide a response that takes into account the conversation history and maintains continuity. If this seems like a clarification or follow-up to a previous topic, reference that context appropriately."""

        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Conversation processing error: {e}")
            return "I'm having trouble processing your follow-up question. Could you rephrase it?"

# agents/tool_agent.py - Example Tool Agent

import requests
import json
from datetime import datetime

class ToolAgent:
    """Agent for executing external tools and API calls"""
    
    def __init__(self, timeout: int = 30, max_retries: int = 3, rate_limit: int = 100):
        self.timeout = timeout
        self.max_retries = max_retries
        self.rate_limit = rate_limit
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
        
        # Available tools
        self.tools = {
            "web_search": self._web_search,
            "calculator": self._calculator,
            "weather": self._get_weather,
            "current_time": self._get_current_time
        }
    
    def process(self, agent_input: Dict[str, Any]) -> str:
        """Determine which tool to use and execute it"""
        message = agent_input.get("message", "")
        
        # Determine which tool to use
        tool_prompt = f"""Analyze this user request and determine which tool would be most appropriate:

Available tools:
- web_search: Search the internet for information
- calculator: Perform mathematical calculations
- weather: Get current weather information (requires location)
- current_time: Get the current date and time

User request: {message}

Respond with just the tool name, or "none" if no tool is needed."""

        try:
            tool_response = self.llm.invoke(tool_prompt)
            tool_name = tool_response.content.strip().lower()
            
            if tool_name in self.tools:
                return self.tools[tool_name](message)
            else:
                return "I couldn't determine an appropriate tool for your request."
                
        except Exception as e:
            logger.error(f"Tool agent error: {e}")
            return f"Error executing tool: {str(e)}"
    
    def _web_search(self, query: str) -> str:
        """Mock web search implementation"""
        return f"Web search results for '{query}': [This would contain actual search results in a real implementation]"
    
    def _calculator(self, expression: str) -> str:
        """Simple calculator tool"""
        try:
            # Extract mathematical expression from the query
            calc_prompt = f"""Extract the mathematical expression from this text and return only the expression to be calculated:
            
Text: {expression}

Return only the mathematical expression (e.g., "2+2" or "sqrt(16)")."""
            
            calc_response = self.llm.invoke(calc_prompt)
            expression = calc_response.content.strip()
            
            # Safe evaluation (in production, use a proper math parser)
            result = eval(expression)
            return f"The result of {expression} is {result}"
            
        except Exception as e:
            return f"Could not calculate the expression: {str(e)}"
    
    def _get_weather(self, query: str) -> str:
        """Mock weather API call"""
        return f"Weather information for query '{query}': [This would contain actual weather data in a real implementation]"
    
    def _get_current_time(self, query: str) -> str:
        """Get current time"""
        now = datetime.now()
        return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"

# agents/research_agent.py - Example Research Agent

class ResearchAgent:
    """Agent for comprehensive research tasks"""
    
    def __init__(self, max_sources: int = 10, research_depth: str = "comprehensive", 
                 include_citations: bool = True):
        self.max_sources = max_sources
        self.research_depth = research_depth
        self.include_citations = include_citations
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.1)
    
    def process(self, agent_input: Dict[str, Any]) -> str:
        """Conduct comprehensive research"""
        message = agent_input.get("message", "")
        
        # Create a research plan
        plan_prompt = f"""Create a research plan for this query: {message}

Provide:
1. Key topics to investigate
2. Types of sources needed
3. Specific questions to answer

Be comprehensive but focused."""

        try:
            plan_response = self.llm.invoke(plan_prompt)
            research_plan = plan_response.content
            
            # Simulate research process
            research_prompt = f"""Based on this research plan, provide a comprehensive response:

Research Plan:
{research_plan}

Original Query: {message}

Provide a thorough, well-structured response that addresses all aspects of the query. Include key findings, multiple perspectives where relevant, and conclude with actionable insights."""

            final_response = self.llm.invoke(research_prompt)
            
            if self.include_citations:
                return final_response.content + "\n\n[Note: In a real implementation, this would include actual citations and sources]"
            else:
                return final_response.content
                
        except Exception as e:
            logger.error(f"Research agent error: {e}")
            return f"Error conducting research: {str(e)}"
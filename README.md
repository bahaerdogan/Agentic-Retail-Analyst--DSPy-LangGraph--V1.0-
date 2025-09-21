# Agentic Retail Analyst (DSPy + LangGraph)

A local AI agent that answers retail analytics questions by combining RAG over documents and SQL queries on a Northwind database. This is an experimental implementation showcasing DSPy Chain-of-Thought reasoning with LangGraph orchestration.

## Current Implementation Status

### ✅ What's Working
- **Basic Question Routing**: Successfully classifies questions as RAG, SQL, or hybrid queries
- **Document Retrieval**: TF-IDF based search over markdown documents with chunking
- **SQL Generation**: Generates working SQL queries for common retail analytics questions
- **Database Integration**: Connects to SQLite Northwind database with proper error handling
- **Answer Synthesis**: Combines results from multiple sources with citations
- **CLI Interface**: Batch processing with configurable models via Ollama

### ⚠️ Known Limitations
- **Routing Accuracy**: Router sometimes misclassifies complex questions (~70% accuracy)
- **SQL Complexity**: Struggles with advanced joins and complex aggregations
- **Error Recovery**: Repair loop works but could be more sophisticated
- **Performance**: No query optimization or caching implemented yet
- **Model Dependency**: Heavily dependent on local model quality (tested with Phi-3.5)

### 🔧 Areas for Improvement
- Better prompt engineering for routing decisions
- More robust SQL query validation and repair
- Enhanced document chunking and retrieval scoring
- Query result caching and performance optimization
- Support for more complex analytical queries

## Architecture Overview
- **Router**: DSPy ChainOfThought classifier (rag | sql | hybrid)
- **Retriever**: TF-IDF document search with scoring
- **NL2SQL**: DSPy ChainOfThought SQL generation
- **Executor**: SQLite query execution with error handling
- **Synthesizer**: DSPy ChainOfThought answer formatting
- **Repair Loop**: Basic error detection and retry (2 iterations max)

## Trade-offs & Assumptions
- **CostOfGoods**: Approximated as 70% of UnitPrice when not available
- **Table Schema**: Uses lowercase table names for consistency
- **Local Models**: Designed for Ollama/local inference (no API costs)
- **Experimental**: This is a proof-of-concept, not production-ready

## Roadmap

**Next versions will be available in the following weeks** with improvements to:
- Enhanced routing accuracy and decision logic
- Better SQL query generation and validation
- Improved error handling and recovery mechanisms
- Performance optimizations and caching
- More comprehensive test coverage
- Support for additional data sources and formats
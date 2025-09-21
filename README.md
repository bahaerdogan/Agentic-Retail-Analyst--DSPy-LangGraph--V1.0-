Agentic Retail Analyst (DSPy + LangGraph)

Disclaimer: This project is a conceptual prototype. It is not part of my academic or professional work. This is a initial version updates will follow.

An experimental AI agent for retail analytics that combines retrieval-augmented generation (RAG) with SQL querying over the Northwind database. It showcases DSPy chain-of-thought reasoning orchestrated with LangGraph.

Current Status
Working Features

Question Routing: Classifies queries as RAG, SQL, or hybrid

Document Retrieval: TF-IDF search over chunked markdown documents

SQL Generation: Produces functional SQL for common analytics queries

Database Integration: Connects to SQLite Northwind with error handling

Answer Synthesis: Merges results from multiple sources with citations

CLI Interface: Batch processing with configurable Ollama models

Limitations

Routing accuracy ~70% on complex queries

Difficulty handling advanced joins and aggregations

Basic error recovery and repair mechanisms

No caching or query optimization

Strongly dependent on local model quality (tested with Phi-3.5)

Areas for Improvement

Improved prompt engineering for routing

More robust SQL validation and repair

Enhanced document chunking and retrieval scoring

Query caching and performance optimization

Support for more complex analytics

Architecture

Router: DSPy Chain-of-Thought classifier (rag | sql | hybrid)

Retriever: TF-IDF search with scoring

NL2SQL: DSPy SQL generator

Executor: SQLite with error handling

Synthesizer: DSPy answer formatter

Repair Loop: Basic error detection and retries (max 2)

Assumptions & Trade-offs

CostOfGoods estimated as 70% of UnitPrice if missing

Table schema standardized to lowercase

Local models preferred (Ollama, no API costs)

Proof-of-concept only, not production-ready
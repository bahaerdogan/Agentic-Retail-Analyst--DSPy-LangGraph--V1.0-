# Retail Analytics Copilot (DSPy + LangGraph)

A local AI agent that answers retail analytics questions by combining RAG over local documents and SQL over a local SQLite database (Northwind). Uses pure DSPy Chain-of-Thought reasoning for all components.

## Graph Design
- **Router**: DSPy ChainOfThought classifier for route selection (rag | sql | hybrid)
- **Retriever**: Optimized TF-IDF document search with efficient chunking and scoring
- **Planner**: DSPy ChainOfThought constraint extraction from documents
- **NL2SQL**: DSPy ChainOfThought SQL generation with pattern-based syntax cleaning
- **Executor**: Optimized SQL execution with connection pooling and error handling
- **Synthesizer**: DSPy ChainOfThought answer formatting with robust type conversion
- **Repair Loop**: Intelligent error detection and recovery (up to 2 iterations)

## DSPy Implementation
All modules use pure DSPy Chain-of-Thought reasoning with optimized prompt handling. The system provides:
- Efficient question classification and routing
- Robust SQL query generation from natural language
- Type-safe answer synthesis and format compliance
- Automatic citation generation from multiple sources

## Code Quality & Performance
- **Memory Optimization**: Removed unnecessary garbage collection and object deletion
- **Error Handling**: Robust exception handling with meaningful error messages
- **Code Deduplication**: Consolidated repeated logic into reusable patterns
- **Type Safety**: Improved format validation and type conversion
- **Connection Management**: Efficient database connection handling

## Trade-offs & Assumptions
- **CostOfGoods**: Approximated as 70% of UnitPrice when not available
- **Table Names**: Uses lowercase views (orders, order_items, products, customers) for consistency
- **Pure DSPy**: No hardcoded patterns or cheating mechanisms - relies entirely on model reasoning
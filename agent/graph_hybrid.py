import json
import dspy
import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Any, List, Dict, Optional
from agent.dspy_signatures import RouterModule, NL2SQLModule, SynthesizerModule, PlannerModule
from agent.rag.retrieval import Retriever
from agent.tools.sqlite_tool import SQLiteTool, SQLResult

logger = logging.getLogger(__name__)

class TraceLogger:
    def __init__(self, trace_file: str = "agent_trace.jsonl"):
        self.trace_file = trace_file
    
    def log_step(self, step: str, data: Dict[str, Any]):
        trace_entry = {"step": step, "timestamp": __import__('time').time(), **data}
        try:
            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

class AgentState(TypedDict):
    question: str
    format_hint: str
    route: str
    retrieved_docs: List[Dict[str, Any]]
    extracted_constraints: Dict[str, Any]
    sql: str
    sql_result: Optional[SQLResult]
    final_answer: Any
    confidence: float
    explanation: str
    citations: List[str]
    error: Optional[str]
    repair_count: int


class HybridAgent:
    def __init__(self, db_path: str, docs_path: str):
        self.db = SQLiteTool(db_path)
        self.retriever = Retriever(docs_path)
        self.router = RouterModule()
        self.planner = PlannerModule()
        self.nl2sql = NL2SQLModule()
        self.synthesizer = SynthesizerModule()
        self.checkpointer = MemorySaver()
        self.tracer = TraceLogger()
        self._schema_cache = None
        self._graph_cache = None
        


    def router_node(self, state: AgentState) -> AgentState:
        result = self.router(question=state["question"])
        state["route"] = getattr(result, 'route', 'hybrid')
        self.tracer.log_step("router", {
            "question": state["question"],
            "route_selection": state["route"]
        })
        return state

    def retriever_node(self, state: AgentState) -> AgentState:
        docs = self.retriever.search(state["question"], top_k=3)
        state["retrieved_docs"] = docs if docs else []
        self.tracer.log_step("retriever", {
            "question": state["question"],
            "received_snippets": [{"id": doc.get("id"), "score": doc.get("score")} for doc in state["retrieved_docs"]]
        })
        return state

    def planner_node(self, state: AgentState) -> AgentState:
        docs_text = "\n".join([f"{doc.get('id', '')}: {doc.get('content', '')}" for doc in state.get("retrieved_docs", [])])
        result = self.planner(question=state["question"], retrieved_docs=docs_text)
        constraints_str = getattr(result, 'constraints', '{}')
        try:
            state["extracted_constraints"] = json.loads(constraints_str)
        except:
            state["extracted_constraints"] = {}
        return state

    def nl2sql_node(self, state: AgentState) -> AgentState:
        if not self._schema_cache:
            self._schema_cache = self.db.schema()
        schema_text = "\n".join([f"Table {s['name']}: {s['sql'][:200]}" for s in self._schema_cache])
        constraints_text = json.dumps(state.get("extracted_constraints", {}))
        
        result = self.nl2sql(
            question=state["question"], 
            schema=schema_text, 
            constraints=constraints_text
        )
        state["sql"] = getattr(result, 'sql', 'SELECT 1')
        
        self.tracer.log_step("nl2sql", {
            "question": state["question"],
            "generated_sql": state["sql"]
        })
        return state

    def executor_node(self, state: AgentState) -> AgentState:
        sql_result = self.db.execute(state.get("sql", "SELECT 1"))
        state["sql_result"] = sql_result
        if sql_result.error:
            state["error"] = sql_result.error
        else:
            state["error"] = None
        self.tracer.log_step("executor", {
            "sql": state.get("sql", ""),
            "sql_error": state.get("error"),
            "rows_returned": len(state["sql_result"].rows) if state["sql_result"] else 0
        })
        return state

    def synthesizer_node(self, state: AgentState) -> AgentState:
        result = self.synthesizer(
            question=state["question"],
            format_hint=state.get("format_hint", "str"),
            sql_result=state.get("sql_result"),
            retrieved_docs=state.get("retrieved_docs", []),
            repair_count=state.get("repair_count", 0)
        )
        state["final_answer"] = getattr(result, 'final_answer', '')
        state["confidence"] = getattr(result, 'confidence', 0.5)
        state["explanation"] = getattr(result, 'explanation', '')
        state["citations"] = getattr(result, 'citations', [])
        self.tracer.log_step("synthesizer", {
            "format_hint": state.get("format_hint", "str"),
            "final_output": state["final_answer"],
            "confidence": state["confidence"],
            "citations": state["citations"]
        })
        return state

    def repair_node(self, state: AgentState) -> AgentState:
        state["repair_count"] = state.get("repair_count", 0) + 1
        
        if state["repair_count"] > 2:
            state["confidence"] = max(0.1, state.get("confidence", 0.5) * 0.3)
            state["error"] = None
            return state
        
        sql_error = self._detect_sql_error(state)
        format_error = self._detect_format_error(state)
        
        if not sql_error and not format_error:
            state["error"] = None
            state["confidence"] = max(0.1, state.get("confidence", 0.5) * (0.9 ** state["repair_count"]))
        
        return state

    def _validate_format(self, answer: Any, format_hint: str) -> bool:
        validators = {
            "int": lambda x: isinstance(x, int) or (isinstance(x, (str, float)) and str(x).replace('.', '').replace('-', '').isdigit()),
            "float": lambda x: isinstance(x, (int, float)) or (isinstance(x, str) and self._is_float(x)),
            "list": lambda x: isinstance(x, list),
            "object": lambda x: isinstance(x, dict)
        }
        
        if format_hint in validators:
            return validators[format_hint](answer)
        elif format_hint.startswith("list[") or format_hint == "list":
            return isinstance(answer, list)
        elif format_hint.startswith("{") or format_hint == "object":
            return isinstance(answer, dict)
        
        return True
    
    def _is_float(self, value: str) -> bool:
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def _detect_sql_error(self, state: AgentState) -> Optional[str]:
        sql_result = state.get("sql_result")
        if not sql_result or not sql_result.error:
            return None
        
        error_lower = sql_result.error.lower()
        error_types = {
            "syntax error": "sql_syntax_error",
            "no such table": "sql_table_error", 
            "no such column": "sql_column_error"
        }
        
        for error_text, error_type in error_types.items():
            if error_text in error_lower:
                return f"{error_type}: {sql_result.error}"
        
        return f"sql_execution_error: {sql_result.error}"

    def _detect_format_error(self, state: AgentState) -> Optional[str]:
        final_answer = state.get("final_answer")
        format_hint = state.get("format_hint", "str")
        if not self._validate_format(final_answer, format_hint):
            return f"format_validation_failed: expected {format_hint}, got {type(final_answer).__name__}"
        return None

    def should_retrieve(self, state: AgentState) -> str:
        return "retriever" if state.get("route") in ["rag", "hybrid"] else "planner"

    def should_plan(self, state: AgentState) -> str:
        return "planner" if state.get("route") in ["sql", "hybrid"] else "synthesizer"

    def should_execute_sql(self, state: AgentState) -> str:
        return "nl2sql" if state.get("route") in ["sql", "hybrid"] else "synthesizer"

    def should_repair(self, state: AgentState) -> str:
        repair_count = state.get("repair_count", 0)
        if repair_count >= 2:
            return END
        sql_error = self._detect_sql_error(state)
        format_error = self._detect_format_error(state)
        if sql_error or format_error:
            return "repair"
        return END

    def after_repair(self, state: AgentState) -> str:
        error = state.get("error", "")
        
        if error.startswith("sql_"):
            return "nl2sql"
        elif error.startswith("format_") or error.startswith("missing_"):
            return "synthesizer"
        else:
            return "synthesizer"

    def build_graph(self) -> StateGraph:
        if self._graph_cache is not None:
            return self._graph_cache
        workflow = StateGraph(AgentState)
        workflow.add_node("router", self.router_node)
        workflow.add_node("retriever", self.retriever_node)
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("nl2sql", self.nl2sql_node)
        workflow.add_node("executor", self.executor_node)
        workflow.add_node("synthesizer", self.synthesizer_node)
        workflow.add_node("repair", self.repair_node)
        workflow.set_entry_point("router")
        workflow.add_conditional_edges("router", self.should_retrieve, {
            "retriever": "retriever",
            "planner": "planner"
        })
        workflow.add_conditional_edges("retriever", self.should_plan, {
            "planner": "planner",
            "synthesizer": "synthesizer"
        })
        workflow.add_conditional_edges("planner", self.should_execute_sql, {
            "nl2sql": "nl2sql",
            "synthesizer": "synthesizer"
        })
        workflow.add_edge("nl2sql", "executor")
        workflow.add_edge("executor", "synthesizer")
        workflow.add_conditional_edges("synthesizer", self.should_repair, {
            "repair": "repair",
            END: END
        })
        workflow.add_conditional_edges("repair", self.after_repair, {
            "nl2sql": "nl2sql",
            "synthesizer": "synthesizer"
        })
        self._graph_cache = workflow.compile(checkpointer=self.checkpointer)
        return self._graph_cache


def create_agent(db_path: str, docs_path: str) -> HybridAgent:
    return HybridAgent(db_path, docs_path)

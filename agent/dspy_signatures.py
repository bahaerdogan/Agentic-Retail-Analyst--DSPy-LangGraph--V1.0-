import json
import dspy
import re
from typing import List, Any, Dict, Optional

class RouterSignature(dspy.Signature):
    question: str = dspy.InputField(desc="User question to classify")
    route: str = dspy.OutputField(desc="Route: 'rag' for policy/definitions, 'sql' for data queries, 'hybrid' for both")

class NL2SQLSignature(dspy.Signature):
    question: str = dspy.InputField(desc="Natural language question")
    schema: str = dspy.InputField(desc="Database schema")
    constraints: str = dspy.InputField(desc="Date/category constraints")
    sql: str = dspy.OutputField(desc="Valid SQLite query")

class SynthesizerSignature(dspy.Signature):
    question: str = dspy.InputField(desc="Original question")
    format_hint: str = dspy.InputField(desc="Required format: int, float, list, dict")
    data: str = dspy.InputField(desc="SQL results or document content")
    final_answer: str = dspy.OutputField(desc="Answer matching format_hint")
    explanation: str = dspy.OutputField(desc="Brief explanation (≤2 sentences)")
    citations: str = dspy.OutputField(desc="JSON list of sources used")

class PlannerSignature(dspy.Signature):
    question: str = dspy.InputField(desc="User question")
    docs: str = dspy.InputField(desc="Retrieved documents")
    constraints: str = dspy.OutputField(desc="JSON constraints object")

class RouterModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(RouterSignature)
    
    def forward(self, question: str) -> dspy.Prediction:
        result = self.classify(question=question)
        route = getattr(result, 'route', 'hybrid').lower().strip()
        
        if route not in ['rag', 'sql', 'hybrid']:
            route = 'hybrid'
        
        return dspy.Prediction(route=route)

class NL2SQLModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(NL2SQLSignature)
    
    def forward(self, question: str, schema: str, constraints: str = "") -> dspy.Prediction:
        result = self.generate(
            question=question,
            schema=schema,
            constraints=constraints
        )
        
        sql = getattr(result, 'sql', '').strip()
        if sql:
            sql = self._clean_sql(sql)
        
        return dspy.Prediction(sql=sql or "SELECT 1 as error")
    
    def _clean_sql(self, sql: str) -> str:
        replacements = {
            r'"Order Details"': 'order_items',
            r'Order_Items': 'order_items', 
            r'CustomerName': 'CompanyName'
        }
        for pattern, replacement in replacements.items():
            sql = re.sub(pattern, replacement, sql)
        return sql
class SynthesizerModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.synthesize = dspy.ChainOfThought(SynthesizerSignature)
    
    def forward(self, question: str, format_hint: str, sql_result: Any = None, retrieved_docs: List[Dict] = None, repair_count: int = 0) -> dspy.Prediction:
        data_str = ""
        if sql_result and hasattr(sql_result, 'rows') and sql_result.rows:
            data_str = json.dumps({
                'columns': sql_result.columns,
                'rows': sql_result.rows[:5]
            })
        elif retrieved_docs:
            data_str = "\n".join([f"{doc.get('id', '')}: {doc.get('content', '')}" for doc in retrieved_docs])
        
        result = self.synthesize(
            question=question,
            format_hint=format_hint,
            data=data_str
        )
        
        final_answer = getattr(result, 'final_answer', self._get_format_default(format_hint))
        explanation = getattr(result, 'explanation', 'Generated from available data')[:200]
        citations_str = getattr(result, 'citations', '[]')
        
        try:
            citations = json.loads(citations_str) if citations_str else []
        except:
            citations = self._generate_basic_citations(sql_result, retrieved_docs)
        
        confidence = self._calculate_confidence(sql_result, retrieved_docs, repair_count)
        
        return dspy.Prediction(
            final_answer=self._format_answer(final_answer, format_hint),
            confidence=confidence,
            explanation=explanation,
            citations=citations
        )
    
    def _generate_basic_citations(self, sql_result: Any, retrieved_docs: List[Dict]) -> List[str]:
        citations = []
        
        if retrieved_docs:
            for doc in retrieved_docs:
                doc_id = doc.get('id', '')
                if doc_id:
                    citations.append(doc_id)
        
        if sql_result and hasattr(sql_result, 'rows') and sql_result.rows:
            citations.extend(["Orders", "Order Details", "Products", "Customers"])
        
        return list(set(citations))
    
    def _calculate_confidence(self, sql_result: Any, retrieved_docs: List[Dict], repair_count: int) -> float:
        confidence = 0.5
        
        if sql_result and hasattr(sql_result, 'rows') and sql_result.rows:
            confidence += 0.3
        if retrieved_docs:
            confidence += 0.2
        
        confidence *= (0.9 ** repair_count)
        return max(0.1, min(1.0, confidence))
    
    def _get_format_default(self, format_hint: str) -> Any:
        defaults = {
            "int": 0,
            "float": 0.0,
            "list": [],
            "object": {}
        }
        
        if format_hint in defaults:
            return defaults[format_hint]
        elif format_hint.startswith("list["):
            return []
        elif format_hint.startswith("{"):
            return {}
        
        return "Error"
    
    def _format_answer(self, answer: str, format_hint: str) -> Any:
        try:
            answer_str = str(answer).strip()
            
            formatters = {
                "int": lambda x: int(float(x)),
                "float": lambda x: round(float(x), 2)
            }
            
            if format_hint in formatters:
                return formatters[format_hint](answer_str)
            elif format_hint.startswith("list[") or format_hint == "list":
                if isinstance(answer, list):
                    return answer
                try:
                    return json.loads(answer_str)
                except:
                    return [answer_str]
            elif format_hint.startswith("{") or format_hint == "object":
                if isinstance(answer, dict):
                    return answer
                try:
                    return json.loads(answer_str)
                except:
                    return {"result": answer_str}
            
            return answer_str
        except:
            return self._get_format_default(format_hint)

class PlannerModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(PlannerSignature)
    
    def forward(self, question: str, retrieved_docs: str) -> dspy.Prediction:
        result = self.extract(question=question, docs=retrieved_docs)
        constraints_str = getattr(result, 'constraints', '{}')
        
        try:
            constraints = json.loads(constraints_str)
        except:
            constraints = {}
        
        return dspy.Prediction(constraints=json.dumps(constraints))

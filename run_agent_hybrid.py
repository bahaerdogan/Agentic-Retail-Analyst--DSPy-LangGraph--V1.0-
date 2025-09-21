import os
import json
import click
import dspy
import logging
from agent.graph_hybrid import create_agent
from typing import Any, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_dspy(model: str = "phi3.5:3.8b-mini-instruct-q4_K_M", api_base: str = "http://localhost:11434", api_key: str = "") -> bool:
    candidates = [f"ollama/{model}", model, f"ollama_chat/{model}"]
    for cand in candidates:
        try:
            lm = dspy.LM(cand, api_base=api_base, api_key=api_key, temperature=0.2, top_p=0.9)
            dspy.configure(lm=lm)
            return True
        except Exception:
            continue
    return False

def format_answer_for_output(answer: Any, format_hint: str) -> Any:
    formatters = {
        "int": lambda x: int(float(str(x))),
        "float": lambda x: round(float(str(x)), 2)
    }
    
    try:
        if format_hint in formatters:
            return formatters[format_hint](answer)
        elif format_hint.startswith("list[") or format_hint == "list":
            if isinstance(answer, list):
                return answer
            return json.loads(str(answer)) if isinstance(answer, str) else [answer]
        elif format_hint.startswith("{") or format_hint == "object":
            if isinstance(answer, dict):
                return answer
            return json.loads(str(answer)) if isinstance(answer, str) else {"result": answer}
        return answer
    except:
        return get_fallback_answer(format_hint)

def process_question(agent, item: Dict) -> Dict:
    question_id = item.get("id", "")
    question = item.get("question", "")
    format_hint = item.get("format_hint", "str")
    
    initial_state = {
        "question": question,
        "format_hint": format_hint,
        "route": "",
        "retrieved_docs": [],
        "extracted_constraints": {},
        "sql": "",
        "sql_result": None,
        "final_answer": None,
        "confidence": 0.0,
        "explanation": "",
        "citations": [],
        "error": None,
        "repair_count": 0
    }
    
    try:
        graph = agent.build_graph()
        config = {"configurable": {"thread_id": f"thread_{question_id}"}}
        result_state = graph.invoke(initial_state, config=config)
        
        final_answer = format_answer_for_output(result_state.get("final_answer", ""), format_hint)
        confidence = result_state.get("confidence", 0.5)
        explanation = result_state.get("explanation", "")
        if len(explanation) > 200:
            sentences = explanation.split(". ")
            explanation = ". ".join(sentences[:2])
            if not explanation.endswith("."):
                explanation += "."
        citations = result_state.get("citations", [])
        
        logger.info(f"Processed {question_id}")
        
        return {
            "id": question_id,
            "final_answer": final_answer,
            "sql": result_state.get("sql", ""),
            "confidence": round(confidence, 2),
            "explanation": explanation,
            "citations": citations
        }
        
    except Exception as e:
        logger.error(f"Error processing {question_id}: {e}")
        return {
            "id": question_id,
            "final_answer": get_fallback_answer(format_hint),
            "sql": "",
            "confidence": 0.1,
            "explanation": f"Error processing question: {str(e)[:100]}",
            "citations": []
        }

def get_fallback_answer(format_hint: str) -> Any:
    fallbacks = {
        "int": 0,
        "float": 0.0,
        "list": [],
        "object": {}
    }
    
    if format_hint in fallbacks:
        return fallbacks[format_hint]
    elif format_hint.startswith("list["):
        return []
    elif format_hint.startswith("{"):
        return {}
    
    return "Error: Unable to generate answer"

@click.command()
@click.option("--batch", type=click.Path(exists=True), required=True)
@click.option("--out", type=click.Path(), required=True)
@click.option("--db", type=click.Path(exists=True), default="data/northwind.sqlite")
@click.option("--docs", type=click.Path(exists=True), default="docs")
@click.option("--model", default="phi3.5:3.8b-mini-instruct-q4_K_M")
@click.option("--api_base", default="http://localhost:11434")
@click.option("--api_key", default="")
def run(batch, out, db, docs, model, api_base, api_key):
    logger.info(f"Starting batch processing with model: {model}")
    
    if not setup_dspy(model=model, api_base=api_base, api_key=api_key):
        raise SystemExit("Failed to setup DSPy with the specified model")
    
    agent = create_agent(db, docs)
    results = []
    
    try:
        with open(batch, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    result = process_question(agent, item)
                    results.append(result)
                    print(f"Processed question {line_num}: {item.get('id', 'unknown')}")
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")
                    continue
    except FileNotFoundError:
        raise SystemExit(f"Input file not found: {batch}")
    except Exception as e:
        raise SystemExit(f"Error reading input file: {e}")
    
    try:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as fw:
            for result in results:
                fw.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"Successfully processed {len(results)} questions")
        print(f"Results written to: {out}")
    except Exception as e:
        raise SystemExit(f"Error writing output file: {e}")

if __name__ == "__main__":
    run()

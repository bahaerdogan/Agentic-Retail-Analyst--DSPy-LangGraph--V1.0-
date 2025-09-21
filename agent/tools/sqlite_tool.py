import sqlite3
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class SQLResult:
    columns: List[str]
    rows: List[Tuple[Any, ...]]
    error: Optional[str] = None

class SQLiteTool:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._schema_cache = None

    @contextmanager
    def _get_connection(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=5000")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass


    def execute(self, query: str, params: Tuple = ()) -> SQLResult:
        if not query or not query.strip():
            return SQLResult(columns=[], rows=[], error="empty_query")
        
        query = query.strip()
        if len(query) > 5000:
            return SQLResult(columns=[], rows=[], error="query_too_long")
        
        try:
            with self._get_connection() as conn:
                cur = conn.execute(query, params)
                rows = cur.fetchmany(500)
                
                if rows:
                    columns = list(rows[0].keys())
                    rows_tuples = [tuple(row) for row in rows]
                else:
                    columns = [d[0] for d in cur.description] if cur.description else []
                    rows_tuples = []
                
                return SQLResult(columns=columns, rows=rows_tuples, error=None)
        except Exception as e:
            return SQLResult(columns=[], rows=[], error=str(e)[:200])

    def schema(self) -> List[Dict[str, str]]:
        if self._schema_cache is not None:
            return self._schema_cache
        
        try:
            with self._get_connection() as conn:
                cur = conn.execute("SELECT name, sql FROM sqlite_master WHERE type IN ('table','view') ORDER BY name")
                all_objects = [{"name": row["name"], "sql": row["sql"] or ""} for row in cur.fetchall()]
                

                relevant_tables = {
                    "Orders", "Products", "Customers", "Categories", "Suppliers", 
                    "Employees", "Shippers", "Regions", "Territories",
                    "orders", "products", "customers", "order_items"
                }
                
                main_tables = []
                for obj in all_objects:
                    name = obj["name"]
                    if name in relevant_tables:
                        if name == "order_items":
                            obj["sql"] = "CREATE VIEW order_items AS SELECT OrderID, ProductID, UnitPrice, Quantity, Discount FROM [Order Details]"
                        main_tables.append(obj)
                
                self._schema_cache = main_tables
                return self._schema_cache
        except Exception as e:
            logger.error(f"Schema retrieval failed: {e}")
            return []



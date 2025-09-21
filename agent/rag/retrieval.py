import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, docs_path: str, chunk_size: int = 400):
        self.docs = self._load_docs(docs_path, chunk_size)
        if not self.docs:
            raise ValueError(f"No text files found in {docs_path}")
        self.vectorizer = TfidfVectorizer(
            stop_words='english', 
            norm="l2", 
            max_features=2000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            dtype=np.float32
        )
        doc_contents = [d["content"] for d in self.docs]
        self.tfidf_matrix = self.vectorizer.fit_transform(doc_contents).astype(np.float32)


    def _load_docs(self, docs_path, chunk_size=500):
        docs = []
        for fname in sorted(os.listdir(docs_path)):
            path = os.path.join(docs_path, fname)
            if not os.path.isfile(path) or not fname.lower().endswith((".txt", ".md")):
                continue
            
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            chunk_idx = 0
            chunk = ""
            
            for p in paragraphs:
                if len(chunk) + len(p) + 1 <= chunk_size:
                    chunk = f"{chunk} {p}" if chunk else p
                else:
                    if chunk:
                        docs.append({
                            "id": f"{fname}::chunk{chunk_idx}",
                            "content": chunk.strip(),
                            "source": fname
                        })
                        chunk_idx += 1
                    chunk = p
            
            if chunk:
                docs.append({
                    "id": f"{fname}::chunk{chunk_idx}",
                    "content": chunk.strip(),
                    "source": fname
                })
        return docs

    def search(self, query: str, top_k: int = 3):
        if not query.strip():
            return []
        
        try:
            query_vector = self.vectorizer.transform([query]).astype(np.float32)
            scores = cosine_similarity(self.tfidf_matrix, query_vector).ravel()
            
            if len(scores) <= top_k:
                top_indices = np.argsort(scores)[::-1]
            else:
                top_indices = np.argpartition(scores, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            results = []
            for idx in top_indices:
                if scores[idx] > 0.01:
                    results.append({
                        "id": self.docs[idx]["id"],
                        "content": self.docs[idx]["content"][:500],
                        "score": float(scores[idx])
                    })
            
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

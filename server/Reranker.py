from flashrank import Ranker, RerankRequest
import os
class Reranker:
    def __init__(self):
        self.reranker = Ranker(os.getenv("rerank_model"), cache_dir="flashrank")

    def rerank_documents(self, documents, prompt):
        if len(documents) == 0:
            return []
        
        # Create passages in the format expected by Flashrank
        passages = [
            {
                "id": i,
                "text": doc["content"],
                "metadata": doc["metadata"]
            }
            for (i, doc) in enumerate(documents)
        ]
        
        #Create a RerankRequest
        rerank_request = RerankRequest(query=prompt, passages=passages)
        
        # Rerank using Flashrank
        rerank_results = self.reranker.rerank(rerank_request)
        
        # Sort by score (higher is better)
        rerank_results.sort(key=lambda x: x['score'], reverse=True)
        # Rename the text field back to content
        for result in rerank_results:
            result['content'] = result['text']
            result['score'] = float(result['score'])
            del result['text']
        
        return rerank_results

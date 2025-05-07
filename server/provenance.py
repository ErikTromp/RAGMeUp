import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def compute_rerank_provenance(reranker, query, documents, answer):
    if os.getenv("attribute_include_query") == "True":
        full_text = query + "\n" + answer
    else:
        full_text = answer
    
    # Score the documents, this will return the same document list but now with a relevance_score in metadata
    scored_documents = reranker.compress_documents(documents, full_text)
    return scored_documents

def compute_llm_provenance(llm, query, context, answer):
    prompt = os.getenv("provenance_llm_prompt")
    # Go over all documents in the context
    provenance_scores = []
    for doc in context:
        # Create the thread to ask the LLM to assign a score to this document for provenance
        new_doc = doc
        new_doc.page_content = new_doc['content'].replace("{", "{{").replace("}", "}}")
        input_chat = [('human', prompt.format_map({"query": query, "context": new_doc, "answer": answer}))]
        (response, _) = llm.generate_response(None, prompt, [])
        score = json.loads(response)['score']
        provenance_scores.append(score)
    
    return provenance_scores

class DocumentSimilarityAttribution:
    def __init__(self):
        device = 'cuda'
        if os.getenv('force_cpu') == "True":
            device = 'cpu'
        self.model = SentenceTransformer(os.getenv('provenance_similarity_llm'), device=device)

    def compute_similarity(self, query, context, answer):
        include_query=True
        if os.getenv("attribute_include_query") == "False":
            include_query=False
        # Encode the answer, query, and context documents
        answer_embedding = self.model.encode([answer])[0]
        context_embeddings = self.model.encode(context)
        
        if include_query:
            query_embedding = self.model.encode([query])[0]

        # Compute similarity scores
        similarity_scores = []
        for i, doc_embedding in enumerate(context_embeddings):
            # Similarity between document and answer
            doc_answer_similarity = cosine_similarity([doc_embedding], [answer_embedding])[0][0]
            
            if include_query:
                # Similarity between document and query
                doc_query_similarity = cosine_similarity([doc_embedding], [query_embedding])[0][0]
                # Average of answer and query similarities
                similarity_score = (doc_answer_similarity + doc_query_similarity) / 2
            else:
                similarity_score = doc_answer_similarity

            similarity_scores.append(similarity_score)

        # Normalize scores
        total_similarity = sum(similarity_scores)
        normalized_scores = [score / total_similarity for score in similarity_scores] if total_similarity > 0 else similarity_scores

        return normalized_scores
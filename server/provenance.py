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
    scored_documents = reranker.rerank_documents(documents, full_text)
    return scored_documents

def compute_llm_provenance(llm, query, context, answer):
    prompt = os.getenv("provenance_llm_prompt")
    # Go over all documents in the context
    provenance_scores = []
    for doc in context:
        # Create the thread to ask the LLM to assign a score to this document for provenance
        new_doc = doc
        new_doc['content'] = new_doc['content'].replace("{", "{{").replace("}", "}}")
        if os.getenv("attribute_include_query") == "False":
            input_chat = prompt.format_map({"query": f"The user asked {query}" , "context": new_doc, "answer": answer})
        else:
            input_chat = prompt.format_map({"query": "", "context": new_doc, "answer": answer})
        (response, _) = llm.generate_response(None, input_chat, [])
        score = response
        provenance_scores.append(score)
    
    return [{"score": score} for score in provenance_scores]

class DocumentSimilarityAttribution:
    def __init__(self):
        device = 'cuda'
        if os.getenv('embedding_cpu') == "True":
            device = 'cpu'
        self.model = SentenceTransformer(os.getenv('provenance_similarity_llm'), device=device)

    def compute_similarity(self, query, context, answer):
        include_query=True
        if os.getenv("attribute_include_query") == "False":
            include_query=False
            
        # Encode the answer, query, and context documents
        answer_embedding = self.model.encode([answer])[0]
        
        # Extract text content from context documents if they're dictionaries
        context_texts = [doc['content'] for doc in context]
        context_embeddings = self.model.encode(context_texts)
        
        if include_query:
            query_embedding = self.model.encode([query])[0]

        # Compute similarity scores
        similarity_scores = []
        for i, doc_embedding in enumerate(context_embeddings):
            # Similarity between document and answer
            doc_answer_similarity = cosine_similarity(
                answer_embedding.reshape(1, -1), 
                answer_embedding.reshape(1, -1)
            )[0][0]
            
            if include_query:
                # Similarity between document and query
                doc_query_similarity = cosine_similarity(
                    doc_embedding.reshape(1, -1), 
                    query_embedding.reshape(1, -1)
                )[0][0]
                # Average of answer and query similarities
                similarity_score = (doc_answer_similarity + doc_query_similarity) / 2
            else:
                similarity_score = doc_answer_similarity

            similarity_scores.append(similarity_score)

        return [{"score": score} for score in similarity_scores]
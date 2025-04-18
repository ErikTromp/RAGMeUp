# RAG Me Up
RAG Me Up is a generic framework (server + UIs) that enables you do to RAG on your own dataset **easily**. Its essence is a small and lightweight server and a couple of ways to run UIs to communicate with the server (or write your own).

RAG Me Up can run on CPU but is best run on any GPU with at least 16GB of vRAM when using the default instruct model.

Combine the power of RAG with the power of fine-tuning - check out our [LLaMa2Lang repository](https://github.com/UnderstandLingBV/LLaMa2Lang) on fine-tuning LLMs which can then be used in RAG Me Up.

# Updates
- **2025-04-18** Updateed Postgres hybrid search
- **2025-02-03** Evaluation frameworks added: RAGAS, DeepEval
- **2024-09-06** Implemented [Re2](https://arxiv.org/abs/2309.06275)
- **2024-09-04** Added an evaluation script that uses Ragas to evaluate your RAG pipeline
- **2024-08-30** Added Ollama compatibility
- **2024-08-27** Using cross encoders now so you can specify your own reranking model
- **2024-07-30** Added multiple provenance attribution methods
- **2024-06-26** Updated readme, added more file types, robust self-inflection
- **2024-06-05** Upgraded to Langchain v0.2

# Installation
## Server
```
git clone https://github.com/FutureClubNL/RAGMeUp.git
cd server
pip install -r requirements.txt
```
Then run the server using `python server.py` from the server subfolder.

## Scala UI
Make sure you have JDK 17+. Download and install [SBT](https://www.scala-sbt.org/) and run `sbt run` from the `server/scala` directory or alternatively download the [compiled binary](https://github.com/FutureClubNL/RAGMeUp/releases/tag/scala-ui) and run `bin/ragemup(.bat)`

## Using Postgres (adviced for production)
RAG Me Up supports Postgres as hybrid retrieval database with both pgvector and pg_search installed. To run Postgres instead of Milvus, follow these steps.

- In the postgres folder is a Dockerfile, build it using `docker build -t ragmeup-pgvector-pgsearch .`
- Run the container using `docker run --name ragmeup-pgvector-pgsearch -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d ragmeup-pgvector-pgsearch`
- Once in use, our custom PostgresBM25Retriever will automatically create the right indexes for you.
- pgvector however, will not do this automatically so you have to create them yourself (perhaps after loading the documents first so the right tables are created):
    - Make sure the vector column is an actual vector (it's not by default): `ALTER TABLE langchain_pg_embedding ALTER COLUMN embedding TYPE vector(384);`
    - Create the index (may take a while with a lot of data): `CREATE INDEX ON langchain_pg_embedding USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);`

# How does RAG Me Up work?
RAG Me Up aims to provide a robust RAG pipeline that is configurable without necessarily writing any code. To achieve this, a couple of strategies are used to make sure that the user query can be accurately answered through the documents provided.

The RAG pipeline is visualized in the image below:
![RAG pipeline drawing](./ragmeup.drawio.svg)

The following steps are executed. Take note that some steps are optional and can be turned off through configuring the `.env` file.

__Top part - Indexing__
1. You collect and make your documents available to RAG Me Up.
2. Using different file type loaders, RAG Me Up will read the contents of your documents. Note that for some document types like JSON and XML, you need to specify additional configuration to tell RAG Me Up what to extract.
3. Your documents get chunked using a recursive splitter.
4. The chunks get converted into document (chunk) embeddings using an embedding model. Note that this model is usually a different one than the LLM you intend to use for chat.
5. RAG Me Up uses a hybrid search strategy, combining dense vectors in the vector database with sparse vectors using BM25. By default, RAG Me Up uses a local [Milvus database](https://milvus.io/).

__Bottom part - Inference__
1. Inference starts with a user asking a query. This query can either be an initial query or a follow-up query with an associated history and documents retrieved before. Note that both (chat history, documents) need to be passed on by a UI to properly handle follow-up querying.
2. A check is done if new documents need to be fetched, this can be due to one of two cases:
    - There is no history given in which case we always need to fetch documents
    - **[OPTIONAL]** The LLM itself will judge whether or not the question - in isolation - is phrased in such a way that new documents are fetched or whether it is a follow-up question on existing documents. A flag called `fetch_new_documents` is set to indicate whether or not new documents need to be fetched.
3. Documents are fetched from both the vector database (dense) and the BM25 index (sparse). Only executed if `fetch_new_documents` is set.
4. **[OPTIONAL]** Reranking is applied to extract the most relevant documents returned by the previous step. Only executed if `fetch_new_documents` is set.
5. **[OPTIONAL]** The LLM is asked to judge whether or not the documents retrieved contain an accurate answer to the user's query. Only executed if `fetch_new_documents` is set.
    - If this is not the case, the LLM is used to rewrite the query with the instruction to optimize for distance based similarity search. This is then fed back into step 3. **but only once** to avoid lengthy or infinite loops.
6. The documents are injected into the prompt with the user query. The documents can come from:
    - The retrieval and reranking of the document databases, if `fetch_new_documents` is set.
    - The history passed on with the initial user query, if `fetch_new_documents` is **not** set.
7. The LLM is asked to answer the query with the given chat history and documents.
8. The answer, chat history and documents are returned.

# Configuration
RAG Me Up uses a `.env` file for configuration, see `.env.template`. The following fields can be configured:

## LLM configuration
- `llm_model` This is the main LLM (instruct or chat) model to use that you will converse with. Default is LLaMa3-8B
- `llm_assistant_token` This should contain the unique query (sub)string that indicates where in a prompt template the assistant's answer starts
- `embedding_model` The model used to convert your documents' chunks into vectors that will be stored in the vector store
- `trust_remote_code` Set this to true if your LLM needs to execute remote code
- `force_cpu` When set to True, forces RAG Me Up to run fully on CPU (not recommended)

### Use OpenAI
If you want to use OpenAI as LLM backend, make sure to set `use_openai` to True and make sure you (externally) set the environment variable `OPENAI_API_KEY` to be your OpenAI API Key.

### Use Gemini
If you want to use Gemini as LLM backend, make sure to set `use_gemini` to True and make sure you (externally) set the environment variable `GOOGLE_API_KEY` to be your Gemini API Key.

### Use Azure OpenAI
If you want to use Azure OpenAI as LLM backend, make sure to set `use_azure` to True and make sure you (externally) set the following environment variables:
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME`

## Use Ollama
If you want to use Ollama as LLM backend, make sure to install Ollama and set `use_ollama` to True. The model to use should be given in `ollama_model`.

## RAG Provenance
One of the biggest, arguably unsolved, challenges of RAG is to do good provenance attribution: tracking which of the source documents retrieved from your database led to the LLM generating its answer (the most). RAG Me Up implements several ways of achieving this, each with its own pros and cons.

The following environment variables can be set for provenance attribution.

- `provenance_method` Can be one of `rerank, attention, similarity, llm`. If `rerank` is `False` and the value of `provenance_method` is either `rerank` or none of the allowed values, provenance attribution is turned completely off
- `provenance_similarity_llm` If `provenance_method` is set to `similarity`, this model will be used to compute the similarity scores
- `provenance_include_query` Set to True or False to include the query itself when attributing provenance
- `provenance_llm_prompt` If `provenance_method` is set to `llm`, this prompt will be used to let the LLM attribute the provenance of each document in isolation.

The different provenance attribution metrics are described below.

### `provenance_method=rerank` (preferred for closed LLMs)
This uses the reranker as the provenance method. While the reranking is already used when retrieving documents (if reranking is turned on), this only applies the rerankers cross-attention to the documents and the *query*. For provenance attribution, we use the same reranking to apply cross-attention to the *answer* (and potentially the query too).

### `provenance_method=attention` (preferred for OS LLMs)
This is probably the most accurate way of tracking provenance but it can only be used with OS LLMs that allow to return the attention weights. The way we track provenance is by looking at the actual attention weights (of the last attention layer in the model) for each token from the answer to the document and vice versa, optionally we do the same for the query if `provenance_include_query=True`.

### `provenance_method=similarity`
This method uses a sentence transformer (LM) to get dense vectors for each document as well as for the answer (and potentially query). We then use a cosine similarity to get the similarity of the document vectors to the answer (+ query).

### `provenance_method=llm`
The LLM that is used to generate messages is now also used to attribute the provenance of each document in isolation. We use the `provenance_llm_prompt` as the prompt to ask the LLM to perform this task. Note that the outcome of this provenance method is highly influenced by the prompt and the strength of the model. As a good practice, make sure you force the LLM to return numbers on a relatively small scale (eg. score from 1 to 3). Using something like a percentage for each document will likely result in random outcomes.

## Data configuration
- `data_directory` The directory that contains your (initial) documents to load into the vector store
- `file_types` Comma-separated list of file types to load. Supported file types: `PDF, JSON, DOCX, XSLX, PPTX, CSV, XML`
- `json_schema` If you are loading JSON, this should be the schema (using `jq_schema`). For example, use `.` for the root of a JSON object if your data contains JSON objects only and `.[0]` for the first element in each JSON array if your data contains JSON arrays with one JSON object in them
- `json_text_content` Whether or not the JSON data should be loaded as textual content or as structured content (in case of a JSON object)
- `xml_xpath` If you are loading XML, this should be the XPath of the documents to load (the tags that contain your text)

## Retrieval configuration
- `vector_store_uri` RAG Me Up caches your vector store on disk if possible to make loading a next time faster. This is the location where the vector store is stored. Remove this file to force a reload of all your documents
- `vector_store_k` The number of documents to retrieve from the vector store
- `rerank` Set to either True or False to enable reranking
- `rerank_k` The number of documents to keep after reranking. Note that if you use reranking, this should be your final target for `k` and `vector_store_k` should be set (significantly) higher. For example, set `vector_store_k` to 10 and `rerank_k` to 3
- `rerank_model` The cross encoder reranking retrieval model to use. Sensible defaults are `cross-encoder/ms-marco-TinyBERT-L-2-v2` for speed and `colbert-ir/colbertv2.0` for accuracy (`antoinelouis/colbert-xm` for multilingual). Set this value to  `flashrank` to use the FlashrankReranker.

## LLM parameters
- `temperature` The chat LLM's temperature. Increase this to create more diverse answers
- `repetition_penalty` The penalty for repeating outputs in the chat answers. Some models are very sensitive to this parameter and need a value bigger than 1.0 (penalty) while others benefit from inversing it (lower than 1.0)
- `max_new_tokens` This caps how much tokens the LLM can generate in its answer. More tokens means slower throughput and more memory usage

## Prompt configuration
- `rag_instruction` An instruction message for the LLM to let it know what to do. Should include a mentioning of it performing RAG and that documents will be given as input context to generate the answer from.
- `rag_question_initial` The initial question prompt that will be given to the LLM only for the first question a user asks, that is, without chat history
- `rag_question_followup` This is a follow-up question the user is asking. While the context resulting from the prompt will be populated by RAG from the vector store, if chat history is present, this prompt will be used instead of `rag_question_initial`

### Document retrieval
- `rag_fetch_new_instruction` RAG Me Up automatically determines whether or not new documents should be fetched from the vector store or whether the user is asking a follow-up question on the already fetched documents by leveraging the same LLM that is used for chat. This environment variable determines the prompt to use to make this decision. Be very sure to instruct your LLM to answer with yes or no only and make sure your LLM is capable enough to follow this instruction
- `rag_fetch_new_question` The question prompt used in conjunction with `rag_fetch_new_instruction` to decide if new documents should be fetched or not

### Rewriting (self-inflection)
- `user_rewrite_loop` Set to either True or False to enable the rewriting of the initial query. Note that a rewrite will always occur at most once
- `rewrite_query_instruction` This is the instruction of the prompt that is used to ask the LLM to judge whether a rewrite is necessary or not. Make sure you force the LLM to answer with yes or no only
- `rewrite_query_question` This is the actual query part of the prompt that isued to ask the LLM to judge a rewrite
- `rewrite_query_prompt` If the rewrite loop is on and the LLM judges a rewrite is required, this is the instruction with question asked to the LLM to rewrite the user query into a phrasing more optimized for RAG. Make sure to instruct your model adequately.

### Re2
- `use_re2` Set to either True or False to enable [Re2 (Re-reading)](https://arxiv.org/abs/2309.06275) which repeats the question, generally improving the quality of the answer generated by the LLM.
- `re2_prompt` The prompt used in between the question and the repeated question to signal that we are re-asking.

## Document splitting configuration
- `splitter` The Langchain document splitter to use. Supported splitters are `RecursiveCharacterTextSplitter` and `SemanticChunker`.
- `chunk_size` The chunk size to use when splitting up documents for `RecursiveCharacterTextSplitter`
- `chunk_overlap` The chunk overlap for `RecursiveCharacterTextSplitter`
- `breakpoint_threshold_type` Sets the breakpoint threshold type when using the `SemanticChunker` ([see here](https://python.langchain.com/v0.2/docs/how_to/semantic-chunker/)). Can be one of: percentile, standard_deviation, interquartile, gradient
- `breakpoint_threshold_amount` The amount to use for the threshold type, in float. Set to `None` to leave default
- `number_of_chunks` The number of chunks to use for the threshold type, in int. Set to `None` to leave default

# Evaluation
While RAG evaluation is difficult and subjective to begin with, frameworks such as [Ragas](https://docs.ragas.io/en/stable/) can give some metrics as to how well your RAG pipeline and its prompts are working, allowing us to benchmark one approach over the other quantitatively.

RAG Me Up uses Ragas to evaluate your pipeline. You can run an evaluation based on your `.env` using `python Ragas_eval.py`. The following configuration parameters can be set for evaluation:

- `ragas_sample_size` The amount of document (chunks) to use in evaluation. These are sampled from your data directory after chunking.
- `ragas_qa_pairs` Ragas works upon questions and ground truth answers. The amount of such pairs to create based on the sampled document chunks is set by this parameter.
- `ragas_question_instruction` The instruction prompt used to generate the questions of the Ragas input pairs.
- `ragas_question_query` The query prompt used to generate the questions of the Ragas input pairs.
- `ragas_answer_instruction` The instruction prompt used to generate the answers of the Ragas input pairs.
- `ragas_answer_query` The query prompt used to generate the answers of the Ragas input pairs.

# Funding
We are actively looking for funding to democratize AI and advance its applications. Contact us at info@commandos.ai if you want to invest.

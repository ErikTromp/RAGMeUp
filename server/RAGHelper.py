import hashlib
import os
import pickle
import re
import torch

from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_community.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    Docx2txtLoader,
    JSONLoader,
    PyPDFDirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
)
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents.base import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from lxml import etree
from PostgresBM25Retriever import PostgresBM25Retriever
from ScoredCrossEncoderReranker import ScoredCrossEncoderReranker
from tqdm import tqdm
from provenance import (
    compute_attention,
    compute_llm_provenance_cloud,
    compute_rerank_provenance,
    compute_llm_provenance,
    DocumentSimilarityAttribution
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

class RAGHelper:
    """
    A helper class to manage retrieval-augmented generation (RAG) processes,
    including data loading, chunking, vector storage, and retrieval.
    """

    def __init__(self, logger):
        """
        Initializes the RAGHelper class and loads environment variables.
        """
        self.is_local = not(any(os.getenv(key) == "True" for key in ["use_openai", "use_gemini", "use_azure", "use_ollama"]))
        if self.is_local:
            self.human_role_name = "user"
        else:
            self.human_role_name = self.human_role_name
        self.logger = logger
        self.chunked_documents = []
        self.embeddings = None
        self.text_splitter = None
        self.db = None
        self.sparse_retriever = None
        self.ensemble_retriever = None
        self.rerank_retriever = None
        self._batch_size = 1000
        self.vector_store_uri = os.getenv("postgres_uri")
        self.vector_store_sparse_uri = self.vector_store_uri.replace("+psycopg", "")
        self.data_dir = os.getenv("data_directory")
        self.file_types = os.getenv("file_types").split(",")
        self.splitter_type = os.getenv("splitter")
        self.rerank = os.getenv("rerank") == "True"
        self.rerank_model = os.getenv("rerank_model")
        self.rerank_k = int(os.getenv("rerank_k"))
        self.vector_store_k = int(os.getenv("vector_store_k"))
        self.chunk_size = int(os.getenv("chunk_size"))
        self.chunk_overlap = int(os.getenv("chunk_overlap"))
        self.breakpoint_threshold_amount = int(os.getenv('breakpoint_threshold_amount')) if os.getenv('breakpoint_threshold_amount', 'None') != 'None' else None
        self.number_of_chunks = None if (value := os.getenv('number_of_chunks',
                                                            None)) is None or value.lower() == 'none' else int(value)
        self.breakpoint_threshold_type = os.getenv('breakpoint_threshold_type')
        self.vector_store_collection = os.getenv("postgres_collection")
        self.xml_xpath = os.getenv("xml_xpath")
        self.json_text_content = (
            os.getenv("json_text _content", "false").lower() == "true"
        )
        self.json_schema = os.getenv("json_schema")

        ### Set up databases
        self.logger.info(f"Setting up PGVector DB.")
        self.db = PGVector(
            embeddings=self.embeddings,
            collection_name=self.vector_store_collection,
            connection=self.vector_store_uri,
            use_jsonb=True,
        )

        self.logger.info(f"Setting up PostgresBM25Retriever.")
        self._initialize_postgresbm25retriever()

        ### Set up LLM and Langchain chains, load the data
        self.llm = self.initialize_llm()
        self.embeddings = self.initialize_embeddings()
        self.load_data()
        self.initialize_rag_chains()
        self.attributor = DocumentSimilarityAttribution() if os.getenv("provenance_method") == "similarity" else None
        self.initialize_rewrite_loops()
    
    ############################
    ### Initialization functions
    ############################
    def get_llm(self):
        """Accessor method to get the LLM. Subclasses can override this."""
        return self.llm
    
    @staticmethod
    def _get_bnb_config():
        """Get BitsAndBytes configuration for model quantization."""
        use_4bit = True
        bnb_4bit_compute_dtype = "float16"
        bnb_4bit_quant_type = "nf4"
        use_nested_quant = False

        return BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=use_nested_quant,
        )
    
    def initialize_llm(self):
        """Initialize the Language Model based on environment configurations."""
        if os.getenv("use_openai") == "True":
            self.logger.info("Initializing OpenAI conversation.")
            return ChatOpenAI(
                model=os.getenv("openai_model_name"),
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
        if os.getenv("use_gemini") == "True":
            self.logger.info("Initializing Gemini conversation.")
            return ChatGoogleGenerativeAI(
                model=os.getenv("gemini_model_name"),
                convert_system_message_to_human=True,
            )
        if os.getenv("use_azure") == "True":
            self.logger.info("Initializing Azure OpenAI conversation.")
            return AzureChatOpenAI(
                openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
            )
        if os.getenv("use_ollama") == "True":
            self.logger.info("Initializing Ollama conversation.")
            return OllamaLLM(model=os.getenv("ollama_model"))

        """Initialize the LLM based on the available hardware and configurations."""
        llm_model = os.getenv('llm_model')
        trust_remote_code = os.getenv('trust_remote_code') == "True"

        if torch.backends.mps.is_available():
            self.logger.info("Initializing LLM on MPS.")
            tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=trust_remote_code)
            model = AutoModelForCausalLM.from_pretrained(
                llm_model,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float16,
                device_map="auto"
            ).to(torch.device("mps"))
        elif os.getenv('force_cpu') == "True":
            self.logger.info("LLM on CPU (slow!).")
            tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=trust_remote_code)
            model = AutoModelForCausalLM.from_pretrained(
                llm_model,
                trust_remote_code=trust_remote_code,
            ).to(torch.device("cpu"))
        else:
            self.logger.info("Initializing LLM on GPU.")
            bnb_config = self._get_bnb_config()
            tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=trust_remote_code)
            model = AutoModelForCausalLM.from_pretrained(
                llm_model,
                quantization_config=bnb_config,
                trust_remote_code=trust_remote_code,
                device_map="auto"
            )

        text_generation_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            temperature=float(os.getenv('temperature')),
            repetition_penalty=float(os.getenv('repetition_penalty')),
            return_full_text=True,
            max_new_tokens=int(os.getenv('max_new_tokens')),
            model_kwargs={
                'device_map': 'auto',
            }
        )
        return HuggingFacePipeline(pipeline=text_generation_pipeline)
    
    def initialize_embeddings(self):
        """Initialize the embeddings based on the CPU/GPU configuration."""
        embedding_model = os.getenv("embedding_model")
        model_kwargs = (
            {"device": "cpu"}
            if os.getenv("force_cpu") == "True"
            else {"device": "cuda"}
        )
        self.logger.info(
            f"Initializing embedding model {embedding_model} with params {model_kwargs}."
        )
        return HuggingFaceEmbeddings(
            model_name=embedding_model, model_kwargs=model_kwargs
        )
    
    def initialize_rag_chains(self):
        """Create the RAG chain for fetching new documents."""
        rag_thread = [
            ("system", os.getenv("rag_fetch_new_instruction")),
            (self.human_role_name, os.getenv("rag_fetch_new_question")),
        ]
        self.logger.info("Initializing RAG chains for fetching new documents.")
        rag_prompt = ChatPromptTemplate.from_messages(rag_thread)
        rag_llm_chain = rag_prompt | self.llm
        self.rag_fetch_new_chain = {"question": RunnablePassthrough()} | rag_llm_chain
    
    def initialize_rewrite_loops(self):
        """Create rewrite loop LLM chains if enabled."""
        if os.getenv("use_rewrite_loop") == "True":
            self.rewrite_ask_chain = self.create_rewrite_ask_chain()
            self.rewrite_chain = self.create_rewrite_chain()

    def create_rewrite_ask_chain(self):
        """Create the chain to ask if a rewrite is needed."""
        rewrite_ask_thread = [
            ("system", os.getenv("rewrite_query_instruction")),
            (self.human_role_name, os.getenv("rewrite_query_question")),
        ]
        rewrite_ask_prompt = ChatPromptTemplate.from_messages(rewrite_ask_thread)
        rewrite_ask_llm_chain = rewrite_ask_prompt | self.llm
        context_retriever = (
            self.rerank_retriever if self.rerank else self.ensemble_retriever
        )
        return {
            "context": context_retriever | RAGHelper.format_documents,
            "question": RunnablePassthrough(),
        } | rewrite_ask_llm_chain

    def create_rewrite_chain(self):
        """Create the chain to perform the actual rewrite."""
        rewrite_thread = [(self.human_role_name, os.getenv("rewrite_query_prompt"))]
        rewrite_prompt = ChatPromptTemplate.from_messages(rewrite_thread)
        rewrite_llm_chain = rewrite_prompt | self.llm
        return {"question": RunnablePassthrough()} | rewrite_llm_chain
    
    def create_interaction_thread(self, history: list, fetch_new_documents: bool) -> list:
        """Create the conversation thread based on user input and history.

        Args:
            user_query (str): The user's query.
            history (list): The history of previous interactions.
            fetch_new_documents (bool): Whether to fetch new documents.

        Returns:
            list: The constructed conversation thread.
        """
        # Create prompt template based on whether we have history or not
        thread = [
            (x["role"], x["content"].replace("{", "(").replace("}", ")"))
            for x in history
        ]
        if fetch_new_documents:
            thread = [
                ("system", os.getenv("rag_instruction")),
                (self.human_role_name, os.getenv("rag_question_initial")),
            ]
        else:
            thread.append((self.human_role_name, os.getenv("rag_question_followup")))
        return thread

    ########################
    ### Data Loading Section
    ########################
    def load_data(self):
        """
        Loads data from various file types and chunks it into an ensemble retriever.
        """
        if os.path.exists(os.getenv("vector_store_load_file")):
            self.logger.info("Documents have been loaded to vector store before, skipping now.")
        else:
            self.logger.info("Loading the documents for the first time.")
            docs = self._load_documents()
            self.logger.info("Chunking the documents.")
            self.chunked_documents = self._split_documents(docs)
            self._deduplicate_chunks()
            self._write_to_vector_store()
            # Create the signal file
            with open(os.getenv("vector_store_load_file"), "w") as f:
                f.write("")
        
        self._setup_retrievers()

    def _load_chunked_documents(self):
        """Loads previously chunked documents from a pickle file."""
        with open(self.document_chunks_pickle, "rb") as f:
            self.logger.info("Loading chunked documents.")
            self.chunked_documents = pickle.load(f)

    def _load_json_files(self):
        """
        Loads JSON files from the data directory.

        Returns:
            list: A list of loaded Document objects from JSON files.
        """
        text_content = self.json_text_content
        loader_kwargs = {"jq_schema": self.json_schema, "text_content": text_content}
        loader = DirectoryLoader(
            path=self.data_dir,
            glob="*.json",
            loader_cls=JSONLoader,
            loader_kwargs=loader_kwargs,
            recursive=True,
            show_progress=True,
        )
        return loader.load()

    def _load_xml_files(self):
        """
        Loads XML files from the data directory and extracts relevant elements.

        Returns:
            list: A list of Document objects created from XML elements.
        """
        loader = DirectoryLoader(
            path=self.data_dir,
            glob="*.xml",
            loader_cls=TextLoader,
            recursive=True,
            show_progress=True,
        )
        xmldocs = loader.load()
        newdocs = []
        for index, doc in enumerate(xmldocs):
            try:
                xmltree = etree.fromstring(doc.page_content.encode("utf-8"))
                elements = xmltree.xpath(self.xml_xpath)
                elements = [
                    etree.tostring(element, pretty_print=True).decode()
                    for element in elements
                ]
                metadata = doc.metadata
                metadata["index"] = index
                newdocs += [
                    Document(page_content=content, metadata=metadata)
                    for content in elements
                ]
            except Exception as e:
                self.logger.error(f"Error processing XML document: {e}")
        return newdocs

    def _load_documents(self):
        """
        Loads documents from specified file types in the data directory.

        Returns:
            list: A list of loaded Document objects.
        """
        docs = []
        for file_type in self.file_types:
            try:
                self.logger.info(f"Loading {file_type} document(s)....")
                if file_type == "pdf":
                    loader = PyPDFDirectoryLoader(self.data_dir)
                    docs += loader.load()
                elif file_type == "json":
                    docs += self._load_json_files()
                elif file_type == "txt":
                    loader = DirectoryLoader(
                        path=self.data_dir,
                        glob="*.txt",
                        loader_cls=TextLoader,
                        recursive=True,
                        show_progress=True,
                    )
                    docs += loader.load()
                elif file_type == "csv":
                    loader = DirectoryLoader(
                        path=self.data_dir,
                        glob="*.csv",
                        loader_cls=CSVLoader,
                        recursive=True,
                        show_progress=True,
                    )
                    docs += loader.load()
                elif file_type == "docx":
                    loader = DirectoryLoader(
                        path=self.data_dir,
                        glob="*.docx",
                        loader_cls=Docx2txtLoader,
                        recursive=True,
                        show_progress=True,
                    )
                    docs += loader.load()
                elif file_type == "xlsx":
                    loader = DirectoryLoader(
                        path=self.data_dir,
                        glob="*.xlsx",
                        loader_cls=UnstructuredExcelLoader,
                        recursive=True,
                        show_progress=True,
                    )
                    docs += loader.load()
                elif file_type == "pptx":
                    loader = DirectoryLoader(
                        path=self.data_dir,
                        glob="*.pptx",
                        loader_cls=UnstructuredPowerPointLoader,
                        recursive=True,
                        show_progress=True,
                    )
                    docs += loader.load()
                elif file_type == "xml":
                    docs += self._load_xml_files()
            except Exception as e:
                print(f"Error loading {file_type} files: {e}")

        return docs

    def _load_json_document(self, filename):
        """Load JSON documents with specific parameters"""
        return JSONLoader(
            file_path=filename,
            jq_schema=self.json_schema,
            text_content=self.json_text_content,
        )

    def _load_document(self, filename):
        """Load documents from the specified file based on its extension."""
        file_type = filename.lower().split(".")[-1]
        loaders = {
            "pdf": PyPDFLoader,
            "json": self._load_json_document,
            "txt": TextLoader,
            "csv": CSVLoader,
            "docx": Docx2txtLoader,
            "xlsx": UnstructuredExcelLoader,
            "pptx": UnstructuredPowerPointLoader,
        }
        self.logger.info(f"Loading {file_type} document....")
        if file_type in loaders:
            docs = loaders[file_type](filename).load()
            return docs
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def _create_recursive_text_splitter(self):
        """
        Creates an instance of RecursiveCharacterTextSplitter.

        Returns:
            RecursiveCharacterTextSplitter: A configured text splitter instance.
        """
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            keep_separator=True,
            separators=[
                "\n \n",
                "\n\n",
                "\n",
                ".",
                "!",
                "?",
                " ",
                ",",
                "\u200b",
                "\uff0c",
                "\u3001",
                "\uff0e",
                "\u3002",
                "",
            ],
        )

    def _create_semantic_chunker(self):
        """
        Creates an instance of SemanticChunker.

        Returns:
            SemanticChunker: A configured semantic chunker instance.
        """
        return SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type=self.breakpoint_threshold_type,
            breakpoint_threshold_amount=self.breakpoint_threshold_amount,
            number_of_chunks=self.number_of_chunks,
        )

    def _initialize_text_splitter(self):
        """Initialize the text splitter based on the environment settings."""
        self.logger.info(f"Initializing {self.splitter_type} splitter.")
        if self.splitter_type == "RecursiveCharacterTextSplitter":
            self.text_splitter = self._create_recursive_text_splitter()
        elif self.splitter_type == "SemanticChunker":
            self.text_splitter = self._create_semantic_chunker()

    def _split_documents(self, docs):
        """
        Splits documents into chunks.

        Args:
            docs (list): A list of loaded Document objects.
        """
        self._initialize_text_splitter()
        self.logger.info("Chunking document(s).")
        chunked_documents = [
            Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "id": hashlib.md5(doc.page_content.encode()).hexdigest(),
                },
            )
            for doc in self.text_splitter.split_documents(docs)
        ]
        return chunked_documents


    ########################
    ### Database I/O section
    ########################
    def _write_to_vector_store(self):
        """Write the document chunks to the vector store in bulk."""
        self.logger.info("Loading data from existing store.")
        # Add the documents 1 by 1, so we can track progress
        with tqdm(
            total=len(self.chunked_documents), desc="Writing vectors to dense and BM25 DB"
        ) as pbar:
            for i in range(0, len(self.chunked_documents), self._batch_size):
                # Slice the documents for the current batch
                batch = self.chunked_documents[i : i + self._batch_size]
                # Prepare documents and their IDs for batch insertion
                documents = [d for d in batch]
                ids = [d.metadata["id"] for d in batch]

                # Add the batch of documents to the database
                self.db.add_documents(documents, ids=ids)
                self.sparse_retriever.add_documents(documents, ids=ids)

                # Update the progress bar by the size of the batch
                pbar.update(len(batch))

    def _initialize_postgresbm25retriever(self):
        """Initializes the PostgresBM25Retriever."""
        self.logger.info("Initializing PostgresBM25Retriever.")
        self.sparse_retriever = PostgresBM25Retriever(
            connection_uri=self.vector_store_sparse_uri,
            table_name="sparse_vectors",
            k=self.vector_store_k,
        )

    def _initialize_reranker(self):
        """Initialize the reranking model based on environment settings."""
        if self.rerank_model == "flashrank":
            self.logger.info("Setting up the FlashrankRerank.")
            self.compressor = FlashrankRerank(top_n=self.rerank_k)
        else:
            self.logger.info("Setting up the ScoredCrossEncoderReranker.")
            self.compressor = ScoredCrossEncoderReranker(
                model=HuggingFaceCrossEncoder(model_name=self.rerank_model),
                top_n=self.rerank_k,
            )
        self.logger.info("Setting up the ContextualCompressionRetriever.")
        self.rerank_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor, base_retriever=self.ensemble_retriever
        )

    def _setup_retrievers(self):
        """Sets up the retrievers based on specified configurations."""
        # Set up the vector retriever
        self.logger.info("Setting up the Vector Retriever.")
        retriever = self.db.as_retriever(
            search_type="mmr", search_kwargs={"k": self.vector_store_k}
        )
        self.logger.info("Setting up the hybrid retriever.")
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.sparse_retriever, retriever], weights=[0.5, 0.5]
        )
        if self.rerank:
            self._initialize_reranker()

    def _add_to_vector_database(self, new_chunks):
        """Add the new document chunks to the vector database."""
        # Deduplicate to prevent conflicts
        documents = list({d.metadata["id"]: d for d in new_chunks}.values())
        ids = [d.metadata["id"] for d in documents]
        self.db.add_documents(documents, ids=ids)

    def _deduplicate_chunks(self):
        """Ensure there are no duplicate entries in the data."""
        self.chunked_documents = list(
            {doc.metadata["id"]: doc for doc in self.chunked_documents}.values()
        )

    def add_document(self, filename):
        """
        Load documents from various file types, extract metadata,
        split the documents into chunks, and store them in a vector database.

        Parameters:
            filename (str): The name of the file to be loaded.

        Raises:
            ValueError: If the file type is unsupported.
        """
        new_docs = self._load_document(filename)

        self.logger.info("chunking the documents.")
        new_chunks = self._split_documents(new_docs)

        # Add new chunks to the vector database
        self._add_to_vector_database(new_chunks)
    

    ##################################
    ### LLM interaction chains section
    ##################################
    def _parse_answer(self, answer):
        if self.is_local:
            end_string = os.getenv("llm_assistant_token")
            return response['text'][response['text'].rindex(end_string) + len(end_string):]
        else:
            if hasattr(response, "content"):
                response = response.content
            elif hasattr(response, "answer"):
                response = response.answer
            elif "answer" in response:
                response = response["answer"]
            return response

    @staticmethod
    def format_documents(docs):
        """
        Formats the documents for better readability.

        Args:
            docs (list): List of Document objects.

        Returns:
            str: Formatted string representation of documents.
        """
        doc_strings = []
        for i, doc in enumerate(docs):
            metadata_string = ", ".join(
                [f"{md}: {doc.metadata[md]}" for md in doc.metadata.keys()]
            )
            doc_strings.append(
                f"Document {i} content: {doc.page_content}\nDocument {i} metadata: {metadata_string}"
            )
        return "\n\n<NEWDOC>\n\n".join(doc_strings)
    
    def handle_rewrite(self, user_query: str) -> str:
        """Determine if a rewrite is needed and perform it if required.

        Args:
            user_query (str): The original user query.

        Returns:
            str: The potentially rewritten user query.
        """
        if os.getenv("use_rewrite_loop") == "True":
            response = self.rewrite_ask_chain.invoke(user_query)
            self.logger.info(f"The response of the rewrite loop is - {response}")
            response = self._parse_answer(response)

            if re.sub(r"\W+ ", "", response).lower().startswith("yes"):
                return self._parse_answer(
                    self.rewrite_chain.invoke(user_query)
                )
        return user_query

    def combine_results(inputs: dict) -> dict:
        """Combine the results of the user query processing.

        Args:
            inputs (dict): The input results.

        Returns:
            dict: A dictionary containing the answer, context, and question.
        """
        combined = {"answer": inputs["answer"], "question": inputs["question"]}
        if "context" in inputs and "docs" in inputs:
            combined.update({"docs": inputs["docs"], "context": inputs["context"]})
        return combined
    
    def handle_user_interaction(self, user_query: str, history: list) -> tuple:
        """Handle user interaction by processing their query and maintaining conversation history.

        Args:
            user_query (str): The user's query.
            history (list): The history of previous interactions.

        Returns:
            tuple: A tuple containing the conversation thread and the reply.
        """
        fetch_new_documents = self.should_fetch_new_documents(user_query, history)

        thread = self.create_interaction_thread(history, fetch_new_documents)
        # Create prompt from prompt template
        prompt = ChatPromptTemplate.from_messages(thread)

        # Create llm chain
        llm_chain = prompt | self.llm

        if fetch_new_documents:
            context_retriever = (
                self.rerank_retriever if self.rerank else self.ensemble_retriever
            )
            retriever_chain = {
                "retriever_docs": context_retriever,  # Lazy retrieval from context retriever
                "question": RunnablePassthrough(),
            } | RunnableLambda(
                lambda input_data: {
                    "docs": input_data["retriever_docs"],
                    "context": RAGHelper.format_documents(input_data["retriever_docs"]),
                    "question": user_query,
                }
            )
            llm_chain = prompt | self.llm | StrOutputParser()
            rag_chain = (
                retriever_chain
                | RunnablePassthrough.assign(
                    answer=lambda x: llm_chain.invoke(
                        {
                            "docs": x["docs"],
                            "context": x["context"],
                            "question": x["question"],
                        }
                    )
                )
                | self.combine_results
            )
        else:
            retriever_chain = {"question": RunnablePassthrough()}
            llm_chain = prompt | self.llm | StrOutputParser()
            rag_chain = (
                retriever_chain
                | RunnablePassthrough.assign(
                    answer=lambda x: llm_chain.invoke({"question": x["question"]})
                )
                | self.combine_results
            )
        user_query = self.handle_rewrite(user_query)
        # Check if we need to apply Re2 to mention the question twice
        if os.getenv("use_re2") == "True":
            user_query = f'{user_query}\n{os.getenv("re2_prompt")}{user_query}'

        # Invoke RAG pipeline
        reply = rag_chain.invoke(user_query)
        # Track provenance if needed
        if fetch_new_documents and os.getenv("provenance_method") in [
            "rerank",
            "attention",
            "similarity",
            "llm",
        ]:
            self.track_provenance(reply, user_query, thread)

        return (thread, reply)

    def should_fetch_new_documents(self, user_query: str, history: list) -> bool:
        """Determine if new documents should be fetched based on user query and history.

        Args:
            user_query (str): The user's query.
            history (list): The history of previous interactions.

        Returns:
            bool: True if new documents should be fetched, False otherwise.
        """
        if not history:
            self.logger.info("There is no content in history, fetching new documents!")
            return True
        response = self.rag_fetch_new_chain.invoke(user_query)
        response = self._parse_answer(response)
        return re.sub(r"\W+ ", "", response).lower().startswith("yes")

    #######################
    ### Provenance handling
    #######################
    def track_provenance(self, reply: str, user_query: str, thread: list) -> None:
        """Track the provenance of the response if applicable.

        Args:
            reply (str): The response from the LLM.
            user_query (str): The original user query.
        """
        # Add the user question and the answer to our thread for provenance computation
        # Retrieve answer and context
        answer = reply.get("answer")
        context = reply.get("docs")

        provenance_method = os.getenv("provenance_method")
        self.logger.info(f"Provenance method: {provenance_method}")

        # Use the reranker if the provenance method is 'rerank'
        if provenance_method == "rerank":
            self.logger.info("Using reranking for provenance attribution.")
            if not self.rerank:
                raise ValueError(
                    "Provenance attribution is set to rerank but reranking is not enabled. "
                    "Please choose another method or enable reranking."
                )

            reranked_docs = compute_rerank_provenance(
                self.compressor, user_query, context, answer
            )
            self.logger.debug(
                f"Reranked documents computed: {len(reranked_docs)} docs reranked."
            )

            # Build provenance scores based on reranked docs
            provenance_scores = []
            for doc in context:
                reranked_score = next(
                    (
                        d.metadata["relevance_score"]
                        for d in reranked_docs
                        if d.page_content == doc.page_content
                    ),
                    None,
                )
                if reranked_score is None:
                    self.logger.warning(
                        f"Document not found in reranked docs: {doc.page_content}"
                    )
                provenance_scores.append(reranked_score)
            self.logger.debug("Provenance scores computed using reranked documents.")

        # Use similarity-based provenance if method is 'similarity'
        elif provenance_method == "similarity":
            self.logger.info("Using similarity-based provenance attribution.")
            provenance_scores = self.attributor.compute_similarity(
                user_query, context, answer
            )
            self.logger.debug("Provenance scores computed using similarity method.")

        # Attention-based provenance, can only work with local LLM
        elif provenance_method == "attention":
            new_history = [{"role": msg["role"], "content": msg["content"].format_map(reply)} for msg in thread]
            new_history.append({"role": "assistant", "content": answer})
            return compute_attention(self.model, self.tokenizer,
                                     self.tokenizer.apply_chat_template(new_history, tokenize=False), user_query,
                                     context, answer)
        # Use LLM-based provenance if method is 'llm'
        elif provenance_method == "llm":
            self.logger.info("Using LLM-based provenance attribution.")
            provenance_scores = compute_llm_provenance_cloud(
                self.llm, user_query, context, answer
            )
            self.logger.debug("Provenance scores computed using LLM-based method.")

        # Add provenance scores to documents
        for i, score in enumerate(provenance_scores):
            reply["docs"][i].metadata["provenance"] = score
            self.logger.debug(f"Provenance score added to doc {i}: {score}")
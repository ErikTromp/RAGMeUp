from flask import Flask, request, jsonify, send_file
import logging
from dotenv import load_dotenv
import os
from RAGHelper import RAGHelper
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
    ToolCorrectnessMetric,
)
from deepeval.test_case import LLMTestCase
import random
from werkzeug.utils import secure_filename


# Define custom metric classes
class CounterfactualErrorHandling:
    """Custom metric to evaluate counterfactual error handling in LLMs."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.name = "CounterfactualHandling Metrics"
        self.include_reason = True
        self.strict_mode = True

    @property
    def __name__(self):
        return self.name

    def measure(self, test_case: LLMTestCase):
        """Evaluate if the LLM can handle counterfactual questions properly."""
        question = test_case.input.lower()
        if "if" in question and "not" in question:
            score = 1.0 if "error" not in test_case.actual_output.lower() else 0.0
        else:
            score = 0.0
        self.score = score
        self.passed = score >= self.threshold
        if self.include_reason:
            self.reason = "Handled counterfactual correctly." if self.passed else "Failed to handle counterfactual."

    async def a_measure(self, test_case: LLMTestCase):
        """Asynchronous evaluation method."""
        return self.measure(test_case)

    def is_successful(self):
        """Determine if the metric evaluation was successful."""
        return self.passed


class LongDistanceInformationExtraction:
    """Custom metric to evaluate long-distance information extraction."""

    def __init__(self, threshold: float = 0.5, distance_threshold: int = 50):
        self.threshold = threshold
        self.distance_threshold = distance_threshold
        self.name = "LongDistance Information Extraction Metrics"
        self.include_reason = True
        self.strict_mode = True

    @property
    def __name__(self):
        return self.name

    def measure(self, test_case: LLMTestCase):
        """Evaluate if the LLM extracts relevant information across long distances in context."""
        context = test_case.retrieval_context
        input_text = test_case.input.lower()
        if len(context) > self.distance_threshold:
            relevant_content = any(input_text in doc.lower() for doc in context)
            score = 1.0 if relevant_content else 0.0
        else:
            score = 0.0
        self.score = score
        self.passed = score >= self.threshold
        if self.include_reason:
            self.reason = "Successfully extracted long-distance information." if self.passed else "Failed to extract long-distance information."

    async def a_measure(self, test_case: LLMTestCase):
        """Asynchronous evaluation method."""
        return self.measure(test_case)

    def is_successful(self):
        """Determine if the metric evaluation was successful."""
        return self.passed

# Initialize Flask application
app = Flask(__name__)

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable parallelism in tokenizers to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Instantiate the RAG Helper class
logger.info("Instantiating RAG helper.")
raghelper = RAGHelper(logger)


@app.route("/add_document", methods=['POST'])
def add_document():
    """
    Add a document to the RAG helper.

    This endpoint expects a JSON payload containing the filename of the document to be added.
    It then invokes the addDocument method of the RAG helper to store the document.

    Returns:
        JSON response with the filename and HTTP status code 200.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        
        logger.info(f"Adding document {filename}")
        data_dir = os.getenv("data_directory")
        location = f"{data_dir}/{file.filename}"

        # Copy over file
        file.save(location)
        raghelper.add_document(location)
    else:
        return jsonify({"error": "File is required"}), 400

    return jsonify({"filename": filename}), 200


@app.route("/deepeval_evaluate", methods=["POST"])
def deepeval_evaluate():
    """
    Endpoint to run DeepEval evaluation logic.
    Expects JSON input specifying evaluation parameters.
    """
    # Default metric configurations
    llm_model = os.getenv("llm_model", "gpt-3.5-turbo")
    metrics = [
        AnswerRelevancyMetric(threshold=0.7, model=llm_model),
        FaithfulnessMetric(threshold=0.7, model=llm_model),
        ContextualPrecisionMetric(threshold=0.7, model=llm_model),
        ContextualRecallMetric(threshold=0.7, model=llm_model),
        ContextualRelevancyMetric(threshold=0.7, model=llm_model),
        HallucinationMetric(threshold=0.5, model=llm_model),
        ToolCorrectnessMetric(threshold=0.5),
        CounterfactualErrorHandling(threshold=0.7),
        LongDistanceInformationExtraction(threshold=0.7, distance_threshold=50)
    ]

    json_data = request.get_json()
    sample_size = int(json_data.get("sample_size", 10))
    qa_pairs_count = int(json_data.get("qa_pairs", 5))

    # Load and shuffle documents
    documents = raghelper.chunked_documents
    random.shuffle(documents)
    document_sample = documents[:min(sample_size, len(documents))]

    # Generate QA pairs
    qa_pairs = []
    for _ in range(qa_pairs_count):
        random.shuffle(document_sample)
        selected_docs = document_sample[:min(len(document_sample), 10)]
        formatted_docs = RAGHelperLocal.format_documents(selected_docs)

        # Generate question and answer
        question = "Generated question placeholder"  # Replace with actual generation logic
        answer = "Generated answer placeholder"  # Replace with actual generation logic

        qa_pairs.append({
            "question": question,
            "ground_truth": answer,
            "context": [doc.page_content for doc in selected_docs],
        })

    logger.info(f"Generated QA Pairs: {qa_pairs}")
    for i, pair in enumerate(qa_pairs):
        if not pair["question"] or not pair["ground_truth"] or not pair["context"]:
            logger.error(f"Incomplete QA pair at index {i}: {pair}")

    # Prepare test cases
    test_cases = [
        LLMTestCase(
            input=pair["question"],
            actual_output=pair["ground_truth"],
            expected_output=pair["ground_truth"],
            retrieval_context=pair["context"],
        )
        for pair in qa_pairs
    ]

    # Evaluate
    results = evaluate(test_cases, metrics)
    logger.info(f"Evaluation Results: {results}")

    return jsonify({"evaluation_results": results})

@app.route("/create_title", methods=['POST'])
def create_title():
    json_data = request.get_json()
    question = json_data.get('question')

    llm = raghelper.get_llm()
    response = llm.invoke([{"role": "user", "content": f"Write a succinct title (few words) for a chat that has the question. {question}\n\nYou NEVER give explanations, only the title. You also stick to the language of the question."}])
    if isinstance(raghelper, RAGHelperCloud):
        response = response.content
    
    return jsonify({"title": response}), 200

@app.route("/chat", methods=['POST'])
def chat():
    """
    Handle chat interactions with the RAG system.

    This endpoint processes the user's prompt, retrieves relevant documents,
    and returns the assistant's reply along with conversation history.

    Returns:
        JSON response containing the assistant's reply, history, documents, and other metadata.
    """
    json_data = request.get_json()
    prompt = json_data.get('prompt')
    history = json_data.get('history', [])
    original_docs = json_data.get('docs', [])
    docs = original_docs

    # Get the LLM response
    (new_history, response) = raghelper.handle_user_interaction(prompt, history)
    if not docs or 'docs' in response:
        docs = response['docs']

    # Break up the response for local LLMs
    if isinstance(raghelper, RAGHelperLocal):
        end_string = os.getenv("llm_assistant_token")
        reply = response['text'][response['text'].rindex(end_string) + len(end_string):]

        # Get updated history
        new_history = [{"role": msg["role"], "content": msg["content"].format_map(response)} for msg in new_history]
        new_history.append({"role": "assistant", "content": reply})
    else:
        # Populate history for other LLMs
        new_history = [{"role": msg[0], "content": msg[1].format_map(response)} for msg in new_history]
        new_history.append({"role": "assistant", "content": response['answer']})
        reply = response['answer']

    # Format documents
    fetched_new_documents = False
    if not original_docs or 'docs' in response:
        fetched_new_documents = True
        new_docs = [{
            's': doc.metadata['source'],
            'c': doc.page_content,
            **({'pk': doc.metadata['pk']} if 'pk' in doc.metadata else {}),
            **({'provenance': float(doc.metadata['provenance'])} if 'provenance' in doc.metadata and doc.metadata[
                'provenance'] is not None else {})
        } for doc in docs if 'source' in doc.metadata]
    else:
        new_docs = docs

    # Build the response dictionary
    response_dict = {
        "reply": reply,
        "history": new_history,
        "documents": new_docs,
        "rewritten": False,
        "question": prompt,
        "fetched_new_documents": fetched_new_documents
    }

    # Check for rewritten question
    if os.getenv("use_rewrite_loop") == "True" and prompt != response['question']:
        response_dict["rewritten"] = True
        response_dict["question"] = response['question']

    return jsonify(response_dict), 200


@app.route("/get_documents", methods=['GET'])
def get_documents():
    """
    Retrieve a list of documents from the data directory.

    This endpoint checks the configured data directory and returns a list of files
    that match the specified file types.

    Returns:
        JSON response containing the list of files.
    """
    data_dir = os.getenv('data_directory')
    file_types = os.getenv("file_types", "").split(",")

    # Filter files based on specified types
    files = [f for f in os.listdir(data_dir)
             if os.path.isfile(os.path.join(data_dir, f)) and os.path.splitext(f)[1][1:] in file_types]

    return jsonify(files)


@app.route("/get_document", methods=['POST'])
def get_document():
    """
    Retrieve a specific document from the data directory.

    This endpoint expects a JSON payload containing the filename of the document to retrieve.
    If the document exists, it is sent as a file response.

    Returns:
        JSON response with the error message and HTTP status code 404 if not found,
        otherwise sends the file as an attachment.
    """
    json_data = request.get_json()
    filename = json_data.get('filename')
    data_dir = os.getenv('data_directory')
    file_path = os.path.join(data_dir, filename)

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    return send_file(file_path,
                     mimetype='application/octet-stream',
                     as_attachment=True,
                     download_name=filename)


@app.route("/delete", methods=['POST'])
def delete_document():
    """
    Delete a specific document from the data directory and the Milvus vector store.

    This endpoint expects a JSON payload containing the filename of the document to delete.
    It removes the document from the Milvus collection and the filesystem.

    Returns:
        JSON response with the count of deleted documents.
    """
    json_data = request.get_json()
    filename = json_data.get('filename')
    data_dir = os.getenv('data_directory')
    file_path = os.path.join(data_dir, filename)

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    # @TODO: Implement me

    return jsonify({"count": result.delete_count})


if __name__ == "__main__":
    app.run(host="0.0.0.0")
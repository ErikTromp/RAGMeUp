from flask import Flask, request, jsonify, send_file
import logging
from dotenv import load_dotenv
import os
from RAGHelper import RAGHelper
from psycopg2 import pool

# Initialize Flask application
app = Flask(__name__)

# Load environment variables
load_dotenv(override=True)

# Set the logging level
logging_level = os.getenv("logging_level")
if logging_level == "DEBUG":
    logging_level = logging.DEBUG
else:
    logging_level = logging.INFO

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging_level,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Set up a connection pool
db_pool = pool.SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    dsn=os.getenv("postgres_uri")
)

# Instantiate the RAG Helper class
logger.info("Instantiating RAG helper.")
raghelper = RAGHelper(logger, db_pool)

@app.route("/create_title", methods=['POST'])
def create_title():
    json_data = request.get_json()
    question = json_data.get('question')

    (response, _) = raghelper.llm.generate_response(
        None,
        f"Write a succinct title (few words) for a chat that has the question. {question}\n\nYou NEVER give explanations, only the title. You also stick to the language of the question.",
        []
    )
    logger.info(f"Title for question {question}: {response}")

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
    (response, documents, fetched_new_documents, rewritten, new_history, provenance_scores) = raghelper.handle_user_interaction(prompt, history)
    if not fetched_new_documents:
        documents = docs

    response_dict = {
        "reply": response,
        "history": new_history,
        "documents": documents,
        "rewritten": rewritten,
        "question": prompt,
        "fetched_new_documents": fetched_new_documents
    }
    if provenance_scores is not None:
        for i, doc in enumerate(response_dict["documents"]):
            response_dict["documents"][i]["provenance"] = provenance_scores[i]["score"]
    return jsonify(response_dict)

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
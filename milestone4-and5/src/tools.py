# This file will contain the LangChain tools for Experiment 1.
import os
import pandas as pd
from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings # No longer needed
from langchain_openai import OpenAIEmbeddings # Added
import chromadb
from langchain_community.vectorstores import Chroma # Import Chroma vector store wrapper
# import google.generativeai as genai # No longer needed
from dotenv import load_dotenv

# Load environment variables (especially OPENAI_API_KEY)
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY") # Changed from GEMINI_API_KEY
if not API_KEY:
    # In a real scenario, might raise an error or use a default key
    print("Warning: OPENAI_API_KEY not found in environment variables.") # Changed
# else: # No specific library configuration needed for OpenAI like genai.configure
    # pass

# Global variable to hold the loaded DataFrame
loaded_data_df = None

@tool
def load_data(file_path: str) -> str:
    """
    Reads time-series data from a CSV file into a pandas DataFrame.

    Args:
        file_path: The path to the CSV file.

    Returns:
        A success message indicating the shape of the loaded data or an error message.

    Raises:
        FileNotFoundError: If the file is not found at the specified path.
        Exception: For other potential errors during file reading (e.g., parsing errors).
    """
    global loaded_data_df
    try:
        df = pd.read_csv(file_path)
        loaded_data_df = df # Store the loaded data globally
        success_message = f"Successfully loaded data from {file_path}. Shape: {df.shape}"
        print(success_message)
        return success_message
    except FileNotFoundError:
        error_message = f"Error: File not found at {file_path}"
        print(error_message)
        return error_message # Return error message instead of raising
    except Exception as e:
        error_message = f"Error loading data from {file_path}: {e}"
        print(error_message)
        return error_message # Return error message instead of raising


# --- RAG Setup ---

# Constants for RAG (Paths relative to project root)
KNOWLEDGE_BASE_PATH = "knowledge/paper.pdf" # Relative to project root
PERSIST_DIRECTORY = "db/chroma_knowledge_db" # Relative to project root
COLLECTION_NAME = "domain_knowledge"

def load_and_split_pdf(file_path: str):
    """Loads PDF and splits it into chunks."""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        print(f"Loaded and split {file_path} into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        print(f"Error loading/splitting PDF {file_path}: {e}")
        return []

# Global variable for the retriever
knowledge_retriever = None

def setup_vector_store(knowledge_pdf_path=None, db_persist_path=None, collection_name=None):
    """
    Sets up the ChromaDB vector store and retriever for the knowledge base.
    Instantiates the LangChain Chroma wrapper and checks if the collection
    needs to be populated (loads, splits, embeds PDF only if empty).
    
    Args:
        knowledge_pdf_path: Optional custom path to the PDF file. If None, uses default KNOWLEDGE_BASE_PATH.
        db_persist_path: Optional custom path to store the vector database. If None, uses default PERSIST_DIRECTORY.
        collection_name: Optional custom collection name. If None, uses default COLLECTION_NAME.
    """
    global knowledge_retriever
    if knowledge_retriever:
        print("Retriever already initialized.")
        return knowledge_retriever

    if not API_KEY:
        print("Error: Cannot setup vector store without OPENAI_API_KEY.")
        return None
    
    # Use provided paths or fall back to defaults
    actual_collection_name = collection_name or COLLECTION_NAME

    try:
        # 1. Initialize Embeddings Model
        embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
        print("OpenAIEmbeddings initialized.")

        # 2. Initialize ChromaDB Client and LangChain Wrapper
        # Use the provided db path or compute the default one
        if db_persist_path:
            persist_path = db_persist_path
        else:
            persist_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', PERSIST_DIRECTORY))
        
        os.makedirs(persist_path, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_path) # Keep client for potential direct operations if needed

        # Instantiate the LangChain Chroma wrapper immediately
        vector_store = Chroma(
            client=client, # Pass the client instance
            collection_name=actual_collection_name,
            embedding_function=embeddings,
            persist_directory=persist_path # Important for loading existing data
        )
        print(f"Chroma vector store wrapper initialized for collection '{actual_collection_name}'.")

        # 3. Check if collection needs population
        # Access the underlying collection count via the wrapper
        # Note: This might implicitly trigger collection creation if it doesn't exist via get_or_create
        # A direct count check might be needed if get_collection is preferred first.
        # Let's try a direct count first for clarity.
        try:
             collection = client.get_collection(name=actual_collection_name)
             doc_count = collection.count()
             print(f"Existing collection '{actual_collection_name}' found with {doc_count} documents.")
        except Exception: # Broad exception, ideally catch specific ChromaDB error
             print(f"Collection '{actual_collection_name}' likely doesn't exist yet.")
             doc_count = 0
             # Ensure collection is created if get failed (get_or_create might be cleaner overall)
             # collection = client.get_or_create_collection(name=actual_collection_name)

        if doc_count == 0:
            print(f"Collection '{actual_collection_name}' is empty. Populating...")
            # 4. Load, Split, Embed, and Add PDF
            # Use the provided PDF path or compute the default one
            if knowledge_pdf_path:
                pdf_path = knowledge_pdf_path
            else:
                pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', KNOWLEDGE_BASE_PATH))
            
            if not os.path.exists(pdf_path):
                print(f"Error: Knowledge base PDF not found at {pdf_path}")
                return None

            print(f"Processing and embedding {pdf_path}...")
            chunks = load_and_split_pdf(pdf_path)
            if not chunks:
                print("Failed to load or split PDF. Cannot populate vector store.")
                return None

            ids = [f"chunk_{i}" for i in range(len(chunks))]
            # Add documents using the vector_store wrapper
            vector_store.add_documents(documents=chunks, ids=ids)
            print(f"Successfully added {len(chunks)} chunks to collection '{actual_collection_name}'.")
            # Persist changes explicitly if needed, though PersistentClient usually handles this
            # client.persist() # Or vector_store.persist() if available
        else:
            print("Collection already populated. Skipping embedding.")

        # 5. Create Retriever
        knowledge_retriever = vector_store.as_retriever(search_kwargs={'k': 3})
        print("Knowledge retriever created.")
        return knowledge_retriever

    except Exception as e:
        print(f"An error occurred during vector store setup: {e}", exc_info=True) # Add exc_info for more details
        return None

# Initialize the vector store on module load (optional, could be lazy)
# setup_vector_store()


@tool
def query_domain_knowledge(query_text: str, knowledge_pdf_path=None, db_persist_path=None) -> str:
    """
    Queries the domain knowledge base (embedded PDF) for relevant information.
    Uses the retriever to find document chunks related to the query_text.

    Args:
        query_text: The question or topic to search for in the knowledge base.
        knowledge_pdf_path: Optional custom path to the PDF file.
        db_persist_path: Optional custom path to the vector database.

    Returns:
        A formatted string containing the relevant document excerpts,
        or a message indicating that no relevant information was found or an error occurred.
    """
    global knowledge_retriever
    # Ensure retriever is initialized
    if not knowledge_retriever:
        print("Retriever not initialized. Attempting setup...")
        setup_vector_store(knowledge_pdf_path=knowledge_pdf_path, db_persist_path=db_persist_path)
        if not knowledge_retriever:
            return "Error: Knowledge base retriever could not be initialized."

    try:
        print(f"Querying knowledge base for: '{query_text}'")
        relevant_docs = knowledge_retriever.invoke(query_text)

        if not relevant_docs:
            return "No relevant information found in the knowledge base for your query."

        # Format the results
        formatted_results = "Relevant Information from Knowledge Base:\n\n"
        for i, doc in enumerate(relevant_docs):
            # Include metadata if available and useful (e.g., page number)
            source_info = doc.metadata.get('page', 'N/A')
            formatted_results += f"--- Source Chunk {i+1} (Page: {source_info}) ---\n"
            formatted_results += doc.page_content
            formatted_results += "\n\n"

        return formatted_results.strip()

    except Exception as e:
        print(f"Error querying knowledge base: {e}")
        return f"An error occurred while querying the knowledge base: {e}"


# --- Core Data Tools ---

@tool
def get_segment(start_row: int, end_row: int, column_name: str = None) -> str:
    """
    Extracts a segment (rows and optionally a specific column) from the globally loaded DataFrame.

    Args:
        start_row: The starting row index (inclusive).
        end_row: The ending row index (inclusive).
        column_name: Optional name of the specific column to extract. If None, returns all columns.

    Returns:
        A string representation of the requested segment DataFrame, or an error message.
    """
    global loaded_data_df
    if loaded_data_df is None:
        error_message = "Error: Data not loaded. Use the 'load_data' tool first."
        print(error_message)
        return error_message

    try:
        # Validate row indices
        # Validate row indices
        max_index = len(loaded_data_df) - 1
        if not (0 <= start_row <= max_index and 0 <= end_row <= max_index):
            error_message = f"Error: Invalid row indices [{start_row}, {end_row}]. Max index is {max_index}."
            print(error_message)
            return error_message
        if start_row > end_row:
            error_message = f"Error: Start row {start_row} cannot be greater than end row {end_row}."
            print(error_message)
            return error_message

        # Extract segment
        segment_df = loaded_data_df.iloc[start_row:end_row + 1] # +1 because iloc is exclusive for the end index

        # Handle column selection
        # Handle column selection
        if column_name:
            if column_name not in loaded_data_df.columns:
                error_message = f"Error: Column '{column_name}' not found in DataFrame."
                print(error_message)
                return error_message
            # Select only the specified column
            segment_df = segment_df[[column_name]]

        success_message = f"Extracted segment from row {start_row} to {end_row}" + (f" for column '{column_name}'" if column_name else "") + f". Shape: {segment_df.shape}"
        print(success_message)
        # Return the segment as a string
        return f"{success_message}\nData:\n{segment_df.to_string()}"

    except Exception as e:
        error_message = f"Error extracting segment [{start_row}:{end_row}" + (f", col='{column_name}'" if column_name else "") + f"]: {e}"
        print(error_message)
        return error_message


@tool
def calculate_basic_stats(segment_str: str) -> str:
    """
    Calculates basic statistics (mean, std dev, min, max) for numeric columns in a DataFrame segment provided as a string.
    The input string should be the output of the 'get_segment' tool.

    Args:
        segment_str: A string representation of the pandas DataFrame segment (likely from get_segment tool).

    Returns:
        A formatted string containing the calculated statistics, or an error message.
    """
    # Attempt to parse the DataFrame from the string representation
    # This is a simplified parsing, assuming the string starts with header lines then data
    # A more robust approach might involve passing JSON or using a shared context object
    try:
        from io import StringIO
        # Find where the actual data starts (skip header lines from get_segment output)
        data_start_index = segment_str.find("Data:\n") + len("Data:\n")
        if data_start_index == -1 + len("Data:\n"):
             return "Error: Could not find 'Data:' marker in input string. Ensure input is from get_segment tool."

        segment_data_str = segment_str[data_start_index:]
        segment_df = pd.read_csv(StringIO(segment_data_str), sep='\s+', index_col=0) # Assuming space-separated, adjust if needed

    except Exception as parse_error:
        return f"Error parsing DataFrame from input string: {parse_error}. Ensure input is valid."


    if segment_df is None or segment_df.empty:
        return "Error: Cannot calculate stats on an empty or None segment after parsing."

    try:
        # Select only numeric columns for statistics
        numeric_segment = segment_df.select_dtypes(include='number')

        if numeric_segment.empty:
            return "No numeric columns found in the parsed segment to calculate statistics."

        stats = numeric_segment.agg(['mean', 'std', 'min', 'max'])

        # Format the statistics into a string
        stats_str = "Basic Statistics for the Segment:\n"
        stats_str += stats.to_string()

        print("Calculated basic statistics for the segment.")
        return stats_str

    except Exception as e:
        print(f"Error calculating statistics: {e}")
        return f"An error occurred during statistics calculation: {e}"

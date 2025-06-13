import os
import pandas as pd
import numpy as np
import boto3
import json
import faiss
from botocore.exceptions import ClientError

# Ensure these paths are correct relative to where you run the script
# Assuming qa_dataset.csv is in VoiceBot_MATRIX_submission/data/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
QA_CSV_PATH = os.path.join(DATA_DIR, 'qa_dataset.csv')
EMBEDDINGS_FILE = os.path.join(DATA_DIR, 'qa_embeddings.npy')
FAISS_INDEX_FILE = os.path.join(DATA_DIR, 'qa_faiss_index.bin')

EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
BEDROCK_REGION = "us-west-2" # Make sure this matches your Bedrock region

def get_embedding(text: str, client) -> np.ndarray:
    """Gets an embedding for the given text using Bedrock Titan Text Embeddings."""
    try:
        body = json.dumps({"inputText": text})
        response = client.invoke_model(
            body=body,
            modelId=EMBEDDING_MODEL_ID,
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response.get('body').read())
        embedding = response_body.get('embedding')
        if not embedding:
            raise ValueError("Embedding not found in response.")
        return np.array(embedding).astype('float32')
    except ClientError as e:
        print(f"Error getting embedding from Bedrock (ClientError): {e}")
        if e.response and 'Error' in e.response:
            print(f"AWS Error Code: {e.response['Error'].get('Code')}")
            print(f"AWS Error Message: {e.response['Error'].get('Message')}")
        return None
    except Exception as e:
        print(f"General error getting embedding: {e}")
        return None

def generate_and_save_faiss_index():
    print(f"Loading QA dataset from: {QA_CSV_PATH}")
    if not os.path.exists(QA_CSV_PATH):
        print(f"Error: {QA_CSV_PATH} not found. Please ensure your qa_dataset.csv is in the 'data' directory.")
        return

    try:
        df = pd.read_csv(QA_CSV_PATH)
        required_columns = ['Question', 'Response'] # Ensure these match your CSV headers
        if not all(col in df.columns for col in required_columns):
            print(f"Error: CSV must contain columns: {required_columns}. Found: {list(df.columns)}")
            return
        print("QA dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading QA dataset: {e}")
        return

    print(f"Initializing Bedrock client for embeddings in region: {BEDROCK_REGION}")
    try:
        bedrock_client = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION)
    except Exception as e:
        print(f"Error initializing Bedrock client: {e}")
        print("Please ensure your AWS credentials are configured and Bedrock is available in the specified region.")
        return

    embeddings = []
    questions = df['Question'].tolist()
    print(f"Generating embeddings for {len(questions)} questions...")

    for i, question in enumerate(questions):
        embedding = get_embedding(question, bedrock_client)
        if embedding is not None:
            embeddings.append(embedding)
        else:
            print(f"Skipping question due to embedding error: {question[:50]}...")
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(questions)} questions...")

    if not embeddings:
        print("No embeddings generated. FAISS index cannot be built.")
        return

    embeddings_array = np.array(embeddings).astype('float32')
    print(f"Generated embeddings with shape: {embeddings_array.shape}")

    print("Building FAISS index...")
    # Using IndexFlatL2 for exact nearest neighbor search with L2 distance
    index = faiss.IndexFlatL2(embeddings_array.shape[1])
    index.add(embeddings_array)
    print(f"FAISS index built. Number of vectors in index: {index.ntotal}")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    print(f"Saving embeddings to: {EMBEDDINGS_FILE}")
    np.save(EMBEDDINGS_FILE, embeddings_array)
    
    print(f"Saving FAISS index to: {FAISS_INDEX_FILE}")
    faiss.write_index(index, FAISS_INDEX_FILE)
    
    print("FAISS index and embeddings saved successfully!")

if __name__ == "__main__":
    generate_and_save_faiss_index() 
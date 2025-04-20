"""
Simple script to test Ollama embeddings with mxbai-embed-large model.
"""

import requests
import numpy as np
import time

def get_embedding_ollama(text, model="mxbai-embed-large"):
    """Get embedding from Ollama local model"""
    start_time = time.time()
    
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",  # Correct endpoint is /api/embeddings (plural)
            json={
                "model": model,
                "prompt": text
            }
        )
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            response_data = response.json()
            
            # For debugging, let's check what's in the response
            embeddings_key = None
            
            # Look for the embeddings key (might be 'embedding' or 'embeddings')
            for key in response_data:
                if key.startswith('embedding'):
                    embeddings_key = key
                    break
            
            if not embeddings_key:
                print(f"Response keys: {list(response_data.keys())}")
                raise Exception("No embedding key found in response")
            
            # Extract the first embedding if it's an array of embeddings
            embedding_data = response_data[embeddings_key]
            if isinstance(embedding_data, list):
                if not embedding_data:  # Empty list
                    raise Exception("Empty embeddings array in response")
                # If it's a list of embeddings, take the first one
                if isinstance(embedding_data[0], list):
                    embedding = np.array(embedding_data[0])
                else:
                    # If it's a single embedding in a list
                    embedding = np.array(embedding_data)
            else:
                # If it's directly the embedding values
                embedding = np.array(embedding_data)
            
            return {
                "embedding": embedding,
                "dimensions": len(embedding),
                "elapsed_time": elapsed_time
            }
        else:
            raise Exception(f"Error calling Ollama API: {response.text}")
    except Exception as e:
        print(f"Error getting embedding: {str(e)}")
        raise

def calculate_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    return dot_product / (norm1 * norm2)

def test_mxbai_embed_large():
    """Test mxbai-embed-large model for embeddings"""
    print("Testing mxbai-embed-large model...")
    
    # Test with short text
    text1 = "This is a test sentence for embedding."
    result1 = get_embedding_ollama(text1)
    print(f"Short text embedding dimensions: {result1['dimensions']}")
    print(f"Short text embedding time: {result1['elapsed_time']:.4f} seconds")
    
    # Test with medium text
    text2 = """
    This is a medium-length text that contains multiple sentences.
    It is designed to test the embedding performance with paragraphs.
    The embedding model should process this paragraph and generate
    a meaningful vector representation of its semantic content.
    """
    result2 = get_embedding_ollama(text2)
    print(f"Medium text embedding dimensions: {result2['dimensions']}")
    print(f"Medium text embedding time: {result2['elapsed_time']:.4f} seconds")
    
    # Test similar and different texts
    similar_text1 = "The quick brown fox jumps over the lazy dog."
    similar_text2 = "A fast auburn fox leaps above the sleepy canine."
    different_text = "Machine learning models process large amounts of data."
    
    similar_result1 = get_embedding_ollama(similar_text1)
    similar_result2 = get_embedding_ollama(similar_text2)
    different_result = get_embedding_ollama(different_text)
    
    similarity_similar = calculate_similarity(similar_result1["embedding"], similar_result2["embedding"])
    similarity_different = calculate_similarity(similar_result1["embedding"], different_result["embedding"])
    
    print(f"Similarity between similar sentences: {similarity_similar}")
    print(f"Similarity between different sentences: {similarity_different}")
    print(f"Similarity ratio (similar/different): {similarity_similar/similarity_different:.2f}x")

if __name__ == "__main__":
    test_mxbai_embed_large()

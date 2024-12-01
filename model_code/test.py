from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import pandas as pd

def extract_keywords(text, top_n=5, ngram_range=(1, 2), diversity=0.7):
    """
    Extract keywords from text using KeyBERT
    
    Args:
        text (str): Input text to extract keywords from
        top_n (int): Number of keywords/keyphrases to extract
        ngram_range (tuple): Length of the resulting keywords/keyphrases
        diversity (float): Diversity of results between 0 and 1
        
    Returns:
        list: List of tuples containing keywords and their scores
    """
    # Initialize KeyBERT with a sentence transformer model
    model = KeyBERT(model='all-MiniLM-L6-v2')
    
    # Extract keywords
    keywords = model.extract_keywords(
        text,
        keyphrase_ngram_range=ngram_range,
        stop_words='english',
        top_n=top_n,
        diversity=diversity
    )
    
    return keywords

# Example usage
if __name__ == "__main__":
    # Sample abstract
    abstract = """
    Neural networks have shown great promise in natural language processing tasks.
    This paper presents a novel approach to text classification using deep learning
    techniques. We demonstrate improved performance on benchmark datasets and
    provide comprehensive analysis of the results. Our method shows significant
    improvements in accuracy and computational efficiency.
    """
    
    # Extract keywords
    results = extract_keywords(abstract)
    
    # Print results
    print("Keywords with scores:")
    for keyword, score in results:
        print(f"{keyword}: {score:.4f}")

def batch_process_abstracts(abstracts_list):
    """
    Process multiple abstracts and extract keywords
    
    Args:
        abstracts_list (list): List of abstracts to process
        
    Returns:
        dict: Dictionary with abstracts as keys and keywords as values
    """
    results = {}
    for abstract in abstracts_list:
        keywords = extract_keywords(abstract)
        results[abstract] = keywords
    
    return results
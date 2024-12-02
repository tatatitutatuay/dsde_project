import joblib
import os

# Load the saved model and vectorizer from 'look_good_model' folder
def load_model(model_filename='keyword_extraction_model2.pkl', vectorizer_filename='tfidf_vectorizer2.pkl'):
    """
    Load the trained model and TF-IDF vectorizer from disk.
    
    Args:
        model_filename (str): The name of the model file to load.
        vectorizer_filename (str): The name of the vectorizer file to load.
        
    Returns:
        tuple: Loaded model and vectorizer.
    """
    model = joblib.load(os.path.join('model2', model_filename))
    vectorizer = joblib.load(os.path.join('model2', vectorizer_filename))
    return model, vectorizer

# Predict keywords from a new abstract using the trained model and vectorizer
def predict_keywords_from_model(abstract, model, vectorizer, min_keywords):
    """
    Predict at least `min_keywords` from an abstract using the trained model and vectorizer, 
    considering multi-word n-grams (bigrams, trigrams).
    
    Args:
        abstract (str): The input abstract text.
        model: Trained model (e.g., logistic regression).
        vectorizer: Fitted TF-IDF vectorizer with n-grams.
        min_keywords (int): Minimum number of keywords to return.
        
    Returns:
        List of tuples: Each tuple contains a keyword (or n-gram) and its corresponding TF-IDF score.
    """
    # Vectorize the entire abstract using the vectorizer (this includes n-grams)
    word_vectors = vectorizer.transform([abstract])  # Vectorize the full abstract
    
    # Get the TF-IDF scores for each n-gram in the abstract
    tfidf_scores = word_vectors.toarray().flatten()
    
    # Extract the feature names (words and n-grams)
    feature_names = vectorizer.get_feature_names_out()
    
    # Combine the features (words/n-grams) with their TF-IDF scores
    ranked_keywords = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)
    
    # Filter out very low-score keywords and return the top n keywords with their scores
    ranked_keywords_filtered = [(word, score) for word, score in ranked_keywords if score > 0]

    # Ensure we return at least `min_keywords` and their scores
    return ranked_keywords_filtered[:min_keywords]

# Example usage
if __name__ == "__main__":
    # Load the model and vectorizer
    model, vectorizer = load_model(model_filename='keyword_extraction_model2.pkl', vectorizer_filename='tfidf_vectorizer2.pkl')

    # New abstract for keyword extraction
    new_abstract = '''Online recruitment platforms typically employ Person-Job Fit models in the core service that automatically match suitable job seekers with appropriate job positions. While existing works leverage historical or contextual information, they often disregard a crucial aspect: job seekers’ social relationships in professional networks. This paper emphasizes the importance of incorporating professional networks into the Person-Job Fit model. Our innovative approach consists of two stages: (1) defining a Workplace Heterogeneous Information Network (WHIN) to capture heterogeneous knowledge, including professional connections and pre-training representations of various entities using a heterogeneous graph neural network; (2) designing a Contextual Social Attention Graph Neural Network (CSAGNN) that supplements users’ missing information with professional connections’ contextual information. We introduce a job-specific attention mechanism in CSAGNN to handle noisy professional networks, leveraging pre-trained entity representations from WHIN. We demonstrate the effectiveness of our approach through experimental evaluations conducted across three real-world recruitment datasets from LinkedIn, showing superior performance compared to baseline models.'''

    # Predict keywords with their TF-IDF scores
    predicted_keywords_with_scores = predict_keywords_from_model(new_abstract, model, vectorizer, 7)

    # Print predicted keywords with scores
    print("Predicted Keywords with Scores:")
    for keyword, score in predicted_keywords_with_scores:
        print(f"{keyword}: {score:.4f}")

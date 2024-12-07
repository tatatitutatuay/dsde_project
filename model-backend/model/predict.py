import joblib
import os

# Load the saved model and vectorizer from 'look_good_model' folder
def load_model(model_filename='keyword_extraction_model_last.pkl', vectorizer_filename='tfidf_vectorizer_last.pkl'):
    """
    Load the trained model and TF-IDF vectorizer from disk.
    
    Args:
        model_filename (str): The name of the model file to load.
        vectorizer_filename (str): The name of the vectorizer file to load.
        
    Returns:
        tuple: Loaded model and vectorizer.
    """
    model = joblib.load(os.path.join('model', model_filename))
    vectorizer = joblib.load(os.path.join('model', vectorizer_filename))
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

# Predict keywords from a new abstract using the trained model and vectorizer
def predict_keywords_from_abstract(abstract_text, min_keywords=5):
        """
        Function to be exported for use in other files.
        Predicts keywords from an abstract using the pre-trained model.
        
        Args:
            abstract_text (str): The input abstract text
            min_keywords (int): Minimum number of keywords to return
            
        Returns:
            List of tuples: Each tuple contains (keyword, score)
        """
        model, vectorizer = load_model()
        return predict_keywords_from_model(abstract_text, model, vectorizer, min_keywords)
    
_all_ = ['predict_keywords_from_abstract']
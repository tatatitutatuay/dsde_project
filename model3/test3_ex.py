import joblib
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from rake_nltk import Rake
import yake
import nltk

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')

# Load the saved model and vectorizer
def load_model(model_filename='keyword_extraction_model3.pkl', vectorizer_filename='tfidf_vectorizer3.pkl'):
    model = joblib.load(os.path.join('model3', model_filename))
    vectorizer = joblib.load(os.path.join('model3', vectorizer_filename))
    return model, vectorizer

# Preprocess the abstract (tokenization, stopword removal)
def preprocess_abstract(abstract):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(abstract.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)

# Combine RAKE, YAKE, and TF-IDF for multi-keyword extraction
def predict_keywords_combined(abstract, model, vectorizer, min_keywords=5):
    preprocessed_abstract = preprocess_abstract(abstract)

    # TF-IDF Predictions
    word_vectors = vectorizer.transform([preprocessed_abstract])
    tfidf_scores = word_vectors.toarray().flatten()
    feature_names = vectorizer.get_feature_names_out()
    tfidf_keywords = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)
    tfidf_keywords = [word for word, score in tfidf_keywords if score > 0][:min_keywords]

    # RAKE Predictions
    rake = Rake()
    rake.extract_keywords_from_text(abstract)
    rake_keywords = rake.get_ranked_phrases()[:min_keywords]

    # YAKE Predictions
    yake_extractor = yake.KeywordExtractor()
    yake_keywords = [kw for kw, score in yake_extractor.extract_keywords(abstract)][:min_keywords]

    # Combine and deduplicate keywords
    combined_keywords = list(set(tfidf_keywords + rake_keywords + yake_keywords))
    
    # Clean and rank keywords
    refined_keywords = clean_and_rank_keywords(combined_keywords)
    
    return refined_keywords

# Clean and rank the combined keywords
def clean_and_rank_keywords(keywords, min_length=2, max_length=5):
    # Filter keywords by length (number of words)
    filtered_keywords = [kw for kw in keywords if min_length <= len(kw.split()) <= max_length]
    
    # Deduplicate more effectively (case-insensitive)
    unique_keywords = list(set(map(str.lower, filtered_keywords)))
    
    # Sort by length or other criteria if needed
    unique_keywords.sort(key=lambda x: -len(x))
    
    return unique_keywords

# Load the model and vectorizer
model, vectorizer = load_model()

# Predict keywords for a new abstract
new_abstract = '''Online recruitment platforms typically employ Person-Job Fit models in the core service that automatically match suitable job seekers with appropriate job positions. While existing works leverage historical or contextual information, they often disregard a crucial aspect: job seekers’ social relationships in professional networks.'''

predicted_keywords = predict_keywords_combined(new_abstract, model, vectorizer, min_keywords=5)

# Print the refined predicted keywords
print("Refined Keywords:")
print(predicted_keywords[:5])

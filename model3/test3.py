import joblib
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from rake_nltk import Rake
import yake

# Load the saved model and vectorizer
def load_model(model_filename='keyword_extraction_model3.pkl', vectorizer_filename='tfidf_vectorizer3.pkl'):
    model = joblib.load(os.path.join('model3', model_filename))
    vectorizer = joblib.load(os.path.join('model3', vectorizer_filename))
    return model, vectorizer

# Function to preprocess a new abstract
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
    return combined_keywords

# Load the model and vectorizer
model, vectorizer = load_model()

# Predict keywords for a new abstract
new_abstract = '''Online recruitment platforms typically employ Person-Job Fit models in the core service that automatically match suitable job seekers with appropriate job positions. While existing works leverage historical or contextual information, they often disregard a crucial aspect: job seekers’ social relationships in professional networks. This paper emphasizes the importance of incorporating professional networks into the Person-Job Fit model. Our innovative approach consists of two stages: (1) defining a Workplace Heterogeneous Information Network (WHIN) to capture heterogeneous knowledge, including professional connections and pre-training representations of various entities using a heterogeneous graph neural network; (2) designing a Contextual Social Attention Graph Neural Network (CSAGNN) that supplements users’ missing information with professional connections’ contextual information. We introduce a job-specific attention mechanism in CSAGNN to handle noisy professional networks, leveraging pre-trained entity representations from WHIN. We demonstrate the effectiveness of our approach through experimental evaluations conducted across three real-world recruitment datasets from LinkedIn, showing superior performance compared to baseline models.'''

predicted_keywords = predict_keywords_combined(new_abstract, model, vectorizer, min_keywords=5)
# Print only the top 5 keywords
print("Top 5 Keywords:")
print(predicted_keywords)

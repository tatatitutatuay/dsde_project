import pandas as pd
import ast
import nltk
import joblib
import os
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from rake_nltk import Rake
import yake  # New addition for keyword extraction

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
file_path = 'data_preparation/given_data/data/data_noTHInAbstract.csv'
data = pd.read_csv(file_path)

# Ensure keywords are converted from string to list
data['keywords'] = data['keywords'].apply(ast.literal_eval)

# Preprocess abstracts: tokenize and generate word-level labels
def prepare_data(row):
    abstract = row['abstract'].lower()
    keywords = [kw.lower() for kw in row['keywords']]
    words = word_tokenize(abstract)
    labels = [1 if any(word in kw for kw in keywords) else 0 for word in words]
    return words, labels

# Apply preprocessing
data['processed'] = data.apply(prepare_data, axis=1)

# Flatten tokenized data for training
words, labels = zip(*[item for sublist in data['processed'] for item in zip(*sublist)])

# Filter out stopwords
stop_words = set(stopwords.words('english'))
filtered_words_labels = [(word, label) for word, label in zip(words, labels) if word not in stop_words]

# Separate words and labels again
filtered_words, filtered_labels = zip(*filtered_words_labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(filtered_words, filtered_labels, test_size=0.2, random_state=42)

# TF-IDF Vectorizer with n-grams (up to 5-grams)
vectorizer = TfidfVectorizer(
    ngram_range=(1, 5),
    stop_words='english',
    max_features=1000,
    min_df=0.0005,
    max_df=0.95
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a logistic regression classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
def save_model(model, vectorizer, model_filename='keyword_extraction_model3.pkl', vectorizer_filename='tfidf_vectorizer3.pkl'):
    if not os.path.exists('model3'):
        os.makedirs('model3')
    joblib.dump(model, os.path.join('model3', model_filename))
    joblib.dump(vectorizer, os.path.join('model3', vectorizer_filename))
    print(f"Model and vectorizer saved.")

save_model(model, vectorizer)

# Load the saved model and vectorizer
def load_model(model_filename='keyword_extraction_model3.pkl', vectorizer_filename='tfidf_vectorizer3.pkl'):
    model = joblib.load(os.path.join('model3', model_filename))
    vectorizer = joblib.load(os.path.join('mode l3', vectorizer_filename))
    return model, vectorizer

# Combine RAKE, YAKE, and TF-IDF for multi-keyword extraction
def predict_keywords_combined(abstract, model, vectorizer, min_keywords=5):
    # TF-IDF Predictions
    word_vectors = vectorizer.transform([abstract])
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

# Example usage for predicting keywords from a new abstract
new_abstract = '''Online recruitment platforms typically employ Person-Job Fit models in the core service that automatically match suitable job seekers with appropriate job positions. While existing works leverage historical or contextual information, they often disregard a crucial aspect: job seekers’ social relationships in professional networks.'''

predicted_keywords_combined = predict_keywords_combined(new_abstract, model, vectorizer, 5)

# Print combined predicted keywords
print("Predicted Keywords:")
print(predicted_keywords_combined)

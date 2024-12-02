import pandas as pd
import ast
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
import nltk
import joblib
import os

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
file_path = 'data_preparation/given_data/data/data_noTHInAbstract.csv'  # Update with your file path if needed
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
words = []
labels = []
for tokens, lbls in data['processed']:
    words.extend(tokens)
    labels.extend(lbls)

# Filter out stopwords
stop_words = set(stopwords.words('english'))
filtered_words_labels = [(word, label) for word, label in zip(words, labels) if word not in stop_words]

# Separate words and labels again
filtered_words, filtered_labels = zip(*filtered_words_labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(filtered_words, filtered_labels, test_size=0.2, random_state=42)

# Vectorize the words using TF-IDF with n-grams (e.g., bigrams or trigrams)
vectorizer = TfidfVectorizer(ngram_range=(1, 4), stop_words='english')  # Remove common English stopwords
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a logistic regression classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Save the model and vectorizer to 'look_good_model' folder
def save_model(model, vectorizer, model_filename='keyword_extraction_model.pkl', vectorizer_filename='tfidf_vectorizer.pkl'):
    # Create the 'look_good_model' directory if it doesn't exist
    if not os.path.exists('look_good_model'):
        os.makedirs('look_good_model')
    
    # Save the model and vectorizer inside the 'look_good_model' folder
    joblib.dump(model, os.path.join('look_good_model', model_filename))
    joblib.dump(vectorizer, os.path.join('look_good_model', vectorizer_filename))
    print(f"Model and vectorizer saved in 'look_good_model/{model_filename}' and 'look_good_model/{vectorizer_filename}'")

# Example usage to save
save_model(model, vectorizer, model_filename='keyword_extraction_model.pkl', vectorizer_filename='tfidf_vectorizer.pkl')

# Load the saved model and vectorizer from 'look_good_model' folder
def load_model(model_filename='keyword_extraction_model.pkl', vectorizer_filename='tfidf_vectorizer.pkl'):
    model = joblib.load(os.path.join('look_good_model', model_filename))
    vectorizer = joblib.load(os.path.join('look_good_model', vectorizer_filename))
    return model, vectorizer

# Example usage to load
model, vectorizer = load_model(model_filename='keyword_extraction_model.pkl', vectorizer_filename='tfidf_vectorizer.pkl')

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

# Example usage for predicting keywords with their scores from a new abstract
new_abstract = '''Based on the covariant underdamped and overdamped Langevin equations
with Stratonovich coupling to multiplicative noises and the associated
Fokker-Planck equations on Riemannian manifold, we present
the first law of stochastic thermodynamics on the trajectory level.
The corresponding fluctuation theorems are also
established, with the total entropy production of the Brownian particle
and the heat reservoir playing the role of dissipation function.'''

# Predict keywords with their TF-IDF scores
predicted_keywords_with_scores = predict_keywords_from_model(new_abstract, model, vectorizer, 5)

# Print predicted keywords with scores
print("Predicted Keywords with Scores:")
for keyword, score in predicted_keywords_with_scores:
    print(f"{keyword}: {score:.4f}")
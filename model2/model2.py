import pandas as pd
import ast
import nltk
import os
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Load the dataset
file_path = 'data_preparation/given_data/data/data_noTHInAbstract.csv'  # Update with your file path if needed
data = pd.read_csv(file_path)

# Ensure keywords are converted from string to list
data['keywords'] = data['keywords'].apply(ast.literal_eval)

# Preprocess abstracts: tokenize, lemmatize, and label words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def prepare_data(row):
    abstract = row['abstract'].lower()
    keywords = [kw.lower() for kw in row['keywords']]
    words = word_tokenize(abstract)
    
    # Lemmatize and remove stopwords
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and word.isalnum()]
    
    # Generate labels
    labels = [1 if any(word in kw for kw in keywords) else 0 for word in lemmatized_words]
    return lemmatized_words, labels

# Apply preprocessing
data['processed'] = data.apply(prepare_data, axis=1)

# Flatten tokenized data for training
words, labels = [], []
for tokens, lbls in data['processed']:
    words.extend(tokens)
    labels.extend(lbls)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(words, labels, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(ngram_range=(1, 4), stop_words='english')  # Remove common English stopwords

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_tfidf, y_train)

# Train a logistic regression classifier
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_resampled, y_resampled)

# Predict on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
def save_model(model, vectorizer, model_filename='keyword_extraction_model2.pkl', vectorizer_filename='tfidf_vectorizer2.pkl'):
    if not os.path.exists('model2'):
        os.makedirs('model2')
    joblib.dump(model, os.path.join('model2', model_filename))
    joblib.dump(vectorizer, os.path.join('model2', vectorizer_filename))
    print(f"Model and vectorizer saved in 'model2/{model_filename}' and 'model2/{vectorizer_filename}'")

save_model(model, vectorizer)

# Load the model and vectorizer
def load_model(model_filename='keyword_extraction_model2.pkl', vectorizer_filename='tfidf_vectorizer2.pkl'):
    model = joblib.load(os.path.join('model2', model_filename))
    vectorizer = joblib.load(os.path.join('model2', vectorizer_filename))
    return model, vectorizer

model, vectorizer = load_model()

# Predict keywords from a new abstract
def predict_keywords_from_model(abstract, model, vectorizer, min_keywords=5):
    # Preprocess the abstract
    abstract = abstract.lower()
    words = word_tokenize(abstract)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and word.isalnum()]
    
    # Vectorize the processed abstract
    word_vectors = vectorizer.transform([' '.join(lemmatized_words)])
    
    # Get TF-IDF scores
    tfidf_scores = word_vectors.toarray().flatten()
    feature_names = vectorizer.get_feature_names_out()
    
    # Rank keywords based on scores
    ranked_keywords = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)
    
    # Return top n keywords with scores
    return [(word, score) for word, score in ranked_keywords if score > 0][:min_keywords]

# Example prediction
new_abstract = '''Based on the covariant underdamped and overdamped Langevin equations
with Stratonovich coupling to multiplicative noises and the associated
Fokker-Planck equations on Riemannian manifold, we present
the first law of stochastic thermodynamics on the trajectory level.
The corresponding fluctuation theorems are also
established, with the total entropy production of the Brownian particle
and the heat reservoir playing the role of dissipation function.'''

predicted_keywords_with_scores = predict_keywords_from_model(new_abstract, model, vectorizer, 5)

# Print predicted keywords with scores
print("Predicted Keywords with Scores:")
for keyword, score in predicted_keywords_with_scores:
    print(f"{keyword}: {score:.4f}")

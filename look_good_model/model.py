import pandas as pd
import ast
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
import joblib
import os

# Download required resources
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load the dataset
file_path = './data_preparation/given_data/data/data_noTHInAbstract.csv'  # Update with your file path if needed
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
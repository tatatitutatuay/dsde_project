import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import ast  # To safely evaluate string representations of lists

# Load your CSV file
df = pd.read_csv('dsde_project\data_preparation\given_data\more_filtered_scopus_data.csv')  # Replace 'your_data.csv' with your file path

# Convert keyword strings to lists (assuming they are stored as string representations of lists)
df['keywords'] = df['keywords'].apply(ast.literal_eval)

# Prepare features (abstracts) and labels (keywords)
X = df['abstract']
y = df['keywords']

# Binarize the labels (convert keyword lists to a binary matrix)
mlb = MultiLabelBinarizer()
y_binarized = mlb.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binarized, test_size=0.2, random_state=42)

# Convert abstracts to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Limit to 5000 features to reduce dimensionality
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a logistic regression model with OneVsRestClassifier for multi-label classification
model = OneVsRestClassifier(LogisticRegression(max_iter=500))
model.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
f1 = f1_score(y_test, y_pred, average='micro')

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Function to predict keywords for a new abstract
def extract_keywords(new_abstract):
    abstract_tfidf = tfidf_vectorizer.transform([new_abstract])
    prediction = model.predict(abstract_tfidf)
    predicted_keywords = mlb.inverse_transform(prediction)
    return predicted_keywords[0] if predicted_keywords else []

# Example usage
new_abstract = """This paper presents a new deep learning approach for natural language processing
        with applications in text classification and sentiment analysis. The proposed
        method combines transformer architectures with attention mechanisms to improve
        performance on various NLP tasks."""  # Replace this with any abstract text
print("Extracted Keywords:", extract_keywords(new_abstract))

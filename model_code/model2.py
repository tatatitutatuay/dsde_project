import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
import joblib
import os

### 5000 rows of data is used for training the model
### scoring is done on the model

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def preprocess_text(text):
    """Enhanced text preprocessing."""
    # Convert to lowercase
    text = text.lower()
    
    # Replace hyphens with spaces to properly split hyphenated terms
    text = text.replace('-', ' ')
    
    # Remove special characters but keep internal apostrophes
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Split into words
    words = text.split()
    
    # Extended stopwords to remove common non-keyword terms
    custom_stopwords = set(stopwords.words('english') + [
        'study', 'research', 'paper', 'using', 'based', 'method', 'results', 'used',
        'proposed', 'shows', 'within', 'including', 'find', 'found', 'show', 'demonstrates',
        'presents', 'discusses', 'investigated', 'analyzed', 'examined'
    ])
    
    # Keep only meaningful words (longer than 2 characters and not in stopwords)
    words = [word for word in words if len(word) > 2 and word not in custom_stopwords]
    
    return ' '.join(words)

def create_keyword_training_data(data):
    """Create enhanced training data with better keyword identification."""
    data['clean_text'] = data['text'].apply(preprocess_text)
    
    # Process keywords
    if isinstance(data['keywords'].iloc[0], str):
        data['keywords'] = data['keywords'].apply(lambda x: set(re.findall(r'\b\w+\b', x.lower())))
    else:
        data['keywords'] = data['keywords'].apply(set)

    X_texts = []
    y_labels = []
    
    for _, row in data.iterrows():
        # Get unique words and bigrams from the text
        words = set(row['clean_text'].split())
        
        # Add bigrams to consider phrases
        text_tokens = row['clean_text'].split()
        bigrams = set([' '.join(text_tokens[i:i+2]) for i in range(len(text_tokens)-1)])
        
        # Create training examples for both unigrams and bigrams
        for term in words.union(bigrams):
            X_texts.append(row['clean_text'])
            y_labels.append(1 if term in row['keywords'] else 0)
    
    return X_texts, y_labels

def train_keyword_model(data):
    """Train an improved keyword extraction model."""
    # Combine text fields with more weight on title
    data['text'] = data['title'] + '. ' + data['title'] + '. ' + data['abstract']
    
    X_texts, y_labels = create_keyword_training_data(data)
    
    # Enhanced TF-IDF vectorizer with better parameters
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        stop_words='english',
        max_features=10000
    )
    
    X = vectorizer.fit_transform(X_texts)
    X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.2, random_state=42)
    
    model = LogisticRegression(
        class_weight='balanced',
        C=1.0,
        max_iter=200,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("Model Performance:")
    print(classification_report(y_test, y_pred))
    
    return model, vectorizer

def extract_keywords_with_scores(text, model, vectorizer, threshold=0.7, max_keywords=7):
    """Extract keywords with probability scores above a threshold."""
    clean_text = preprocess_text(text)
    words = set(clean_text.split())
    
    # Generate bigrams from the text
    tokens = clean_text.split()
    bigrams = [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)]
    
    # Combine unigrams and bigrams
    terms = list(words) + bigrams
    
    # Prepare for prediction
    X_texts = [clean_text] * len(terms)
    X = vectorizer.transform(X_texts)
    
    # Get probabilities and combine with terms
    probas = model.predict_proba(X)[:, 1]
    term_probas = list(zip(terms, probas))
    
    # Sort by probability and filter duplicates
    seen = set()
    keywords = []
    for term, prob in sorted(term_probas, key=lambda x: x[1], reverse=True):
        if prob >= threshold and term not in seen and len(keywords) < max_keywords:
            keywords.append((term, prob))
            seen.update(term.split())
    
    return keywords

def save_model(model, vectorizer, model_dir='models'):
    """Save the model and vectorizer."""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    joblib.dump(model, os.path.join(model_dir, 'keyword_model2.joblib'))
    joblib.dump(vectorizer, os.path.join(model_dir, 'vectorizer2.joblib'))
    print(f"Model and vectorizer saved in {model_dir} directory")

def load_model(model_dir='models'):
    """Load the model and vectorizer."""
    model_path = os.path.join(model_dir, 'keyword_model2.joblib')
    vectorizer_path = os.path.join(model_dir, 'vectorizer2.joblib')
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Model or vectorizer not found. Please train the model first.")
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

if __name__ == "__main__":
    # Load and prepare data
    data = pd.read_csv('data_preparation/given_data/filtered_scopus_data.csv')
    data = data[['abstract', 'keywords', 'title']].dropna()
    data = data.head(5000)
    
    # Train and save model
    model, vectorizer = train_keyword_model(data)
    save_model(model, vectorizer)
    
    # Test on the example text
    test_text = """This research investigates the integration of renewable energy sources within smart grid systems, emphasizing optimization strategies for energy distribution and storage. We analyze the implementation of advanced control algorithms for managing distributed energy resources, including solar, wind, and battery storage systems. The study examines real-time demand response mechanisms and their impact on grid stability and energy efficiency. Our findings indicate that machine learning-based control systems can significantly improve grid reliability while maximizing renewable energy utilization. The research also addresses challenges in grid modernization, energy storage scalability, and the economic implications of renewable integration."""
    
    # Extract keywords with confidence scores
    keywords_with_scores = extract_keywords_with_scores(test_text, model, vectorizer, threshold=0.7)
    print("\nExtracted keywords with confidence scores:")
    for keyword, score in keywords_with_scores:
        print(f"{keyword}: {score:.3f}")
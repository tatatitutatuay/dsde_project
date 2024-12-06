# pip install scikit-learn pandas nltk tqdm joblib

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import time
import joblib
from tqdm import tqdm

class KeywordExtractor:
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        """
        Initialize the keyword extractor with configurable parameters
        """
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
        self.classifier = None
        self.mlb = MultiLabelBinarizer()
        self.best_threshold = 0.5
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """
        Preprocess text data
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        
        # Part of speech tagging - keep only nouns and adjectives
        pos_tags = nltk.pos_tag(tokens)
        tokens = [word for word, pos in pos_tags if pos.startswith(('NN', 'JJ'))]
        
        return ' '.join(tokens)
    
    def prepare_data(self, abstracts, keywords):
        """
        Prepare data for training
        """
        # Preprocess abstracts
        print("Preprocessing abstracts...")
        processed_abstracts = [self.preprocess_text(abs_) for abs_ in tqdm(abstracts)]
        
        # Convert keywords to list format if they're strings
        processed_keywords = []
        for kw in keywords:
            if isinstance(kw, str):
                # Handle different keyword separators
                kw = kw.replace(';', ',')
                kw_list = [k.strip().lower() for k in kw.split(',')]
                processed_keywords.append(kw_list)
            else:
                processed_keywords.append([k.lower() for k in kw])
        
        return processed_abstracts, processed_keywords
    
    def create_vectorizer(self):
        """
        Create and configure TF-IDF vectorizer
        """
        return TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
    
    def train(self, abstracts, keywords, test_size=0.2, random_state=42):
        """
        Train the keyword extraction model with optimization
        """
        print("Starting training pipeline...")
        
        # Prepare data
        processed_abstracts, processed_keywords = self.prepare_data(abstracts, keywords)
        
        # Create and fit vectorizer
        print("Vectorizing abstracts...")
        self.vectorizer = self.create_vectorizer()
        X = self.vectorizer.fit_transform(processed_abstracts)
        
        # Transform keywords to binary matrix
        print("Transforming keywords...")
        y = self.mlb.fit_transform(processed_keywords)
        
        # Check if y contains only one class
        if len(np.unique(y)) == 1:
            raise ValueError("The target variable contains only one class. Please ensure that the dataset has more than one class for classification.")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Create base classifier
        base_classifier = LogisticRegression(
            solver='saga',
            multi_class='ovr',
            random_state=random_state,
            max_iter=500
        )
        
        # Create multi-output classifier
        self.classifier = MultiOutputClassifier(base_classifier, n_jobs=-1)
        
        # Define parameter grid for optimization
        param_grid = {
            'estimator__C': [0.1, 1.0, 10.0],
            'estimator__max_iter': [100, 200],
            'estimator__class_weight': [None, 'balanced']
        }
        
        # Perform grid search
        print("Performing grid search...")
        grid_search = GridSearchCV(
            self.classifier,
            param_grid,
            cv=3,
            scoring='f1_macro',
            n_jobs=-1
        )
        
        # Fit model with progress bar
        with tqdm(total=1) as pbar:
            grid_search.fit(X_train, y_train)
            pbar.update(1)
        
        # Get best model
        self.classifier = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Evaluate on test set
        print("\nEvaluating model...")
        self.evaluate(X_test, y_test)
        
        return self
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model's performance
        """
        # Predict on test set
        y_pred = self.classifier.predict(X_test)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, 
            y_pred, 
            average='macro'
        )
        
        print("\nModel Performance:")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")
        
        return precision, recall, f1
    
    def predict(self, abstract, top_k=5):
        """
        Predict keywords for a new abstract
        """
        # Preprocess abstract
        processed_abstract = self.preprocess_text(abstract)
        
        # Vectorize
        X = self.vectorizer.transform([processed_abstract])
        
        # Get probability predictions
        y_pred_proba = self.classifier.predict_proba(X)
        
        # Get top k keywords based on probability
        top_k_indices = np.argsort(y_pred_proba[0])[::-1][:top_k]
        top_keywords = self.mlb.classes_[top_k_indices]
        
        return top_keywords

    
    def save_model(self, filepath):
        """
        Save the trained model and necessary components
        """
        model_components = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'mlb': self.mlb,
            'best_threshold': self.best_threshold,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range
        }
        joblib.dump(model_components, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a trained model
        """
        model_components = joblib.load(filepath)
        
        instance = cls(
            max_features=model_components['max_features'],
            ngram_range=model_components['ngram_range']
        )
        
        instance.vectorizer = model_components['vectorizer']
        instance.classifier = model_components['classifier']
        instance.mlb = model_components['mlb']
        instance.best_threshold = model_components['best_threshold']
        
        return instance

# Example usage:
def main():
    # Load your data
    df = pd.read_csv('C:/Users/USER/Desktop/my-git/dsde_project/data_preparation/given_data/data/data_noTHInAbstract.csv')
    
    # Initialize extractor
    extractor = KeywordExtractor(max_features=5000, ngram_range=(1, 2))
    
    # Train model
    extractor.train(
        abstracts=df['abstract'].values,
        keywords=df['keywords'].values
    )
    
    # Save model
    extractor.save_model('keyword_extractor_model.joblib')
    
    # Example prediction
    test_abstract = "Online recruitment platforms typically employ Person-Job Fit models in the core service that automatically match suitable job seekers with appropriate job positions. While existing works leverage historical or contextual information, they often disregard a crucial aspect: job seekers’ social relationships in professional networks. This paper emphasizes the importance of incorporating professional networks into the Person-Job Fit model. Our innovative approach consists of two stages: (1) defining a Workplace Heterogeneous Information Network (WHIN) to capture heterogeneous knowledge, including professional connections and pre-training representations of various entities using a heterogeneous graph neural network; (2) designing a Contextual Social Attention Graph Neural Network (CSAGNN) that supplements users’ missing information with professional connections’ contextual information. We introduce a job-specific attention mechanism in CSAGNN to handle noisy professional networks, leveraging pre-trained entity representations from WHIN. We demonstrate the effectiveness of our approach through experimental evaluations conducted across three real-world recruitment datasets from LinkedIn, showing superior performance compared to baseline models."
    # The actual keyword: ['Person-Job', 'Fit,', 'Heterogeneous', 'Information', 'Network,', 'Graph', 'Neural', 'Network']
    predicted_keywords = extractor.predict(test_abstract, top_k=5)
    print(f"Predicted keywords: {predicted_keywords}")

if __name__ == "__main__":
    main()
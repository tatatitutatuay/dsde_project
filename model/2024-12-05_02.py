# pip install scikit-learn pandas nltk tqdm joblib psutil
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
import psutil
import multiprocessing

class KeywordExtractor:
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        """
        Initialize the keyword extractor with configurable parameters
        """
        # Calculate optimal number of jobs based on CPU cores
        self.n_jobs = min(multiprocessing.cpu_count() - 1, 8)  # Leave one core free
        self.chunk_size = 10000  # Chunk size for batch processing
        
        # Download required NLTK data
        for resource in ['punkt', 'stopwords', 'averaged_perceptron_tagger']:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                nltk.download(resource)
        
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
        self.classifier = None
        self.mlb = MultiLabelBinarizer()
        self.best_threshold = 0.5
        self.stop_words = set(stopwords.words('english'))
        
        # Memory management
        self.mem = psutil.virtual_memory()
        self.batch_size = self._calculate_batch_size()
    
    def _calculate_batch_size(self):
        """Calculate optimal batch size based on available memory"""
        available_mem = self.mem.available
        estimated_sample_size = 1024 * 1024  # Estimated size per sample in bytes
        return min(max(1000, available_mem // (estimated_sample_size * 2)), 5000)
    
    def preprocess_text(self, text):
        """
        Preprocess text data with optimized tokenization
        """
        # Convert to lowercase and remove special characters in one pass
        text = re.sub(r'[^\w\s]|\d+', ' ', text.lower())
        
        # Tokenize and remove stopwords in one pass
        tokens = [t for t in word_tokenize(text) if t not in self.stop_words and len(t) > 2]
        
        # Batch POS tagging for efficiency
        pos_tags = nltk.pos_tag(tokens)
        return ' '.join(word for word, pos in pos_tags if pos.startswith(('NN', 'JJ')))
    
    def _batch_preprocess(self, texts):
        """Process texts in batches to manage memory"""
        processed_texts = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            processed_batch = joblib.Parallel(n_jobs=self.n_jobs)(
                joblib.delayed(self.preprocess_text)(text) for text in batch
            )
            processed_texts.extend(processed_batch)
        return processed_texts
    
    def prepare_data(self, abstracts, keywords):
        """
        Prepare data for training with parallel processing
        """
        print(f"Preprocessing abstracts using {self.n_jobs} CPU cores...")
        processed_abstracts = self._batch_preprocess(abstracts)
        
        # Process keywords in parallel
        def process_keyword(kw):
            if isinstance(kw, str):
                kw = kw.replace(';', ',')
                return [k.strip().lower() for k in kw.split(',')]
            return [k.lower() for k in kw]
        
        processed_keywords = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(process_keyword)(kw) for kw in keywords
        )
        
        return processed_abstracts, processed_keywords
    
    def create_vectorizer(self):
        """
        Create and configure TF-IDF vectorizer with CPU optimization
        """
        return TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words='english',
            min_df=2,
            max_df=0.95,
            dtype=np.float32,  # Use float32 to reduce memory usage
            norm='l2',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True  # Apply sublinear scaling for better feature scaling
        )
    
    def train(self, abstracts, keywords, test_size=0.2, random_state=42):
        """
        Train the keyword extraction model with CPU optimization
        """
        print(f"Starting training pipeline using {self.n_jobs} CPU cores...")
        
        # Prepare data in batches
        processed_abstracts, processed_keywords = self.prepare_data(abstracts, keywords)
        
        # Create and fit vectorizer
        print("Vectorizing abstracts...")
        self.vectorizer = self.create_vectorizer()
        X = self.vectorizer.fit_transform(processed_abstracts)
        
        # Transform keywords to binary matrix
        print("Transforming keywords...")
        y = self.mlb.fit_transform(processed_keywords)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Configure base classifier for CPU optimization
        base_classifier = LogisticRegression(
            solver='saga',
            multi_class='ovr',
            random_state=random_state,
            max_iter=200,
            tol=1e-4,
            n_jobs=1  # Set to 1 as we're using MultiOutputClassifier for parallelization
        )
        
        # Create multi-output classifier with optimal CPU utilization
        self.classifier = MultiOutputClassifier(
            base_classifier,
            n_jobs=self.n_jobs
        )
        
        # Define parameter grid for optimization
        param_grid = {
            'estimator__C': [0.1, 1.0, 10.0],
            'estimator__class_weight': [None, 'balanced']
        }
        
        # Perform grid search with CPU optimization
        print(f"Performing grid search with {self.n_jobs} parallel jobs...")
        grid_search = GridSearchCV(
            self.classifier,
            param_grid,
            cv=3,
            scoring='f1_macro',
            n_jobs=self.n_jobs,
            verbose=1
        )
        
        # Fit model with progress tracking
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
    
    def predict(self, abstract, top_k=5):
        """
        Predict keywords with batch processing
        """
        # Preprocess abstract
        processed_abstract = self.preprocess_text(abstract)
        
        # Vectorize
        X = self.vectorizer.transform([processed_abstract])
        
        # Get probability predictions in batches
        y_pred_proba = self.classifier.predict_proba(X)
        
        # Get top k keywords based on probability
        keywords = []
        for i, estimator_proba in enumerate(y_pred_proba):
            proba = estimator_proba[0]
            if proba.max() > self.best_threshold:
                keywords.append(self.mlb.classes_[i])
        
        return sorted(set(keywords))[:top_k]

    def save_model(self, filepath):
        """Save the model with compression"""
        model_components = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'mlb': self.mlb,
            'best_threshold': self.best_threshold,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range
        }
        joblib.dump(model_components, filepath, compress=3)
        print(f"Model saved to {filepath}")

# Example usage
def main():
    # Load data in chunks to manage memory
    df = pd.read_csv('data_noTHInAbstract.csv', chunksize=10000)
    data = pd.concat(df)

    # Initialize extractor with CPU optimization
    extractor = KeywordExtractor(max_features=5000, ngram_range=(1, 2))

    # Train model
    extractor.train(
        abstracts=data['abstract'].values,
        keywords=data['keywords'].values
    )

    # Save model with compression
    extractor.save_model('keyword_extractor_model.joblib')

    # Example prediction
    test_abstract = "Online recruitment platforms typically employ Person-Job Fit models in the core service that automatically match suitable job seekers with appropriate job positions. While existing works leverage historical or contextual information, they often disregard a crucial aspect: job seekers’ social relationships in professional networks. This paper emphasizes the importance of incorporating professional networks into the Person-Job Fit model. Our innovative approach consists of two stages: (1) defining a Workplace Heterogeneous Information Network (WHIN) to capture heterogeneous knowledge, including professional connections and pre-training representations of various entities using a heterogeneous graph neural network; (2) designing a Contextual Social Attention Graph Neural Network (CSAGNN) that supplements users’ missing information with professional connections’ contextual information. We introduce a job-specific attention mechanism in CSAGNN to handle noisy professional networks, leveraging pre-trained entity representations from WHIN. We demonstrate the effectiveness of our approach through experimental evaluations conducted across three real-world recruitment datasets from LinkedIn, showing superior performance compared to baseline models."
    # The actual keyword: ['Person-Job', 'Fit,', 'Heterogeneous', 'Information', 'Network,', 'Graph', 'Neural',
    predicted_keywords = extractor.predict(test_abstract, top_k=5)
    print(f"Predicted keywords: {predicted_keywords}")

if __name__ == "__main__":
    main()

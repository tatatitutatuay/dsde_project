import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import joblib
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

import nltk
nltk.download('averaged_perceptron_tagger')


class PyTorchKeywordExtractor:
    def __init__(self, max_features=5000, ngram_range=(1, 2), hidden_dim=128):
        """
        Initialize the PyTorch-based keyword extractor with configurable parameters
        """
        nltk.download('punkt')
        nltk.download('stopwords')
        
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.hidden_dim = hidden_dim
        self.vectorizer = None
        self.mlb = MultiLabelBinarizer()
        self.stop_words = set(stopwords.words('english'))
        self.best_threshold = 0.5
        
    def preprocess_text(self, text):
        """
        Preprocess text data
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        
        pos_tags = nltk.pos_tag(tokens)
        tokens = [word for word, pos in pos_tags if pos.startswith(('NN', 'JJ'))]
        
        return ' '.join(tokens)
    
    def prepare_data(self, abstracts, keywords):
        """
        Prepare data for training
        """
        print("Preprocessing abstracts...")
        processed_abstracts = [self.preprocess_text(abs_) for abs_ in tqdm(abstracts)]
        
        processed_keywords = []
        for kw in keywords:
            if isinstance(kw, str):
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
    
    def train(self, abstracts, keywords, test_size=0.2, random_state=42, epochs=10, batch_size=64):
        """
        Train the keyword extraction model with PyTorch
        """
        print("Starting training pipeline...")

        processed_abstracts, processed_keywords = self.prepare_data(abstracts, keywords)

        print("Vectorizing abstracts...")
        self.vectorizer = self.create_vectorizer()
        X = self.vectorizer.fit_transform(processed_abstracts).toarray()

        print("Transforming keywords...")
        y = self.mlb.fit_transform(processed_keywords)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Convert to torch tensors
        train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        # Initialize model
        model = KeywordExtractionModel(X_train.shape[1], self.hidden_dim, y_train.shape[1])

        # Loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters())

        # Training loop
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in tqdm(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

        # Evaluate on test set
        self.evaluate(model, test_loader)

        # Save model
        self.save_model(model, 'pytorch_keyword_extractor.pth')

        return X_train, X_test, y_train, y_test  # Return the data splits

        
    def evaluate(self, model, test_loader):
        """
        Evaluate the PyTorch model's performance
        """
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                preds = torch.sigmoid(outputs).numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        
        precision, recall, f1, _ = precision_recall_fscore_support(np.array(all_labels), np.array(all_preds) > self.best_threshold, average='macro')
        
        print("\nModel Performance:")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")
        
    def predict(self, abstract, model, top_k=5):
        """
        Predict keywords for a new abstract using the trained model
        """
        processed_abstract = self.preprocess_text(abstract)
        X = self.vectorizer.transform([processed_abstract]).toarray()
        
        model.eval()
        with torch.no_grad():
            y_pred_proba = torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()
        
        keywords = []
        for i, proba in enumerate(y_pred_proba[0]):
            if proba > self.best_threshold:
                keywords.append(self.mlb.classes_[i])
        
        return sorted(set(keywords))[:top_k]
    
    def save_model(self, model, filepath):
        """
        Save the trained PyTorch model and necessary components
        """
        torch.save(model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath, input_dim, hidden_dim, output_dim):
        """
        Load a trained PyTorch model
        """
        model = KeywordExtractionModel(input_dim, hidden_dim, output_dim)
        model.load_state_dict(torch.load(filepath))
        model.eval()
        return model


class KeywordExtractionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=10):
        """
        Initialize a simple feedforward neural network for multi-label classification
        """
        super(KeywordExtractionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Example usage:
def main():
    # Load your data
    df = pd.read_csv('C:/Users/USER/Desktop/my-git/dsde_project/data_preparation/given_data/data/data_noTHInAbstract.csv')
    
    # Initialize extractor
    extractor = PyTorchKeywordExtractor(max_features=5000, ngram_range=(1, 2))
    
    # Train model and get X_train and y_train dimensions
    X_train, X_test, y_train, y_test = extractor.train(
        abstracts=df['abstract'].values,
        keywords=df['keywords'].values
    )
    
    # Example prediction
    test_abstract = "Online recruitment platforms typically employ Person-Job Fit models in the core service that automatically match suitable job seekers with appropriate job positions..."
    model = PyTorchKeywordExtractor.load_model(
        'pytorch_keyword_extractor.pth',
        input_dim=X_train.shape[1],  # Ensure the correct input dimension
        hidden_dim=extractor.hidden_dim,
        output_dim=y_train.shape[1]  # Ensure the correct output dimension
    )
    predicted_keywords = extractor.predict(test_abstract, model, top_k=5)
    print(f"Predicted keywords: {predicted_keywords}")


if __name__ == "__main__":
    main()

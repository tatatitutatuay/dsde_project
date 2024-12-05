# pip install torch torchtext pandas nltk tqdm joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from torchtext.vocab import build_vocab_from_iterator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import time
import joblib
from tqdm import tqdm

class KeywordDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=512):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Convert tokens to indices
        tokens = [self.vocab[token] for token in text.split()]
        # Pad or truncate
        if len(tokens) < self.max_length:
            tokens = tokens + [self.vocab['<pad>']] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        return torch.tensor(tokens), torch.tensor(self.labels[idx])

class KeywordExtractorModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_size * 2, hidden_size * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 2, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)  # (batch_size, sequence_length, hidden_size)
        # Transpose for Conv1d
        embedded = embedded.transpose(1, 2)  # (batch_size, hidden_size, sequence_length)
        encoded = self.encoder(embedded)  # (batch_size, hidden_size * 4, 1)
        encoded = encoded.squeeze(-1)  # (batch_size, hidden_size * 4)
        return self.classifier(encoded)

class GPUKeywordExtractor:
    def __init__(self, max_features=5000, batch_size=32, hidden_size=256):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')
        
        self.max_features = max_features
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stop_words = set(stopwords.words('english'))
        self.vocab = None
        self.model = None
        self.mlb = MultiLabelBinarizer()

    def preprocess_text(self, text):
        """Preprocess text data"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        pos_tags = nltk.pos_tag(tokens)
        tokens = [word for word, pos in pos_tags if pos.startswith(('NN', 'JJ'))]
        return ' '.join(tokens)

    def prepare_data(self, abstracts, keywords):
        print("Preprocessing data...")
        processed_abstracts = [self.preprocess_text(abs_) for abs_ in tqdm(abstracts)]
        
        processed_keywords = []
        for kw in keywords:
            if isinstance(kw, str):
                kw = kw.replace(';', ',')
                kw_list = [k.strip().lower() for k in kw.split(',')]
                processed_keywords.append(kw_list)
            else:
                processed_keywords.append([k.lower() for k in kw])
        
        # Build vocabulary
        def yield_tokens(texts):
            for text in texts:
                yield text.split()
        
        self.vocab = build_vocab_from_iterator(
            yield_tokens(processed_abstracts),
            specials=['<unk>', '<pad>'],
            max_tokens=self.max_features
        )
        self.vocab.set_default_index(self.vocab['<unk>'])
        
        return processed_abstracts, processed_keywords

    def train(self, abstracts, keywords, epochs=10, test_size=0.2, random_state=42):
        processed_abstracts, processed_keywords = self.prepare_data(abstracts, keywords)
        
        # Transform keywords to binary matrix
        y = self.mlb.fit_transform(processed_keywords)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_abstracts, y, test_size=test_size, random_state=random_state
        )
        
        # Create datasets
        train_dataset = KeywordDataset(X_train, y_train, self.vocab)
        test_dataset = KeywordDataset(X_test, y_test, self.vocab)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize model
        self.model = KeywordExtractorModel(
            len(self.vocab), 
            self.hidden_size, 
            y.shape[1]
        ).to(self.device)
        
        # Initialize optimizer and loss function
        optimizer = optim.AdamW(self.model.parameters(), lr=2e-4)
        criterion = nn.BCELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2)
        
        # Training loop
        best_f1 = 0
        print("Starting training...")
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for batch_texts, batch_labels in pbar:
                batch_texts = batch_texts.to(self.device)
                batch_labels = batch_labels.float().to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_texts)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': train_loss / (pbar.n + 1)})
            
            # Evaluation
            precision, recall, f1 = self.evaluate(test_loader)
            print(f"\nEpoch {epoch+1} - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
            
            scheduler.step(f1)
            if f1 > best_f1:
                best_f1 = f1
                torch.save(self.model.state_dict(), 'best_model.pt')
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pt'))
        return self

    def evaluate(self, dataloader):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_texts, batch_labels in dataloader:
                batch_texts = batch_texts.to(self.device)
                outputs = self.model(batch_texts)
                predictions = (outputs > 0.5).float().cpu().numpy()
                all_preds.extend(predictions)
                all_labels.extend(batch_labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        precision = np.mean(np.sum((all_preds == 1) & (all_labels == 1), axis=1) / (np.sum(all_preds == 1, axis=1) + 1e-10))
        recall = np.mean(np.sum((all_preds == 1) & (all_labels == 1), axis=1) / (np.sum(all_labels == 1, axis=1) + 1e-10))
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        return precision, recall, f1

    def predict(self, abstract, top_k=5):
        self.model.eval()
        processed_abstract = self.preprocess_text(abstract)
        
        # Create dataset for single abstract
        dataset = KeywordDataset([processed_abstract], [[0] * len(self.mlb.classes_)], self.vocab)
        dataloader = DataLoader(dataset, batch_size=1)
        
        with torch.no_grad():
            batch_texts, _ = next(iter(dataloader))
            batch_texts = batch_texts.to(self.device)
            outputs = self.model(batch_texts)
            probs = outputs[0].cpu().numpy()
        
        # Get top k keywords
        top_indices = np.argsort(probs)[-top_k:]
        return [self.mlb.classes_[idx] for idx in top_indices]

    def save_model(self, filepath):
        model_components = {
            'vocab': self.vocab,
            'model_state': self.model.state_dict(),
            'mlb': self.mlb,
            'max_features': self.max_features,
            'hidden_size': self.hidden_size
        }
        torch.save(model_components, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath):
        model_components = torch.load(filepath)
        instance = cls(
            max_features=model_components['max_features'],
            hidden_size=model_components['hidden_size']
        )
        instance.vocab = model_components['vocab']
        instance.mlb = model_components['mlb']
        
        # Initialize and load model
        instance.model = KeywordExtractorModel(
            len(instance.vocab),
            instance.hidden_size,
            len(instance.mlb.classes_)
        ).to(instance.device)
        instance.model.load_state_dict(model_components['model_state'])
        
        return instance

def main():
    # Load your data
    df = pd.read_csv('data_noTHInAbstract.csv')
    
    # Initialize extractor
    extractor = GPUKeywordExtractor(max_features=5000, batch_size=32)
    
    # Train model
    extractor.train(
        abstracts=df['abstract'].values,
        keywords=df['keywords'].values,
        epochs=10
    )
    
    # Save model
    extractor.save_model('keyword_extractor_model.pt')
    
    # Example prediction
    test_abstract = "Online recruitment platforms typically employ Person-Job Fit models..."
    predicted_keywords = extractor.predict(test_abstract, top_k=5)
    print(f"Predicted keywords: {predicted_keywords}")

if __name__ == "__main__":
    main()

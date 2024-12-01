import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import ast

def load_and_preprocess_data(csv_path):
    """
    Load and preprocess data from CSV file
    
    Args:
        csv_path (str): Path to CSV file containing 'abstract' and 'keywords' columns
        
    Returns:
        tuple: (list of abstracts, list of keyword lists)
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Convert string representation of keywords to actual lists
    df['keywords'] = df['keywords'].apply(ast.literal_eval)
    
    return df['abstract'].tolist(), df['keywords'].tolist()

class KeywordDataset(Dataset):
    def __init__(self, texts, keywords, tokenizer, max_length=512):
        self.texts = texts
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create vocabulary from all keywords
        self.keyword_vocab = set()
        for keyword_list in keywords:
            self.keyword_vocab.update(keyword_list)
        self.keyword_vocab = list(self.keyword_vocab)
        
        # Create keyword to index mapping
        self.keyword2idx = {word: idx for idx, word in enumerate(self.keyword_vocab)}
    
    def __len__(self):
        return len(self.texts)
    
    def create_keyword_vector(self, keywords):
        """Create a binary vector indicating presence of keywords"""
        vector = torch.zeros(len(self.keyword_vocab))
        for keyword in keywords:
            if keyword in self.keyword2idx:
                vector[self.keyword2idx[keyword]] = 1
        return vector
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        keywords = self.keywords[idx]
        
        # Tokenize text
        text_encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create keyword vector
        keyword_vector = self.create_keyword_vector(keywords)
        
        return {
            'input_ids': text_encoding['input_ids'].squeeze(),
            'attention_mask': text_encoding['attention_mask'].squeeze(),
            'keywords': keyword_vector
        }

class KeywordExtractor(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', num_keywords=100):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_keywords)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return self.sigmoid(logits)

def train_model(model, train_loader, val_loader, num_epochs=5, learning_rate=2e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Training on device: {device}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            keywords = batch['keywords'].float().to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, keywords)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                keywords = batch['keywords'].float().to(device)
                
                outputs = model(input_ids, attention_mask)
                val_loss += criterion(outputs, keywords).item()
        
        print(f'Epoch {epoch + 1}:')
        print(f'Training Loss: {total_loss / len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss / len(val_loader):.4f}')

def extract_keywords(model, text, tokenizer, dataset, top_k=5, threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Tokenize text
    encoding = tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    # Get predicted keywords
    scores = outputs[0].cpu().numpy()
    top_indices = np.argsort(scores)[-top_k:]
    
    keywords = []
    for idx in top_indices:
        if scores[idx] > threshold:
            keywords.append(dataset.keyword_vocab[idx])
    
    return keywords

# Example usage
if __name__ == "__main__":
    # Load data from CSV
    csv_path = "/content/drive/MyDrive/DSDE/more_filtered_scopus_data.csv"  # Replace with your CSV path
    abstracts, keywords = load_and_preprocess_data(csv_path)
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Create dataset
    dataset = KeywordDataset(abstracts, keywords, tokenizer)
    
    # Initialize model with correct number of keywords
    model = KeywordExtractor(num_keywords=len(dataset.keyword_vocab))
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Train model
    train_model(model, train_loader, val_loader)
    
    # Save the trained model
    model_save_path = "/content/drive/MyDrive/DSDE/keyword_extractor.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Example text to extract keywords from
    example_text = """This paper presents a new deep learning approach for natural language processing
        with applications in text classification and sentiment analysis. The proposed
        method combines transformer architectures with attention mechanisms to improve
        performance on various NLP tasks."""
    
    # Extract keywords
    extracted_keywords = extract_keywords(model, example_text, tokenizer, dataset)
    
    print("Extracted Keywords:", extracted_keywords)
    
    

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import wandb  # Optional: for logging
import os
from datetime import datetime
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

# pip install torch transformers tqdm wandb pandas scikit-learn nltk rouge-score

class KeywordDataset(Dataset):
    def __init__(self, abstracts, keywords, tokenizer, max_length=512):
        self.abstracts = abstracts
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.abstracts)
    
    def __getitem__(self, idx):
        abstract = str(self.abstracts[idx])
        keyword_list = self.keywords[idx]
        
        # Tokenize abstract
        abstract_encoding = self.tokenizer(
            abstract,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize keywords
        keyword_text = ', '.join(keyword_list)
        keyword_encoding = self.tokenizer(
            keyword_text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'abstract_ids': abstract_encoding['input_ids'].squeeze(),
            'abstract_mask': abstract_encoding['attention_mask'].squeeze(),
            'keyword_ids': keyword_encoding['input_ids'].squeeze(),
            'keyword_mask': keyword_encoding['attention_mask'].squeeze()
        }
        
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = self._get_state_dict(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            self.status = f'EarlyStopping counter: {self.counter} out of {self.patience}'
            if self.counter >= self.patience:
                self.status = f'EarlyStopping triggered after {self.counter} epochs'
                return True
        else:
            self.best_loss = val_loss
            self.best_model = self._get_state_dict(model)
            self.counter = 0
        return False
    
    def _get_state_dict(self, model):
        if isinstance(model, torch.nn.DataParallel):
            return model.module.state_dict()
        return model.state_dict()
    
    def restore_best_model(self, model):
        if self.restore_best_weights and self.best_model is not None:
            model.load_state_dict(self.best_model)
            
class TrainingConfig:
    def __init__(self,
                num_epochs=30,
                batch_size=8,
                learning_rate=2e-5,
                weight_decay=0.01,
                patience=7,
                min_delta=1e-4,
                gradient_clip_val=1.0,
                accumulation_steps=2,
                warmup_steps=1000,
                checkpoint_dir='checkpoints',
                use_wandb=False):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_delta = min_delta
        self.gradient_clip_val = gradient_clip_val
        self.accumulation_steps = accumulation_steps
        self.warmup_steps = warmup_steps
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb = use_wandb
        
def train_and_evaluate(model, train_loader, val_loader, config, device):
    """
    Complete training loop with optimization techniques
    """
    if config.use_wandb:
        wandb.init(project="keyword-generation", config=vars(config))
    
    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Initialize optimizer with weight decay
    optimizer = AdamW([
        {'params': [p for n, p in model.named_parameters() if 'bias' not in n],
        'weight_decay': config.weight_decay},
        {'params': [p for n, p in model.named_parameters() if 'bias' in n],
        'weight_decay': 0.0}
    ], lr=config.learning_rate)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                patience=3, verbose=True)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.patience,
                                min_delta=config.min_delta)
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        epoch_steps = 0
        
        
        # Training phase
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs}') as pbar:
            for i, batch in enumerate(pbar):
                # Move batch to device
                abstract_ids = batch['abstract_ids'].to(device)
                abstract_mask = batch['abstract_mask'].to(device)
                keyword_ids = batch['keyword_ids'].to(device)
                
                # Forward pass
                outputs = model(abstract_ids, abstract_mask, keyword_ids)
                loss = criterion(outputs.view(-1, outputs.size(-1)),
                                keyword_ids.view(-1))
                
                # Scale loss for gradient accumulation
                loss = loss / config.accumulation_steps
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.gradient_clip_val
                )
                
                # Gradient accumulation
                if (i + 1) % config.accumulation_steps == 0:
                    # Warmup steps
                    if global_step < config.warmup_steps:
                        lr_scale = min(1., float(global_step + 1) /
                                    config.warmup_steps)
                        for pg in optimizer.param_groups:
                            pg['lr'] = lr_scale * config.learning_rate
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                # Update metrics
                total_loss += loss.item() * config.accumulation_steps
                epoch_steps += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': total_loss / epoch_steps})
                
                if config.use_wandb:
                    wandb.log({
                        'train_loss': loss.item() * config.accumulation_steps,
                        'learning_rate': optimizer.param_groups[0]['lr']
                    })
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                abstract_ids = batch['abstract_ids'].to(device)
                abstract_mask = batch['abstract_mask'].to(device)
                keyword_ids = batch['keyword_ids'].to(device)
                
                outputs = model(abstract_ids, abstract_mask, keyword_ids)
                loss = criterion(outputs.view(-1, outputs.size(-1)),
                                keyword_ids.view(-1))
                
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Logging
        print(f'\nEpoch {epoch + 1}')
        print(f'Average training loss: {total_loss / epoch_steps:.4f}')
        print(f'Average validation loss: {avg_val_loss:.4f}')
        
        if config.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'avg_train_loss': total_loss / epoch_steps,
                'avg_val_loss': avg_val_loss
            })
        
        # Save checkpoint if best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f'best_model_epoch_{epoch + 1}.pt'
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config
            }, checkpoint_path)
        
        # Early stopping check
        if early_stopping(avg_val_loss, model):
            print(early_stopping.status)
            break
    
    # Restore best model
    early_stopping.restore_best_model(model)
    
    if config.use_wandb:
        wandb.finish()
    
    return model

def prepare_data(csv_path, tokenizer_name="allenai/scibert_scivocab_uncased"):
    """
    Load and prepare data for training
    """
    # Load data
    print("Loading data from CSV...")
    df = pd.read_csv(csv_path)
    
    # Process keywords (assuming they're comma-separated in the CSV)
    def process_keywords(kw):
        if isinstance(kw, str):
            return [k.strip() for k in kw.split(',')]
        return []
    
    df['keywords'] = df['keywords'].apply(process_keywords)
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = KeywordDataset(
        abstracts=train_df['abstract'].values,
        keywords=train_df['keywords'].values,
        tokenizer=tokenizer
    )
    
    val_dataset = KeywordDataset(
        abstracts=val_df['abstract'].values,
        keywords=val_df['keywords'].values,
        tokenizer=tokenizer
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader, tokenizer

class KeywordGenerator(nn.Module):
    def __init__(self, encoder_model_name, decoder_model_name, tokenizer):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_model_name)
        self.decoder = AutoModel.from_pretrained(decoder_model_name)
        self.tokenizer = tokenizer
        
        # Add projection layers
        self.encoder_projection = nn.Linear(
            self.encoder.config.hidden_size,
            self.decoder.config.hidden_size
        )
        
        self.keyword_projection = nn.Linear(
            self.decoder.config.hidden_size,
            len(tokenizer.vocab)
        )
        
    def forward(self, abstract_ids, abstract_mask, keyword_ids=None):
        print(f"abstract_ids shape: {abstract_ids.shape}, device: {abstract_ids.device}")
        print(f"abstract_mask shape: {abstract_mask.shape}, device: {abstract_mask.device}")
        if keyword_ids is not None:
            print(f"keyword_ids shape: {keyword_ids.shape}, device: {keyword_ids.device}")

        # Ensure abstract_ids and abstract_mask are on the same device
        abstract_ids = abstract_ids.to(self.encoder.device)
        abstract_mask = abstract_mask.to(self.encoder.device)
        
        # Encode abstract
        encoder_outputs = self.encoder(
            input_ids=abstract_ids,
            attention_mask=abstract_mask
        ).last_hidden_state
        
        # Project encoder outputs
        encoder_outputs = self.encoder_projection(encoder_outputs)
        
        # Decode
        if keyword_ids is not None:
            # Ensure keyword_ids is on the right device
            keyword_ids = keyword_ids.to(self.decoder.device)
            
            # Create cross attention mask
            cross_attention_mask = abstract_mask.unsqueeze(1).unsqueeze(2)
            
            decoder_outputs = self.decoder(
                input_ids=keyword_ids,
                encoder_hidden_states=encoder_outputs,
                encoder_attention_mask=abstract_mask  # Explicitly pass attention mask
            ).last_hidden_state
        else:
            # Inference mode implementation (similar modifications)
            batch_size = abstract_ids.size(0)
            decoder_inputs = torch.ones(
                (batch_size, 1),
                dtype=torch.long,
                device=abstract_ids.device
            ) * self.tokenizer.bos_token_id
            
            max_length = 512
            for _ in range(max_length - 1):
                decoder_outputs = self.decoder(
                    input_ids=decoder_inputs,
                    encoder_hidden_states=encoder_outputs,
                    encoder_attention_mask=abstract_mask
                ).last_hidden_state
                
                next_token_logits = self.keyword_projection(
                    decoder_outputs[:, -1, :]
                )
                next_token = next_token_logits.argmax(dim=-1)
                
                decoder_inputs = torch.cat(
                    [decoder_inputs, next_token.unsqueeze(-1)],
                    dim=-1
                )
                
                if (next_token == self.tokenizer.eos_token_id).all():
                    break
            
            return decoder_inputs
        
        # Project to vocabulary
        keyword_logits = self.keyword_projection(decoder_outputs)
        return keyword_logits

class KeywordPredictor:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        self.rouge = Rouge()
        
    def predict_keywords(self, abstract, max_length=512):
        """
        Predict keywords for a single abstract
        """
        # Tokenize abstract
        inputs = self.tokenizer(
            abstract,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            # Generate keyword tokens
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=4,
                length_penalty=0.8,
                early_stopping=True
            )
            
            # Decode keywords
            predicted_keywords = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Split into individual keywords
            keywords = [kw.strip() for kw in predicted_keywords.split(',')]
            
        return keywords

def evaluate_model(model, test_loader, tokenizer, device):
    """
    Evaluate model performance on test set
    """
    model.eval()
    predictor = KeywordPredictor(model, tokenizer, device)
    
    all_true_keywords = []
    all_pred_keywords = []
    bleu_scores = []
    rouge_scores = {'rouge-1': [], 'rouge-2': [], 'rouge-l': []}
    
    print("Evaluating model...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # Get batch data
            abstract_ids = batch['abstract_ids'].to(device)
            abstract_mask = batch['abstract_mask'].to(device)
            
            # Get original abstracts
            abstracts = tokenizer.batch_decode(
                abstract_ids,
                skip_special_tokens=True
            )
            
            # Get true keywords
            true_keywords = tokenizer.batch_decode(
                batch['keyword_ids'],
                skip_special_tokens=True
            )
            
            # Predict keywords for each abstract
            for abstract, true_kw in zip(abstracts, true_keywords):
                pred_keywords = predictor.predict_keywords(abstract)
                
                # Store for metrics calculation
                all_true_keywords.append(true_kw.split(','))
                all_pred_keywords.append(pred_keywords)
                
                # Calculate BLEU score
                bleu = sentence_bleu([true_kw.split()], ' '.join(pred_keywords).split())
                bleu_scores.append(bleu)
                
                # Calculate ROUGE scores
                try:
                    rouge_score = predictor.rouge.get_scores(
                        ' '.join(pred_keywords),
                        true_kw
                    )[0]
                    for metric in rouge_scores:
                        rouge_scores[metric].append(rouge_score[metric]['f'])
                except:
                    # Handle empty predictions/references
                    for metric in rouge_scores:
                        rouge_scores[metric].append(0)
    
    # Calculate metrics
    results = calculate_metrics(
        all_true_keywords,
        all_pred_keywords,
        bleu_scores,
        rouge_scores
    )
    
    return results

def calculate_metrics(true_keywords, pred_keywords, bleu_scores, rouge_scores):
    """
    Calculate various evaluation metrics
    """
    # Flatten keyword lists for precision/recall calculation
    true_flat = set([kw for sublist in true_keywords for kw in sublist])
    pred_flat = set([kw for sublist in pred_keywords for kw in sublist])
    
    # Calculate precision, recall, F1
    tp = len(true_flat.intersection(pred_flat))
    fp = len(pred_flat - true_flat)
    fn = len(true_flat - pred_flat)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Average BLEU and ROUGE scores
    avg_bleu = np.mean(bleu_scores)
    avg_rouge = {metric: np.mean(scores) for metric, scores in rouge_scores.items()}
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'bleu_score': avg_bleu,
        'rouge_scores': avg_rouge
    }

def test_on_new_data(model_path, test_csv, tokenizer_name="allenai/scibert_scivocab_uncased"):
    """
    Test the model on new data
    """
    # Load model and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Load model (assuming you have the KeywordGenerator class from previous code)
    model = KeywordGenerator(
        encoder_model_name=tokenizer_name,
        decoder_model_name="bert-base-uncased",
        tokenizer=tokenizer
    )
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    
    # Load test data
    test_df = pd.read_csv(test_csv)
    test_dataset = KeywordDataset(
        abstracts=test_df['abstract'].values,
        keywords=test_df['keywords'].values,
        tokenizer=tokenizer
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, drop_last=True)
    
    # Evaluate
    results = evaluate_model(model, test_loader, tokenizer, device)
    
    # Print results
    print("\nModel Evaluation Results:")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"BLEU Score: {results['bleu_score']:.4f}")
    print("\nROUGE Scores:")
    for metric, score in results['rouge_scores'].items():
        print(f"{metric}: {score:.4f}")
    
    return results

def generate_keywords_for_single_abstract(abstract, model_path, tokenizer_name="allenai/scibert_scivocab_uncased"):
    """
    Generate keywords for a single abstract
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Load model
    model = KeywordGenerator(
        encoder_model_name=tokenizer_name,
        decoder_model_name="bert-base-uncased",
        tokenizer=tokenizer
    )
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    
    # Create predictor
    predictor = KeywordPredictor(model, tokenizer, device)
    
    # Generate keywords
    keywords = predictor.predict_keywords(abstract)
    
    return keywords

def main():
    # Initialize data
    train_loader, val_loader, tokenizer = prepare_data('C:/Users/USER/Desktop/my-git/dsde_project/data_preparation/given_data/data/try.csv')
    
    # Initialize model (using the code from previous example)
    model = KeywordGenerator(
        encoder_model_name="allenai/scibert_scivocab_uncased",
        decoder_model_name="bert-base-uncased",
        tokenizer=tokenizer
    )
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize training config
    config = TrainingConfig(
        num_epochs=30,
        batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        patience=7,
        min_delta=1e-4,
        gradient_clip_val=1.0,
        accumulation_steps=2,
        warmup_steps=1000,
        checkpoint_dir='checkpoints',
        use_wandb=False
    )
    
    # Train model
    model = train_and_evaluate(model, train_loader, val_loader, config, device)
    
    # Save final model
    torch.save(model.state_dict(), 'final_model.pt')
    
    # Test on new data
    results = test_on_new_data(
        model_path='final_model.pt',
        test_csv='test_data.csv'
    )
    
    # Example of generating keywords for a single abstract
    test_abstract = "Online recruitment platforms typically employ Person-Job Fit models in the core service that automatically match suitable job seekers with appropriate job positions. While existing works leverage historical or contextual information, they often disregard a crucial aspect: job seekers’ social relationships in professional networks. This paper emphasizes the importance of incorporating professional networks into the Person-Job Fit model. Our innovative approach consists of two stages: (1) defining a Workplace Heterogeneous Information Network (WHIN) to capture heterogeneous knowledge, including professional connections and pre-training representations of various entities using a heterogeneous graph neural network; (2) designing a Contextual Social Attention Graph Neural Network (CSAGNN) that supplements users’ missing information with professional connections’ contextual information. We introduce a job-specific attention mechanism in CSAGNN to handle noisy professional networks, leveraging pre-trained entity representations from WHIN. We demonstrate the effectiveness of our approach through experimental evaluations conducted across three real-world recruitment datasets from LinkedIn, showing superior performance compared to baseline models."
    # The actual keyword: ['Person-Job', 'Fit,', 'Heterogeneous', 'Information', 'Network,', 'Graph', 'Neural', 'Network']
    
    keywords = generate_keywords_for_single_abstract(
        abstract=test_abstract,
        model_path='final_model.pt'
    )
    
    print("\nGenerated keywords for test abstract:")
    print(keywords)

# Example usage:
if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import spacy
import os
from models.transformer import Transformer

# Load English and Spanish tokenizer
spacy_eng = spacy.load("en_core_web_sm")
spacy_esp = spacy.load("es_core_news_sm")

# Tokenizer functions
def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

def tokenize_esp(text):
    return [tok.text for tok in spacy_esp.tokenizer(text)]

# Custom Dataset with Sampling
class TranslationDataset(Dataset):
    def __init__(self, data_path, input_vocab_size, target_vocab_size, num_samples=None):
        self.data = pd.read_csv(data_path, sep='\t', header=None)
        self.data.columns = ['en', 'es', 'attr']

        # Sample the data if num_samples is provided
        if num_samples is not None and num_samples < len(self.data):
            self.data = self.data.sample(n=num_samples, random_state=42).reset_index(drop=True)

        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size

        self.en_vocab = self.build_vocab(self.data['en'])
        self.es_vocab = self.build_vocab(self.data['es'])
        self.es_index_to_word = {index: word for word, index in self.es_vocab.items()}

    def build_vocab(self, sentences):
        vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        idx = 4
        for sentence in sentences:
            for token in tokenize_eng(sentence):
                if token not in vocab:
                    vocab[token] = idx
                    idx += 1
        return vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        en_sentence = self.data.iloc[idx, 0]
        es_sentence = self.data.iloc[idx, 1]

        en_tokens = [self.en_vocab.get(token, self.en_vocab['<UNK>']) for token in tokenize_eng(en_sentence)]
        es_tokens = [self.es_vocab.get(token, self.es_vocab['<UNK>']) for token in tokenize_esp(es_sentence)]

        en_tensor = torch.tensor([self.en_vocab['<SOS>']] + en_tokens + [self.en_vocab['<EOS>']])
        es_tensor = torch.tensor([self.es_vocab['<SOS>']] + es_tokens + [self.es_vocab['<EOS>']])

        # Ensure tokens are within vocab size
        en_tensor = torch.clamp(en_tensor, max=self.input_vocab_size-1)
        es_tensor = torch.clamp(es_tensor, max=self.target_vocab_size-1)

        return en_tensor, es_tensor

# Hyperparameters
num_layers = 4
d_model = 128
num_heads = 8
dff = 512
input_vocab_size = 10000  # Adjust this based on your dataset
target_vocab_size = 26038  # Adjusted based on your trg max index
dropout_rate = 0.1
batch_size = 32
epochs = 20

# Instantiate model
transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=input_vocab_size,
    target_vocab_size=target_vocab_size,
    dropout_rate=dropout_rate
)

# Optimizer and loss function
optimizer = optim.Adam(transformer.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming <PAD> token has index 0

# DataLoader
def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=0, batch_first=True)
    trg_batch = pad_sequence(trg_batch, padding_value=0, batch_first=True)
    return src_batch, trg_batch

# Create a smaller dataset for testing
num_samples = 1000  # Use a smaller number for quicker testing
dataset = TranslationDataset('../spa.csv', input_vocab_size, target_vocab_size, num_samples=num_samples)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Training function with model saving
def train(model, dataloader, criterion, optimizer, epochs, save_path):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (src, trg) in enumerate(dataloader):
            src = src.to(device)
            trg = trg.to(device)

            # Check range of src and trg
            print(f"Batch {batch_idx + 1}/{len(dataloader)} - src max index: {src.max().item()}, src min index: {src.min().item()}")
            print(f"Batch {batch_idx + 1}/{len(dataloader)} - trg max index: {trg.max().item()}, trg min index: {trg.min().item()}")

            trg_input = trg[:, :-1]
            trg_output = trg[:, 1:]

            # Forward pass
            optimizer.zero_grad()
            output = model(src, trg_input)

            # Compute loss
            output = output.reshape(-1, output.shape[-1])
            trg_output = trg_output.reshape(-1)
            loss = criterion(output, trg_output)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

    # Save the model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# Translate function with debugging
def translate_sentence(model, sentence, src_vocab, trg_vocab, src_tokenizer, trg_index_to_word, max_length=50):
    model.eval()
    
    # Tokenize the sentence
    tokens = [src_vocab.get(token, src_vocab['<UNK>']) for token in src_tokenizer(sentence)]
    src_tensor = torch.tensor([src_vocab['<SOS>']] + tokens + [src_vocab['<EOS>']]).unsqueeze(0).to(device)
    
    print(f"Source Tensor: {src_tensor}")
    
    # Encode the source sentence
    src_enc = model.encoder(src_tensor)
    
    # Initialize the target sequence with <SOS>
    trg_indices = [trg_vocab['<SOS>']]
    
    consecutive_repeats = 0
    last_token = None
    
    for _ in range(max_length):
        trg_tensor = torch.tensor(trg_indices).unsqueeze(0).to(device)
        
        print(f"Target Tensor: {trg_tensor}")
        
        # Decode the encoded source sentence
        output = model.decoder(trg_tensor, src_enc)
        
        if output.size(1) == 0:
            print("Decoder output is empty, breaking out of the loop.")
            break
        
        output = output[:, -1, :]  # Get the last output token
        pred_token = output.argmax(1).item()
        
        trg_indices.append(pred_token)
        
        if pred_token == trg_vocab['<EOS>']:
            break
        
        # Check for consecutive repeats
        if pred_token == last_token:
            consecutive_repeats += 1
            if consecutive_repeats > 5:  # Stop if the same token is repeated more than 5 times consecutively
                print("Stopping early due to repeated tokens.")
                break
        else:
            consecutive_repeats = 0
        
        last_token = pred_token
    
    trg_tokens = [trg_index_to_word.get(idx, '<UNK>') for idx in trg_indices]
    return trg_tokens[1:-1]  # Remove <SOS> and <EOS> tokens

# Training the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer.to(device)
train(transformer, dataloader, criterion, optimizer, epochs, 'transformer_model.pth')

# Load the model for inference
transformer.load_state_dict(torch.load('transformer_model.pth'))

# Translate a sentence
sentence = "This is a test sentence."
translated_sentence = translate_sentence(transformer, sentence, dataset.en_vocab, dataset.es_vocab, tokenize_eng, dataset.es_index_to_word)
print(f"Translated sentence: {' '.join(translated_sentence)}")
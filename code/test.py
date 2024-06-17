import torch
from models.transformer import Transformer
from torch import nn
# Define some parameters for the transformer
num_layers = 4
d_model = 128
num_heads = 8
dff = 512
input_vocab_size = 10000
target_vocab_size = 10000
dropout_rate = 0.1

# Create a transformer instance
transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=input_vocab_size,
    target_vocab_size=target_vocab_size,
    dropout_rate=dropout_rate
)

# Create some sample input data
batch_size = 32
context_len = 50
target_len = 60

context = torch.randint(0, input_vocab_size, (batch_size, context_len))
target = torch.randint(0, target_vocab_size, (batch_size, target_len))

# Run the transformer model
logits = transformer(context, target)

# Print the shape of the output logits
print(f"Logits shape: {logits.shape}")

# Assertions to check the correctness
assert logits.shape == (batch_size, target_len, target_vocab_size), "Logits shape is incorrect"
print("Logits shape is correct.")

# Check the values
print("Sample logits output:", logits[0, 0, :5].detach().cpu().numpy())

logits = transformer(context, target)

# Create dummy targets for calculating loss
dummy_targets = torch.randint(0, target_vocab_size, (batch_size, target_len)).long()

# Define a loss function
criterion = nn.CrossEntropyLoss()

# Reshape logits and targets for loss calculation
logits_reshaped = logits.view(-1, target_vocab_size)
dummy_targets_reshaped = dummy_targets.view(-1)

# Calculate the loss
loss = criterion(logits_reshaped, dummy_targets_reshaped)
print(f"Loss: {loss.item()}")

# Perform a backward pass
loss.backward()

# Check gradients
print("Gradients for the first layer of the encoder:")
print(transformer.encoder.enc_layers[0].self_attention.mha.in_proj_weight.grad)

# Ensure gradients are not None
assert transformer.encoder.enc_layers[0].self_attention.mha.in_proj_weight.grad is not None, "Gradients are not flowing"
print("Gradients are flowing correctly.")
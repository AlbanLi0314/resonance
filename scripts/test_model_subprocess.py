#!/usr/bin/env python3
"""
Test model loading in a subprocess to isolate any state issues.
"""
import subprocess
import sys

# The code to run in a subprocess
test_code = '''
import os
import sys

# Set environment before any imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

# Disable MPS
if hasattr(torch.backends, 'mps'):
    original_is_available = torch.backends.mps.is_available
    torch.backends.mps.is_available = lambda: False
    print(f"MPS was: {original_is_available()}, now disabled")

print(f"PyTorch version: {torch.__version__}")
print(f"Device: cpu (forced)")

# Import transformers
print("Importing transformers...")
from transformers import AutoTokenizer, AutoModel

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print("Tokenizer loaded!")

print("Loading model...")
model = AutoModel.from_pretrained("bert-base-uncased")
print("Model loaded successfully!")

# Test encoding
inputs = tokenizer("test input", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
print(f"Output shape: {outputs.last_hidden_state.shape}")
print("SUCCESS!")
'''

if __name__ == "__main__":
    print("Running model test in subprocess...")
    result = subprocess.run(
        [sys.executable, "-c", test_code],
        capture_output=True,
        text=True,
        env={
            **dict(__import__('os').environ),
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )

    print("STDOUT:")
    print(result.stdout)

    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    print(f"Return code: {result.returncode}")

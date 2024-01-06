import torch

MAX_EPOCHS=40
LEARNING_RATE = 1e-4
MAX_LEARNING_RATE = 1E-3
STEPS_PER_EPOCH = 2084
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
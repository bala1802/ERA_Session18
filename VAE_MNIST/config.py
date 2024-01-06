import torch

LEARNING_RATE = 1e-4
MAX_LEARNING_RATE = 1E-3
STEPS_PER_EPOCH = 10 #FIXME
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
import torch
import multiprocessing

IMG_SIZE    = 128
BATCH_SIZE  = 16
NUM_WORKERS = int(multiprocessing.cpu_count() / 2)

EPOCHS = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'
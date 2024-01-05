from torch.utils.data import DataLoader

import config

def get_data_loader(data_path):
    return DataLoader(data_path,batch_size = config.BATCH_SIZE, shuffle=True,num_workers=config.NUM_WORKERS, pin_memory=True)
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

from vae_model import VAE
import config

def train(train_dataloader, val_dataloader):
    pl.seed_everything(1234)
    model = VAE().to('cuda')
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=config.MAX_EPOCHS,
                         logger= [CSVLogger(save_dir="logs/")],
                         callbacks=[LearningRateMonitor(logging_interval="step")])
    trainer.fit(model, train_dataloader ,val_dataloader)
    return model, trainer
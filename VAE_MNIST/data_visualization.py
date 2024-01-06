import torch
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import torchvision.transforms as T
from matplotlib.pyplot import figure

def plot_metrics(trainer):
    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    del metrics["step"]
    metrics.set_index("epoch", inplace=True)
    display(metrics.dropna(axis=1, how="all").head(400))
    sn.relplot(data=metrics, kind="line")

def generate_image(x, y, wrong_y, model,num_predictions=25):
    figure(figsize=(8, 3), dpi=300)
    t2img = T.ToPILImage()

    with torch.no_grad():

        x_encoded   = model.encoder(x.to('cuda'),wrong_y.to('cuda'))
        mu, log_var = model.fc_mu(x_encoded), model.fc_var(x_encoded)

        std = torch.exp(log_var/2)
        q = torch.distributions.Normal(mu,std)
        z = q.rsample()

        x_hat = model.decoder(z,wrong_y.to('cuda'))

        fig = plt.figure(figsize=(10,10))

        for idx in np.arange(num_predictions):

          ax = fig.add_subplot(5,5,idx + 1,xticks=[],yticks=[])
          img = x_hat[idx].to('cpu')
          plt.imshow(img.permute(1,2,0))
          ax.set_title(f"Label/Image: {wrong_y[idx]} / {y[idx]}")

        fig.tight_layout()
        plt.show()

def plot_results(train_dataloader, model):
    val_batch = next(iter(train_dataloader))
    x, y, y_changed  = val_batch['image'],val_batch['correct_label'],val_batch['changed_label']
    wrong_y = torch.flip(y_changed,[0])
    generate_image(x,y,wrong_y, model.to('cuda'))
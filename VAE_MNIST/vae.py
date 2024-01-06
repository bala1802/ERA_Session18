import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import OneCycleLR
import torch.optim as optim

class VAE(pl.LightningModule):
  def __init__(self,enc_out_dim=32, latent_dim=256, featureDim=64*24*24):

        super().__init__()

        self.save_hyperparameters()
        # encoder layers
        self.encConv1 = nn.Conv2d(2,32,3)
        self.encConv2 = nn.Conv2d(32,64,3)
        self.fc_mu = nn.Linear(featureDim,latent_dim)
        self.fc_var = nn.Linear(featureDim, latent_dim)

        # decoder layers
        self.deFC1 = nn.Linear(latent_dim + 10, featureDim)
        self.deConv1 = nn.ConvTranspose2d(64,32,3)
        self.deConv2 = nn.ConvTranspose2d(32,1,3)

        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        # classifier head layers
        self.chConv1 = nn.Conv2d(1,64,3)
        self.chout   = nn.Linear(64*26*26,10)
        # classifier loss
        self.bce_loss = nn.BCELoss(reduction = 'none')

  def encoder(self,x,y):
        # add label to image
        y = F.one_hot(y, num_classes = x.shape[2]).unsqueeze(1).unsqueeze(2) # one hot y
        y = torch.ones(x.shape).to('cuda') * y
        t = torch.cat((x,y),dim=1)
        x = F.relu(self.encConv1(t))
        x = F.relu(self.encConv2(x))
        batch_size = x.shape[0]
        x = x.view(batch_size,-1)
        return x

  def decoder(self,z,y):
        y = F.one_hot(y, num_classes = 10)
        z = torch.cat((z, y), dim=1) # b, 256 + 10
        x_hat = F.relu(self.deFC1(z))
        x_hat = x_hat.view(-1,64,24,24)
        x_hat = F.relu(self.deConv1(x_hat))
        x_hat = torch.sigmoid(self.deConv2(x_hat))
        return x_hat

  def multilabel_classifier(self,x_hat):
        x_hat_class = F.relu(self.chConv1(x_hat))
        batch_size = x_hat_class.shape[0]
        x_hat_class = x_hat_class.view(batch_size,-1)
        x_hat_class = F.sigmoid(self.chout(x_hat_class))
        return x_hat_class

  def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        scheduler = OneCycleLR(
                optimizer,
                max_lr= 1E-3,
                pct_start = 5/self.trainer.max_epochs,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=len(train_dataloader),
                div_factor=100,
                three_phase=False,
                final_div_factor=100,
                anneal_strategy='linear'
            )
        return {
             "optimizer": optimizer,
             "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
             }

  def kl_divergence(self,z,mu,std):
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

  def gaussian_likelihood(self,mean,logscale,sample):
        scale   = torch.exp(logscale)
        dist    = torch.distributions.Normal(mean,scale)
        log_pxz = dist.log_prob(sample)
        return log_pxz.sum(dim=(1,2,3))

  def forward(self,x,y,y_changed):
        # encoder
        x_encoded = self.encoder(x,y_changed)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)
        std = torch.exp(log_var/2)
        q = torch.distributions.Normal(mu,std)
        z = q.rsample()

        # decoder
        x_hat = self.decoder(z,y_changed)

        # multi label classifier
        x_multilabel = self.multilabel_classifier(x_hat)

        return x_hat,x_multilabel,z,mu,std

  def training_epoch_end(self,outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()

        print("****Training****")
        print(f'Epoch {self.current_epoch}: Train loss {loss}')
        
  def validation_epoch_end(self,outputs):
        loss = torch.stack([x for x in outputs]).mean()

        print("****Validation****")
        print(f'Epoch {self.current_epoch}: Validation loss {loss}')
        

  def mutli_label_loss(self,x_multilabel,y_combo_one_hot):
        ml_loss = self.bce_loss(x_multilabel,y_combo_one_hot).sum(dim=1)
        return ml_loss

  def training_step(self,batch,batch_idx):

        x, y, y_changed = batch['image'],batch['correct_label'],batch['changed_label']

        # forward pass thru model
        x_hat,x_multilabel,z,mu,std = self(x,y,y_changed)

        # vae loss
        recon_loss = self.gaussian_likelihood(x_hat,self.log_scale,x)
        kl = self.kl_divergence(z,mu,std)
        elbo = kl - recon_loss
        
        # combine y and y changed before sending to bce loss
        y_combo = torch.logical_or(F.one_hot(y, num_classes = 10), F.one_hot(y_changed, num_classes = 10)).float()
        ml_loss = self.mutli_label_loss(x_multilabel,y_combo)
        total_loss = (1 * ml_loss) + (0.001 * elbo)
              
        total_loss = total_loss.mean()
        
        self.log_dict({
            'elbo': (0.001 * elbo).mean(),
            'kl': kl.mean(),
            #'recon_loss': -1 * recon_loss.mean(),
            'multilabel loss':ml_loss.mean(),
            'total loss':total_loss
        })
        return total_loss
    
  def validation_step(self,batch,batch_idx):

        x, y, y_changed = batch['image'],batch['correct_label'],batch['changed_label']

        # forward pass thru model
        x_hat,x_multilabel,z,mu,std = self(x,y,y_changed)

        # vae loss
        recon_loss = self.gaussian_likelihood(x_hat,self.log_scale,x)
        kl = self.kl_divergence(z,mu,std)
        elbo = kl - recon_loss
        
        # combine y and y changed before sending to bce loss
        y_combo = torch.logical_or(F.one_hot(y, num_classes = 10), F.one_hot(y_changed, num_classes = 10)).float()
        ml_loss = self.mutli_label_loss(x_multilabel,y_combo)
        total_loss = (1 * ml_loss) + (0.001 * elbo)
              
        val_loss = total_loss.mean()
        
        return val_loss
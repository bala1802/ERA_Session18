import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from resnet18_encoder import ResNet18Enc
from resnet18_decoder import ResNet18Dec

class VAE(pl.LightningModule):

    def __init__(self,latent_dim=512):

        super().__init__()

        self.save_hyperparameters()
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        self.encoder = ResNet18Enc(z_dim=latent_dim)
        self.decoder = ResNet18Dec(z_dim=latent_dim)

        self.chConv1 = nn.Conv2d(3,64,3)
        self.chout   = nn.Linear(64*30*30,10)
        
        self.bce_loss = nn.BCELoss(reduction = 'none')

        self.training_step_outputs = []
        self.validation_step_outputs = []
    
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
                max_lr= 1E-2,
                pct_start = 5/self.trainer.max_epochs,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=10, #FIXME
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
    
    def mutli_label_loss(self,x_multilabel,y_combo_one_hot):
        ml_loss = self.bce_loss(x_multilabel,y_combo_one_hot).sum(dim=1)
        return ml_loss

    def forward(self,x,y):
        mu,log_var = self.encoder(x,y)
        std = torch.exp(log_var/2)
        q = torch.distributions.Normal(mu,std)
        z = q.rsample()
        x_hat = self.decoder(z,y)
        return mu,log_var,std,z,x_hat
    
    def on_train_epoch_end(self):
        loss = torch.stack(self.training_step_outputs).mean()
        print("****Training****")
        print(f'Epoch {self.current_epoch}: Train loss {loss}')
    
    def on_validation_epoch_end(self):
        loss = torch.stack(self.validation_step_outputs).mean()
        
        print("****Validation****")
        print(f'Epoch {self.current_epoch}: Validation loss {loss}')
    
    def validation_step(self,batch,batch_idx):

        x, y, y_changed = batch['image'],batch['correct_label'],batch['changed_label']

        mu,log_var = self.encoder(x,y_changed)
        std = torch.exp(log_var/2)
        q = torch.distributions.Normal(mu,std)
        z = q.rsample()
        x_hat = self.decoder(z,y_changed)

        # multi label classifier
        x_multilabel = self.multilabel_classifier(x_hat)

        # vae loss
        recon_loss = self.gaussian_likelihood(x_hat,self.log_scale,x)
        kl = self.kl_divergence(z,mu,std)
        elbo = kl - recon_loss

        # classifier loss
        y_combo = torch.logical_or(F.one_hot(y, num_classes = 10), F.one_hot(y_changed, num_classes = 10)).float()
        ml_loss = self.mutli_label_loss(x_multilabel,y_combo)

        # weight loss
        total_loss = (1 * ml_loss) + (0.01 * elbo)
        total_loss = total_loss.mean()

        self.validation_step_outputs.append(total_loss)

        return total_loss
    
    def training_step(self,batch,batch_idx):

        x, y, y_changed = batch['image'],batch['correct_label'],batch['changed_label']

        mu,log_var = self.encoder(x,y_changed)
        std = torch.exp(log_var/2)
        q = torch.distributions.Normal(mu,std)
        z = q.rsample()
        x_hat = self.decoder(z,y_changed)

        x_multilabel = self.multilabel_classifier(x_hat)
        
        # vae loss
        recon_loss = self.gaussian_likelihood(x_hat,self.log_scale,x)
        kl = self.kl_divergence(z,mu,std)
        elbo = kl - recon_loss

        # classifier loss
        y_combo = torch.logical_or(F.one_hot(y, num_classes = 10), F.one_hot(y_changed, num_classes = 10)).float()
        ml_loss = self.mutli_label_loss(x_multilabel,y_combo)
        
        total_loss = (1 * ml_loss) + (0.01 * elbo)
        total_loss = total_loss.mean()
        
        self.log_dict({
            'elbo': (0.001 * elbo).mean(),
            'kl': kl.mean(),
            'recon_loss': -1 * recon_loss.mean(),
            'multilabel loss':ml_loss.mean(),
            'total loss':total_loss
        })
        return total_loss
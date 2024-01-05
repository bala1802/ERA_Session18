import torch
from tqdm.auto import tqdm
import config

def train_unet(my_unet,loss_fn,device,train_dataloader,test_dataloader,optimizer):
  train_losses,test_losses = [],[]

  for epoch in tqdm(range(config.EPOCHS)):

    my_unet.train()
    train_loss = 0
    for batch, (X,y) in enumerate(train_dataloader):
      X,y = X.to(device), y.to(device)
      y_pred = my_unet(X)

      loss = loss_fn(y_pred,y.squeeze(1))
      train_loss += loss.item()

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    train_loss = train_loss / len(train_dataloader)

    # test model
    my_unet.eval()
    test_loss = 0

    with torch.inference_mode():
      for batch, (X,y) in enumerate(test_dataloader):
        X,y = X.to(device), y.to(device)
        y_pred = my_unet(X)

        loss = loss_fn(y_pred,y.squeeze(1))
        test_loss += loss.item()

      test_loss = test_loss / len(test_dataloader)

    print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"test_loss: {test_loss:.4f} "
          )
    train_losses.append(train_loss)
    test_losses.append(test_loss)

  return train_losses,test_losses
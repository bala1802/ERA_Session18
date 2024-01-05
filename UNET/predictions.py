import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as T

def VisualizeResults(unet_model, test_dataloader):
    unet_model.eval()
    unet_model.to('cpu')
    transform = T.ToPILImage()
    (test_pets_inputs, test_pets_targets) = next(iter(test_dataloader))
    fig, arr = plt.subplots(6, 3, figsize=(15, 15)) # batch size 16

    for index in range(6):
      img = test_pets_inputs[index].unsqueeze(0)
      pred_y = unet_model(img)
      pred_y = nn.Softmax(dim=1)(pred_y)
      pred_mask = pred_y.argmax(dim=1)
      pred_mask = pred_mask.unsqueeze(1).to(torch.float)


      arr[index,0].imshow(transform(test_pets_inputs[index]))
      arr[index,0].set_title('Processed Image')
      arr[index,1].imshow(transform(test_pets_targets[index].float()))
      arr[index,1].set_title('Actual Masked Image ')
      arr[index,2].imshow(pred_mask.squeeze(0).squeeze(0))
      arr[index,2].set_title('Predicted Masked Image ')

print("Hello world")
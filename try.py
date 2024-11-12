import torch
from core import SegTrainer
from configs import MyConfig
from datasets.all_prostate import AllProstateDataset
import warnings
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    config = MyConfig()
    config.init_dependent_config()

    trainer = SegTrainer(config)
    pit = AllProstateDataset(config, mode='val')
    image, label = pit.__getitem__(0)

    # Convert image to tensor and move it to the appropriate device
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = torch.tensor(image, dtype=torch.float32)  # Convert to tensor
    image = image.to(trainer.device)  # Move tensor to the device

    print(image.shape)

    # Perform prediction
    preds = trainer.model(image)
    colormap = {0: 0, 1:255}
    preds = colormap[preds.max(dim=1)[1]].cpu().numpy()

    # Move predictions to CPU and convert to numpy
    # preds = preds.cpu().detach().numpy()  # Move to CPU and detach

    # Assuming preds is [batch_size, channels, height, width]
    # Remove batch dimension and channel dimension for visualization
    preds = preds[0]  # Take first image and first channel

    # Normalize or scale the image to 0-255 for better visualization if needed
    preds = (preds - preds.min()) / (preds.max() - preds.min())  # Normalize to [0, 1]
    preds = (preds * 255).astype(np.uint8)  # Scale to [0, 255]
    # preds = 255 - preds  # Invert colors

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    # Display the image using matplotlib
    axes[0].imshow(preds, cmap='gray')
    axes[1].imshow(label, cmap='gray')
    # plt.imshow(preds, cmap='gray')  # Use 'gray' colormap for single-channel images
    # plt.colorbar()  # Optional: add a colorbar to the side
    # plt.title('Predicted Image')
    plt.axis('off')  # Optional: turn off axis
    plt.show()  # Display the image

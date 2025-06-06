"""
U-Net Training Script for Semantic Segmentation

Usage:
1. Edit the configuration variables in the main() function
2. Run: python unet_train.py

Expected folder structure:
data/
├── images/
│   ├── image001.png
│   ├── image002.png
│   └── ...
└── masks/
    ├── image001.png  # Corresponding mask
    ├── image002.png
    └── ...

The script will automatically:
- Load PNG images and masks
- Split data into train/validation sets
- Train the U-Net model
- Save the best model as 'best_unet_model.pth'
- Generate training plots and visualizations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, img_size=(256, 256)):
        """
        Args:
            image_dir (string): Directory with all the images
            mask_dir (string): Directory with all the masks
            transform (callable, optional): Optional transform to be applied on a sample
            img_size (tuple): Target size for images and masks
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_size = img_size

        # Get all image files
        self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        self.mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

        # Ensure we have matching pairs
        assert len(self.image_files) == len(self.mask_files), f"Number of images ({len(self.image_files)}) and masks ({len(self.mask_files)}) don't match"

        print(f"Found {len(self.image_files)} image-mask pairs")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image and mask
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        # Read image and convert to RGB
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale

        # Resize to target size
        image = image.resize(self.img_size, Image.BILINEAR)
        mask = mask.resize(self.img_size, Image.NEAREST)

        # Convert to numpy arrays
        image = np.array(image)
        mask = np.array(mask)

        # Normalize mask values (assuming binary segmentation: 0 and 255 -> 0 and 1)
        mask = (mask > 128).astype(np.float32)

        # Apply transforms if provided
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            # Convert to torch tensors
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask

def dice_coefficient(pred, target, smooth=1e-6):
    """Calculate Dice coefficient for binary segmentation
    Handles both 3D and 4D tensors automatically"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    # Handle both 3D [batch, height, width] and 4D [batch, channels, height, width] tensors
    if pred.dim() == 4:
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    else:  # 3D tensor (after squeeze operation)
        intersection = (pred * target).sum(dim=(1, 2))
        union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()

def pixel_accuracy(pred, target):
    """Calculate pixel accuracy"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    correct = (pred == target).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        # Flatten tensors
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target.float())
        dice_loss = self.dice(pred, target)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss

def train_model(model, train_loader, val_loader, num_epochs, device, learning_rate=1e-4):
    """Training function"""
    criterion = CombinedLoss(alpha=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        train_acc = 0.0

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (images, masks) in enumerate(train_bar):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # Reshape for loss calculation
            outputs = outputs.squeeze(1)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            # Calculate metrics
            train_loss += loss.item()
            dice = dice_coefficient(outputs, masks)
            acc = pixel_accuracy(outputs, masks)
            train_dice += dice.item()
            train_acc += acc.item()

            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{dice.item():.4f}',
                'Acc': f'{acc.item():.4f}'
            })

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_acc = 0.0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, masks in val_bar:
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                outputs = outputs.squeeze(1)

                loss = criterion(outputs, masks)
                val_loss += loss.item()

                dice = dice_coefficient(outputs, masks)
                acc = pixel_accuracy(outputs, masks)
                val_dice += dice.item()
                val_acc += acc.item()

                val_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Dice': f'{dice.item():.4f}',
                    'Acc': f'{acc.item():.4f}'
                })

        # Calculate epoch averages
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_dice /= len(train_loader)
        val_dice /= len(val_loader)
        train_acc /= len(train_loader)
        val_acc /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val Acc: {val_acc:.4f}')
        print('-' * 60)

        # Update learning rate
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_dice': val_dice,
            }, 'best_unet_model.pth')
            print(f'New best model saved with validation loss: {val_loss:.4f}')

    return train_losses, val_losses

def visualize_predictions(model, dataset, device, num_samples=4):
    """Visualize model predictions"""
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))

    with torch.no_grad():
        for i in range(num_samples):
            # Get random sample
            idx = np.random.randint(0, len(dataset))
            image, mask = dataset[idx]

            # Add batch dimension and predict
            image_batch = image.unsqueeze(0).to(device)
            pred = model(image_batch)
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()

            # Convert tensors to numpy for visualization
            image_np = image.permute(1, 2, 0).numpy()
            mask_np = mask.numpy()

            # Plot
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(mask_np, cmap='gray')
            axes[i, 1].set_title('Ground Truth Mask')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title('Predicted Mask')
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig('predictions_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    # =============================================================================
    # CONFIGURATION - Edit these variables to match your setup
    # =============================================================================
    IMAGE_DIR = "/Users/christian/Downloads/15168163/Training_data/Train"  # Directory containing training images (.png)
    MASK_DIR = "/Users/christian/Downloads/15168163/Training_data/Mask"  # Directory containing training masks (.png)
    EPOCHS = 20  # Number of training epochs
    BATCH_SIZE = 8  # Batch size for training
    LEARNING_RATE = 1e-4  # Learning rate
    IMG_SIZE = 256  # Input image size (will resize to IMG_SIZE x IMG_SIZE)
    VAL_SPLIT = 0.2  # Validation split ratio (0.2 = 20% for validation)
    N_CLASSES = 1  # Number of output classes (1 for binary segmentation)
    # =============================================================================

    # Set device and configure DataLoader settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f'Using device: {device}')

    # Configure DataLoader settings based on device
    use_pin_memory = device.type == 'cuda'
    num_workers = 2 if device.type == 'mps' else 4  # Reduce workers for MPS

    # Create datasets
    full_dataset = SegmentationDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        img_size=(IMG_SIZE, IMG_SIZE)
    )

    # Split dataset
    train_size = int((1 - VAL_SPLIT) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    # Create model
    model = UNet(n_channels=3, n_classes=N_CLASSES).to(device)
    print(f'Model created with {sum(p.numel() for p in model.parameters())} parameters')

    # Train model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        EPOCHS, device, LEARNING_RATE
    )

    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Visualize some predictions
    visualize_predictions(model, full_dataset, device)

    print('Training completed! Best model saved as best_unet_model.pth')
    print('To use this model for inference, load it with:')
    print('checkpoint = torch.load("best_unet_model.pth")')
    print('model.load_state_dict(checkpoint["model_state_dict"])')

if __name__ == '__main__':
    main()
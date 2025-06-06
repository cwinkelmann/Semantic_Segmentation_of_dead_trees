"""
U-Net Training Script for Semantic Segmentation with Data Augmentation

Installation requirements:
pip install torch torchvision albumentations opencv-python matplotlib tqdm pillow numpy

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
- Apply data augmentations during training
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
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_augmentation(img_size, augmentation_prob=0.5, rotation_limit=45,
                              brightness_contrast_limit=0.2, hue_saturation_limit=20):
    """
    Create training augmentation pipeline optimized for forestry/aerial imagery

    Augmentations included:
    - Geometric: Flips, rotations (simulate different viewing angles)
    - Photometric: Brightness, contrast, hue changes (simulate different lighting/seasons)
    - Weather: Noise, blur (simulate atmospheric conditions)
    - Shadow simulation (important for forest canopy analysis)
    """
    train_transform = [
        # Resize to target size
        A.Resize(img_size, img_size),

        # Geometric augmentations (good for aerial imagery)
        A.HorizontalFlip(p=augmentation_prob),
        A.VerticalFlip(p=augmentation_prob),
        A.RandomRotate90(p=augmentation_prob),
        A.ShiftScaleRotate(
            shift_limit=0.0625,  # Small shifts
            scale_limit=0.1,  # Small scaling
            rotate_limit=rotation_limit,
            interpolation=1,
            border_mode=0,
            p=augmentation_prob
        ),

        # Optical/atmospheric augmentations (simulate different lighting/weather)
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=brightness_contrast_limit,
                contrast_limit=brightness_contrast_limit,
                p=1.0
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            A.HueSaturationValue(
                hue_shift_limit=hue_saturation_limit,
                sat_shift_limit=hue_saturation_limit,
                val_shift_limit=hue_saturation_limit,
                p=1.0
            ),
        ], p=augmentation_prob),

        # Weather/atmospheric effects
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.Blur(blur_limit=3, p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=augmentation_prob * 0.3),  # Lower probability for noise

        # Shadow simulation (important for forestry)
        A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),  # Lower half more likely to have shadows
            num_shadows_lower=1,
            num_shadows_upper=2,
            shadow_dimension=5,
            p=augmentation_prob * 0.3
        ),

        # Normalization (ImageNet stats work well for natural images)
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),

        # Convert to PyTorch tensor
        ToTensorV2(),
    ]

    return A.Compose(train_transform)


def get_validation_augmentation(img_size):
    """
    Create validation augmentation pipeline (only resize and normalize)
    """
    val_transform = [
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]

    return A.Compose(val_transform)


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
            transform (callable, optional): Albumentations transform to be applied
            img_size (tuple): Target size for images and masks
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_size = img_size

        # Get all image files
        self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        if not self.image_files:
            # Try jpg if no png files found
            self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

        self.mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        if not self.mask_files:
            # Try jpg if no png files found
            self.mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.jpg")))

        # Ensure we have matching pairs
        assert len(self.image_files) == len(
            self.mask_files), f"Number of images ({len(self.image_files)}) and masks ({len(self.mask_files)}) don't match"

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

        # Convert to numpy arrays
        image = np.array(image)
        mask = np.array(mask)

        # Normalize mask values (assuming binary segmentation: 0 and 255 -> 0 and 1)
        mask = (mask > 128).astype(np.uint8)  # Keep as uint8 for albumentations

        # Apply augmentations if provided
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

            # Convert mask to long tensor for loss calculation
            mask = mask.long()
        else:
            # Fallback: basic preprocessing without augmentation
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

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
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
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')
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

        print(f'Epoch {epoch + 1}/{num_epochs}:')
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
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

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
            if image.dim() == 3:  # If image is normalized, denormalize for visualization
                # Denormalize using ImageNet stats
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image_np = image.permute(1, 2, 0).numpy()
                image_np = std * image_np + mean
                image_np = np.clip(image_np, 0, 1)
            else:
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
    plt.savefig('predictions_visualization_augmentation.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_augmentations(dataset, transform, num_samples=4):
    """Visualize augmentation effects on sample images"""
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    for i in range(num_samples):
        # Get a sample
        idx = np.random.randint(0, len(dataset))

        # Get original image and mask (without transform)
        image_path = dataset.image_files[idx]
        mask_path = dataset.mask_files[idx]

        image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = (mask > 128).astype(np.uint8)

        # Resize for consistency
        from albumentations import Resize
        resize_transform = Resize(256, 256)
        resized = resize_transform(image=image, mask=mask)
        image_resized = resized['image']
        mask_resized = resized['mask']

        # Apply augmentation
        augmented = transform(image=image_resized, mask=mask_resized)
        aug_image = augmented['image']
        aug_mask = augmented['mask']

        # Convert tensor back to numpy for visualization
        if isinstance(aug_image, torch.Tensor):
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            aug_image_np = aug_image.permute(1, 2, 0).numpy()
            aug_image_np = std * aug_image_np + mean
            aug_image_np = np.clip(aug_image_np, 0, 1)
        else:
            aug_image_np = aug_image / 255.0

        if isinstance(aug_mask, torch.Tensor):
            aug_mask_np = aug_mask.numpy()
        else:
            aug_mask_np = aug_mask

        # Plot
        axes[i, 0].imshow(image_resized)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask_resized, cmap='gray')
        axes[i, 1].set_title('Original Mask')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(aug_image_np)
        axes[i, 2].set_title('Augmented Image')
        axes[i, 2].axis('off')

        axes[i, 3].imshow(aug_mask_np, cmap='gray')
        axes[i, 3].set_title('Augmented Mask')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig('augmentation_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    # =============================================================================
    # CONFIGURATION - Edit these variables to match your setup
    # =============================================================================
    IMAGE_DIR = "/Users/christian/Downloads/15168163/Training_data/Train"  # Directory containing training images (.png)
    MASK_DIR = "/Users/christian/Downloads/15168163/Training_data/Mask"  # Directory containing training masks (.png)
    EPOCHS = 20  # Number of training epochs
    BATCH_SIZE = 16  # Batch size for training
    LEARNING_RATE = 1e-4  # Learning rate
    IMG_SIZE = 256  # Input image size (will resize to IMG_SIZE x IMG_SIZE)
    VAL_SPLIT = 0.2  # Validation split ratio (0.2 = 20% for validation)
    N_CLASSES = 1  # Number of output classes (1 for binary segmentation)

    # AUGMENTATION SETTINGS
    USE_AUGMENTATION = True  # Enable/disable data augmentation
    VISUALIZE_AUGMENTATIONS = True  # Show augmentation examples before training
    AUGMENTATION_PROB = 0.5  # Probability for most augmentations
    ROTATION_LIMIT = 45  # Max rotation angle in degrees
    BRIGHTNESS_CONTRAST_LIMIT = 0.2  # Brightness/contrast variation
    HUE_SATURATION_LIMIT = 20  # Hue/saturation variation
    # =============================================================================

    # Set device and configure DataLoader settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f'Using device: {device}')

    # Configure DataLoader settings based on device
    use_pin_memory = device.type == 'cuda'
    num_workers = 2 if device.type == 'mps' else 4  # Reduce workers for MPS

    # Create augmentation transforms
    if USE_AUGMENTATION:
        print("Creating augmentation pipelines...")
        train_transform = get_training_augmentation(
            img_size=IMG_SIZE,
            augmentation_prob=AUGMENTATION_PROB,
            rotation_limit=ROTATION_LIMIT,
            brightness_contrast_limit=BRIGHTNESS_CONTRAST_LIMIT,
            hue_saturation_limit=HUE_SATURATION_LIMIT
        )
        val_transform = get_validation_augmentation(img_size=IMG_SIZE)
        print("✓ Augmentation pipelines created")
    else:
        print("Augmentation disabled - using basic preprocessing only")
        train_transform = None
        val_transform = None

    # Create full dataset first
    full_dataset = SegmentationDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        transform=None,  # We'll apply transforms after splitting
        img_size=(IMG_SIZE, IMG_SIZE)
    )

    # Split dataset indices
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(VAL_SPLIT * dataset_size))

    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Create separate datasets with different transforms
    train_dataset = SegmentationDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        transform=train_transform,
        img_size=(IMG_SIZE, IMG_SIZE)
    )

    val_dataset = SegmentationDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        transform=val_transform,
        img_size=(IMG_SIZE, IMG_SIZE)
    )

    # Create subset samplers
    from torch.utils.data import SubsetRandomSampler
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    print(f'Training samples: {len(train_indices)}')
    print(f'Validation samples: {len(val_indices)}')

    # Visualize augmentations if requested
    if USE_AUGMENTATION and VISUALIZE_AUGMENTATIONS:
        print("Generating augmentation examples...")
        visualize_augmentations(train_dataset, train_transform, num_samples=3)

    # Create data loaders with samplers
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
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
    plt.savefig('training_history_augmentations.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Visualize some predictions
    visualize_predictions(model, train_dataset, device)

    print('Training completed! Best model saved as best_unet_model.pth')
    print('To use this model for inference, load it with:')
    print('checkpoint = torch.load("best_unet_model.pth")')
    print('model.load_state_dict(checkpoint["model_state_dict"])')

    if USE_AUGMENTATION:
        print(f'\nAugmentation settings used:')
        print(f'- Augmentation probability: {AUGMENTATION_PROB}')
        print(f'- Rotation limit: {ROTATION_LIMIT}°')
        print(f'- Brightness/contrast limit: {BRIGHTNESS_CONTRAST_LIMIT}')
        print(f'- Hue/saturation limit: {HUE_SATURATION_LIMIT}')


if __name__ == '__main__':
    main()
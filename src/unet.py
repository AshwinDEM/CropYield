# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchsummary import summary

# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()

#         # Encoder (Downsampling path)
#         self.enc1 = self.double_conv(3, 64)
#         self.enc2 = self.double_conv(64, 128)
#         self.enc3 = self.double_conv(128, 256)
#         self.enc4 = self.double_conv(256, 512)

#         # Bottleneck
#         self.bottleneck = self.double_conv(512, 1024)

#         # Decoder (Upsampling path)
#         self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
#         self.dec4 = self.double_conv(1024, 512)
        
#         self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
#         self.dec3 = self.double_conv(512, 256)
        
#         self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.dec2 = self.double_conv(256, 128)
        
#         self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.dec1 = self.double_conv(128, 64)

#         # Output layer
#         self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

#     def double_conv(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         # Encoder
#         enc1 = self.enc1(x)
#         enc2 = self.enc2(self.downsample(enc1))
#         enc3 = self.enc3(self.downsample(enc2))
#         enc4 = self.enc4(self.downsample(enc3))

#         # Bottleneck
#         bottleneck = self.bottleneck(self.downsample(enc4))

#         # Decoder
#         dec4 = self.upconv4(bottleneck)
#         dec4 = torch.cat((dec4, enc4), dim=1)
#         dec4 = self.dec4(dec4)
        
#         dec3 = self.upconv3(dec4)
#         dec3 = torch.cat((dec3, enc3), dim=1)
#         dec3 = self.dec3(dec3)
        
#         dec2 = self.upconv2(dec3)
#         dec2 = torch.cat((dec2, enc2), dim=1)
#         dec2 = self.dec2(dec2)
        
#         dec1 = self.upconv1(dec2)
#         dec1 = torch.cat((dec1, enc1), dim=1)
#         dec1 = self.dec1(dec1)

#         # Output
#         out = self.out_conv(dec1)
#         return torch.sigmoid(out)

#     def downsample(self, x):
#         return nn.MaxPool2d(kernel_size=2, stride=2)(x)

# # Initialize the model
# model = UNet()

# # Move the model to GPU if available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)

# # Print model summary
# summary(model, input_size=(3, 256, 256))

# # Define the loss function and optimizer
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-4)


# # Assuming train_loader and val_loader are defined DataLoader objects

# num_epochs = 50

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0

#     for images, masks in train_loader:
#         images, masks = images.to(device), masks.to(device)

#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, masks)

#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item() * images.size(0)

#     epoch_loss = running_loss / len(train_loader.dataset)
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

#     # Validate the model on the validation set if needed
#     model.eval()
#     # Add validation loop here if needed


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

class CropDataset(Dataset):
    def __init__(self, image_folder, mask_folder, transform=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform
        self.image_files = sorted(os.listdir(image_folder))
        self.mask_files = sorted(os.listdir(mask_folder))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        mask_path = os.path.join(self.mask_folder, self.mask_files[idx])

        image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB
        mask = Image.open(mask_path).convert('L')  # Ensure mask is in grayscale

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return {'image': image, 'mask': mask}

# Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset and DataLoader
dataset = CropDataset(image_folder='test/images', mask_folder='test/masks', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)  # Single channel output
        )
        self.upconv = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.encoder(x)
        mid = self.middle(enc1)
        dec = self.upconv(mid)
        dec = torch.cat((enc1, dec), dim=1)
        dec = self.decoder(dec)
        return dec


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(10):  # Adjust number of epochs as needed
    model.train()
    running_loss = 0.0
    
    for batch in dataloader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device).unsqueeze(1)  # Add channel dimension for masks

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks.float())  # Ensure masks are float type
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')

# Save the model
torch.save(model.state_dict(), 'unet_model.pth')


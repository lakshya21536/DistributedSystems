import torch
import os
import torchaudio
import torchvision
import math
import torch.nn as nn
from Pipeline import *
from torch.utils.data import Dataset, DataLoader, Subset

# Image Downloader
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_image_dataset_downloader = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,transform=transform)
test_image_dataset_downloader = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,transform=transform)

train_size = int(0.8 * len(train_image_dataset_downloader))
val_size = len(train_image_dataset_downloader) - train_size
train_image_dataset, val_image_dataset = torch.utils.data.random_split(train_image_dataset_downloader, [train_size, val_size])

class ImageDataset(Dataset):
    def __init__(self,split:str="train")->None:
        super().__init__()
        if split not in ["train", "test", "val"]:
            raise Exception("Data split must be in [train, test, val]")
        
        self.datasplit = split
        self.dataset = None  
        if split=='train':
            self.dataset=train_image_dataset
        elif split=='test':
            self.dataset=test_image_dataset_downloader
        elif split=='val':
            self.dataset=val_image_dataset
            
    def __len__(self)->int:
        return len(self.dataset)    
    
    def __getitem__(self, index:int)->tuple:
        image, label = self.dataset[index]
        return image, label

# Audio Downloader
train_audio_dataset_downloader = torchaudio.datasets.SPEECHCOMMANDS(root='./data',download=True)
# test_audio_dataset_downloader = torchaudio.datasets.SPEECHCOMMANDS(root='./data',download=True)

# train_size = int(0.8 * len(train_audio_dataset_downloader))
# val_size = len(train_audio_dataset_downloader) - train_size
# train_audio_dataset, val_audio_dataset = torch.utils.data.random_split(train_audio_dataset_downloader, [train_size, val_size])

class AudioDataset(Dataset):
    def __init__(self, split: str = "train") -> None:
        super().__init__()

        if split not in ["train", "test", "val"]:
            raise Exception("Data split must be in [train, test, val]")

        self.datasplit = split
        self.datasplit = split
        self.dataset = None  
        if split=='train':
            print("hehehehe")
            self.dataset=train_audio_dataset_downloader
            # print(self.dataset.labels)
        elif split=='test':
            self.dataset=test_audio_dataset_downloader
        elif split=='val':
            self.dataset=val_audio_dataset

    def __len__(self):
        return len(self.dataset)

    def pad_waveform(self, waveform, max_size):
        return torch.nn.functional.pad(waveform, (0, max_size - waveform.size(1)))

    def __getitem__(self, index):
        try:
            print(f"Calling __getitem__ for index: {index}")
            waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[index]
            print(f"Label: {label}")
            
            # Calculate max_size based on the waveforms in the current batch
            max_size = max(item[0].size(1) for item in self.dataset[:index + 1])
            print(f"Max Size: {max_size}")

            padded_waveform = self.pad_waveform(waveform, max_size)
            return padded_waveform, label
        except Exception as e:
            print(f"Error in __getitem__ for index {index}: {e}")
            raise e


# class AudioDataset(Dataset):
#     def __init__(self, split:str="train") -> None:
#         super().__init__()
#         if split not in ["train", "test", "val"]:
#             raise Exception("Data split must be in [train, test, val]")
        
#         self.datasplit = split
#         self.dataset = torchaudio.datasets.SPEECHCOMMANDS(root='./data', download=True)
    
#     def __len__(self):
#         return len(self.dataset)
    
#     def __getitem__(self, index):
#         waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[index]
#         return waveform, label

# class Resnet_Q1(nn.Module):
#     def __init__(self,
#                  *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         """
#         Write your code here
#         """

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection when input and output dimensions are not the same
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # if(self.shortcut):
        identity = self.shortcut(identity)  
            
        # out += self.shortcut(identity)
        out+=identity
        out = self.relu(out)

        return out

class Resnet_Q1(nn.Module):
    def __init__(self, num_classes=10):
        super(Resnet_Q1, self).__init__()
        self.block=BasicBlock
        self.num_blocks=[4, 4, 4, 6]
        self.in_channels = 64

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks
        self.layer1 = self._make_layer(self.block, 64, self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, self.num_blocks[3], stride=2)

        # Average pooling and fully connected layer
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
        
# class VGG_Q2(nn.Module):
#     def __init__(self,
#                  *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         """
#         Write your code here
#         """

class VGG_Q2(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG_Q2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, math.ceil(64 * 0.65), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(math.ceil(64 * 0.65), math.ceil(64 * 0.65), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(math.ceil(64 * 0.65), math.ceil(128 * 0.65), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(math.ceil(128 * 0.65), math.ceil(128 * 0.65), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(math.ceil(128 * 0.65), math.ceil(128 * 0.65), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(math.ceil(128 * 0.65), math.ceil(256 * 0.65), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(math.ceil(256 * 0.65), math.ceil(256 * 0.65), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(math.ceil(256 * 0.65), math.ceil(256 * 0.65), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(math.ceil(256 * 0.65), math.ceil(512 * 0.65), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(math.ceil(512 * 0.65), math.ceil(512 * 0.65), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(math.ceil(512 * 0.65), math.ceil(512 * 0.65), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(math.ceil(512 * 0.65) * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
# class Inception_Q3(nn.Module):
#     def __init__(self,
#                  *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         """
#         Write your code here
#         """


class Inception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Inception, self).__init__()

        # 1x1 Convolution Block
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[0], kernel_size=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(inplace=True)
        )

        # 5x5 Convolution Block
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[1], kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels[1]),
            nn.ReLU(inplace=True)
        )

        # 3x3 Convolution Block inside 5x5 Block
        self.conv3x3_5x5 = nn.Sequential(
            nn.Conv2d(out_channels[1], out_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels[2]),
            nn.ReLU(inplace=True)
        )

        # 3x3 Max Pooling Block
        self.maxpool3x3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Forward pass through each branch and concatenate the results
        conv1x1_output = self.conv1x1(x)
        conv5x5_output = self.conv5x5(x)
        conv3x3_5x5_output = self.conv3x3_5x5(conv5x5_output)
        maxpool3x3_output = self.maxpool3x3(x)

        inception_output = torch.cat([conv1x1_output, conv3x3_5x5_output, maxpool3x3_output], dim=1)

        return inception_output

class Inception_Q3(nn.Module):
    def __init__(self, num_classes=10):
        super(Inception_Q3, self).__init__()

        # Define the architecture using Inception_Q3 blocks
        self.block1 = Inception(3, [64, 128, 128])
        self.block2 = Inception(384, [192, 384, 256])
        self.block3 = Inception(1024, [256, 512, 256])
        self.block4 = Inception(1024, [384, 512, 384])

        # Global Average Pooling and Fully Connected Layer
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Forward pass through each block
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Global Average Pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # Fully Connected Layer
        x = self.fc(x)

        return x
            
class CustomNetwork_Q4(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Write your code here
        """

best_accuracy = 0.0
best_model_state_dict = None

def trainer(gpu="F", dataloader=None, network=None, criterion=None, optimizer=None):
    device = torch.device("cuda:0") if gpu == "T" else torch.device("cpu")
    network = network.to(device)
    
    global best_accuracy
    global best_model_state_dict
    # checkpoint_dir="checkpoints"

    # if not os.path.exists(checkpoint_dir):
    #     os.makedirs(checkpoint_dir)

    for epoch in range(EPOCH):
        network.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        i=0
        print(f"Epoch {epoch+1}/{EPOCH}")

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            print(labels)

            optimizer.zero_grad()

            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(i," ",loss.item())
            i+=1

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        accuracy = correct_predictions / total_samples
        print("Training Epoch: {}, [Loss: {}, Accuracy: {}]".format(epoch, total_loss, accuracy))

        # Save checkpoint
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state_dict = network.state_dict().copy()
            
        # checkpoint_filename = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': network.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': total_loss,
        #     'accuracy': accuracy,
        # }, checkpoint_filename)

        # print(f"Checkpoint saved at {checkpoint_filename}")


def validator(gpu="F", dataloader=None, network=None, criterion=None, optimizer=None):
    device = torch.device("cuda:0") if gpu == "T" else torch.device("cpu")
    
    global best_accuracy
    global best_model_state_dict

    # Load checkpoint
    # checkpoint = torch.load("C:\\Users\\Dell\\OneDrive\\Desktop\\DL\\Assignment-2\\checkpoints\\checkpoint_epoch_3.pth")
    # network.load_state_dict(checkpoint['model_state_dict'])
    # network = network.to(device)
    # network.eval()
        
    for epoch in range(EPOCH):
        network.load_state_dict(best_model_state_dict)
        network = network.to(device)
        network.eval()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = network(inputs)
                loss = criterion(outputs, labels)
                print(loss.item())

                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        accuracy = correct_predictions / total_samples
        average_loss = total_loss / len(dataloader)

        print("Validation Epoch: {}, [Loss: {}, Accuracy: {}]".format(epoch, total_loss, accuracy))


def evaluator(gpu="F", dataloader=None, network=None, criterion=None, optimizer=None):
    device = torch.device("cuda:0") if gpu == "T" else torch.device("cpu")

    global best_accuracy
    global best_model_state_dict
    # Load checkpoint
    # checkpoint = torch.load("C:\\Users\\Dell\\OneDrive\\Desktop\\DL\\Assignment-2\\checkpoints\\checkpoint_epoch_3.pth")
    # network.load_state_dict(checkpoint['model_state_dict'])
    # network = network.to(device)
    # network.eval()
    
    for epoch in range(EPOCH):
        network.load_state_dict(best_model_state_dict)
        network = network.to(device)
        network.eval()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = network(inputs)
                loss = criterion(outputs, labels)

                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        accuracy = correct_predictions / total_samples
        average_loss = total_loss / len(dataloader)

        print("Validation Epoch: {}, [Loss: {}, Accuracy: {}]".format(epoch, total_loss, accuracy))
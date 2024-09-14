import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
import os

# # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
# from scipy.interpolate import interp1d

parser = argparse.ArgumentParser(description="Template")
# # Dataset options

# #Data - Data needs to be pre-filtered and filtered data is available

# ### BLOCK DESIGN ###
# #Data
#parser.add_argument('-ed', '--eeg-dataset', default=r"data\block\eeg_55_95_std.pth", help="EEG dataset path") #55-95Hz
parser.add_argument('-ed', '--eeg_dataset', default='/root/autodl-tmp/DreamDiffusion/datasets/mne_data/eeg_5_95_std.pth', help="EEG dataset path") #5-95Hz
#parser.add_argument('-ed', '--eeg-dataset', default=r"data\block\eeg_14_70_std.pth", help="EEG dataset path") #14-70Hz
#Splits
parser.add_argument('-sp', '--splits-path', default='/root/autodl-tmp/DreamDiffusion/datasets/mne_data/block_splits_by_image_all.pth', help="splits path") #All subjects
#parser.add_argument('-sp', '--splits-path', default=r"data\block\block_splits_by_image_single.pth", help="splits path") #Single subject
### BLOCK DESIGN ###

parser.add_argument('-sn', '--split-num', default=0, type=int, help="split number") #leave this always to zero.

#Subject selecting
parser.add_argument('-sub','--subject', default=0, type=int, help="choose a subject from 1 to 6, default is 0 (all subjects)")

#Time options: select from 20 to 460 samples from EEG data
parser.add_argument('-tl', '--time_low', default= 20, type=float, help="lowest time value")
parser.add_argument('-th', '--time_high', default= 460,  type=float, help="highest time value")

# Model type/options
parser.add_argument('-mt','--model_type', default='lstm', help='specify which generator should be used: lstm|EEGChannelNet')
# It is possible to test out multiple deep classifiers:
# - lstm is the model described in the paper "Deep Learning Human Mind for Automated Visual Classification”, in CVPR 2017
# - model10 is the model described in the paper "Decoding brain representations by multimodal learning of neural activity and visual features", TPAMI 2020
parser.add_argument('-mp','--model_params', default='', nargs='*', help='list of key=value pairs of model options')
parser.add_argument('--pretrained_net', default='lstm__subject0_epoch_500.pth', help="path to pre-trained net (to continue training)")

# Training options
parser.add_argument("-b", "--batch_size", default=128, type=int, help="batch size")
parser.add_argument('-o', '--optim', default="Adam", help="optimizer")
parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, help="learning rate")
parser.add_argument('-lrdb', '--learning-rate-decay-by', default=0.5, type=float, help="learning rate decay factor")
parser.add_argument('-lrde', '--learning-rate-decay-every', default=10, type=int, help="learning rate decay period")
parser.add_argument('-dw', '--data-workers', default=4, type=int, help="data loading workers")
parser.add_argument('-e', '--epochs', default=500, type=int, help="training epochs")

# Save options
parser.add_argument('-sc', '--saveCheck', default=100, type=int, help="learning rate")

# Backend options
parser.add_argument('--no-cuda', default=False, help="disable CUDA", action="store_true")

# Parse arguments
opt = parser.parse_args()
# print(opt)



# # Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import torch; torch.utils.backcompat.broadcast_warning.enabled = True
# from torch.utils.data import DataLoader
# import torch.optim
# import torch.backends.cudnn as cudnn; cudnn.benchmark = True
# import torch; torch.utils.backcompat.broadcast_warning.enabled = True
# from torch.utils.data import DataLoader
# import numpy as np





# # Dataset class
# class EEGDataset:

#     # Constructor
#     def __init__(self, eeg_signals_path):
#         # Load EEG signals
#         loaded = torch.load(eeg_signals_path)
#         if opt.subject != 0:
#             self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset'])) if
#                          loaded['dataset'][i]['subject'] == opt.subject]
#         else:
#             self.data = loaded['dataset']
#         self.labels = loaded["labels"]
#         self.images = loaded["images"]

#         # Compute size
#         self.size = len(self.data)
#         self.data_len = 512

#     # Get size
#     def __len__(self):
#         return self.size

#     # Get item
#     def __getitem__(self, i):
#         # Process EEG
#         eeg = self.data[i]["eeg"].float().t()
#         eeg = eeg[opt.time_low:opt.time_high, :]

#         eeg = np.array(eeg.transpose(0, 1))
#         x = np.linspace(0, 1, eeg.shape[-1])
#         x2 = np.linspace(0, 1, self.data_len)
#         f = interp1d(x, eeg)
#         eeg = f(x2)
#         eeg = torch.from_numpy(eeg).float()

#         if opt.model_type == "model10":
#             eeg = eeg.t()
#             eeg = eeg.view(1, 128, opt.time_high - opt.time_low)
#         # Get label
#         # label = self.data[i]["label"]
#         label = torch.tensor(self.data[i]["label"]).long()

#         # Return
#         return eeg, label

# # Splitter class
# class Splitter:

#     def __init__(self, dataset, split_path, split_num=0, split_name="train"):
#         # Set EEG dataset
#         self.dataset = dataset
#         # Load split
#         loaded = torch.load(split_path)
#         self.split_idx = loaded["splits"][split_num][split_name]
#         # Filter data
#         self.split_idx = [i for i in self.split_idx if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
#         # Compute size
#         self.size = len(self.split_idx)
#         self.num_voxels = 440
#         self.data_len = 512

#     # Get size
#     def __len__(self):
#         return self.size

#     # Get item
#     def __getitem__(self, i):
#         # Get sample from dataset
#         eeg, label = self.dataset[self.split_idx[i]]
#         # Return
#         return eeg, label


# # Load dataset
# dataset = EEGDataset(opt.eeg_dataset)
# eeg_dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
# # Create loaders
# loaders = {split: DataLoader(Splitter(dataset, split_path=opt.splits_path, split_num=opt.split_num, split_name=split),
#                              batch_size=opt.batch_size, drop_last=True, shuffle=True) for split in
#            ["train", "val", "test"]}
# # model_ckpt = torch.load('cnn_model.ckpt')

# # Hyper parameters
# num_epochs = 100
# num_classes = 40
# batch_size = opt.batch_size
# learning_rate = 0.001

# # MNIST dataset
# train_dataset = torchvision.datasets.MNIST(root='../../data/',
#                                            train=True,
#                                            transform=transforms.ToTensor(),
#                                            download=True)
#
# test_dataset = torchvision.datasets.MNIST(root='../../data/',
#                                           train=False,
#                                           transform=transforms.ToTensor())
#
# # Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)

# Convolutional neural network (two convolutional layers) 1
class ConvNet(nn.Module):
    def __init__(self, num_classes=40):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(128, 1024, kernel_size=5, stride=1, padding=2),
            # nn.Conv1d(in_channels=128, out_channels=512, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv1d(512, 16, kernel_size=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(8, 8, kernel_size=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


        self.class_embedding = nn.Embedding(num_classes, 128)
        self.pos_embed = nn.Parameter(torch.zeros(67, 128),
                                      requires_grad=False)  # fixed sin-cos embedding

        # torch.Size([100, 2048])
        self.fc = nn.Linear(2 * 2 * 80, 40)

    def forward(self, x, target):
#         class_embedding = self.class_embedding(target).view(1, 128)

        class_embedding = torch.tensor(self.class_embedding(target).repeat(1,1,1)).to(device)
#         class_embedding = class_embedding.repeat(1,1,1).view(1, 128, 1)
        
#         class_embedding = class_embedding.expand(x.shape[0], -1, -1)
#         if class_embedding.shape[0] % batch_size != 0 :
#             class_embedding = torch.cat((class_embedding,self.pos_embed), dim=0)
#         class_embedding_ = class_embedding.expand(x.shape[0], -1, -1)

#         out = torch.cat((class_embedding_, x), dim=2)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = out.reshape(out.size(0), -1)
#         out = self.fc(out)
        return class_embedding

# model = ConvNet(num_classes).to(device)
# model.load_state_dict(torch.load('/root/autodl-tmp/DreamDiffusion/class_cnn_model.ckpt'))

# Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
# total_step = len(eeg_dataloader)



# train
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(eeg_dataloader):
#         images = images.to(device)
#         labels = labels.to(device)
#
#         # Forward pass
#         outputs = model(images,labels)
#         loss = criterion(outputs, labels)
#
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         t = i + 1
#         # if (i+1) % 100 == 0:
#         print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
#                .format(epoch+1, num_epochs, i+1, total_step, loss.item()))



# # Test the model
# model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in loaders["test"]:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images,labels)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#     print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
# torch.save(model.state_dict(), 'class_cnn_model.ckpt')
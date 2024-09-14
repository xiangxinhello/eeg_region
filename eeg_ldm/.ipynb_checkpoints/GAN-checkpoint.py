from sklearn.metrics import confusion_matrix
import seaborn as sns

from    torch.nn import functional as F
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from resnet18 import resnet18
from torch import nn, optim
import numpy as np
import pandas as pd
import seaborn as sns
from dataset import eeg_pretrain_dataset
from torch.utils.data import DataLoader
from dataset import  create_EEG_dataset_GAN


seed = 2022

latent_size = 64
hidden_size = 256
image_size = 31
num_epochs = 200
batch_size = 100

# create dataset and dataloader
# dataset_pretrain = EEGDataset(path='../datasets/mne_data/')
local_rank = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset_pretrain = eeg_pretrain_dataset(path='../datasets/mne_data/')
# print(f'Dataset size: {len(dataset_pretrain)}\n Time len: {dataset_pretrain.data_len}')
# sampler = torch.utils.data.DistributedSampler(dataset_pretrain,
#                                               rank=local_rank) if torch.cuda.device_count() > 1 else None
#
# dataloader_eeg = DataLoader(dataset_pretrain, batch_size=batch_size, sampler=sampler,
#                             shuffle=(sampler is None), pin_memory=True)
#
# for data_iter_step, (data_dcit) in enumerate(dataloader_eeg):
#     samples = data_dcit['eeg']



eeg_latents_dataset_train, eeg_latents_dataset_test = create_EEG_dataset_GAN(
    eeg_signals_path='/xiangxin_project/DreamDiffusion-main/datasets/mne_data/eeg_5_95_std.pth',
    splits_path= '/xiangxin_project/DreamDiffusion-main/datasets/mne_data/block_splits_by_image_single.pth', subject = 4
)

train_loader = DataLoader(eeg_latents_dataset_train, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(eeg_latents_dataset_test, batch_size=batch_size, shuffle=False)
special_dimension = 65536
# sampler = torch.utils.data.DistributedSampler(dataset_pretrain,
#                                               rank=local_rank) if torch.cuda.device_count() > 1 else None

# train_loader = DataLoader(dataset_pretrain, batch_size=batch_size, sampler=sampler,
#                             shuffle=(sampler is None), pin_memory=True)


# 定义卷积神经网络模型
# 定义一个卷积神经网络模型
class ConvNet(nn.Module):
    def __init__(self, input_channels, output_size):
        super(ConvNet, self).__init__()
        self.conv1d = nn.Conv1d(input_channels, output_size, kernel_size=3)  # 一维卷积层
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        self.fc = nn.Linear(output_size, 65536)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # 将张量展平
        x = self.fc(x)
        return x

class Self_Attn(nn.Module):
    '''
    Self attention Layer
    '''

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.l1 = nn.Linear(31,512)
        self.l2 = nn.Linear(512,256)
        self.l3 = nn.Linear(256,128)
        self.l4 = nn.Linear(128,64)
        self.l5 = nn.Linear(64,31)

        self.query_conv = nn.Linear(31, 3)   #30x3
        self.key_conv = nn.Linear(31, 3)     #30x3
        self.value_conv = nn.Linear(31, 100)  #30x31
        self.gamma = nn.Parameter(torch.zeros(1))
        #         self.out = nn.Linear()
        self.softmax = nn.Softmax(dim=-1)

        self.l6 = nn.Linear(100, 128)
        self.l7 = nn.Linear(128, 64)
        self.l8 = nn.Linear(64, 32)
        self.l9 = nn.Linear(32, 30)
        # self.l2 = nn.Linear(9, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = self.l5(x)

        proj_query = F.relu(self.query_conv(x))
        # print("1:")
        # print(proj_query.shape)
        proj_key = F.relu(self.key_conv(x))
        # print("2:")
        # print(proj_key.shape)
        proj_key = proj_key.reshape(3, 100)

        energy = torch.matmul(proj_query, proj_key)  # 20x3 20x3 -> 20x20
        #         energy = ener
        # print(energy.shape)
        attention = self.softmax(energy)

        proj_value = F.relu(self.value_conv(x))     #20x30
        # print(proj_value.shape)
        proj_value = proj_value.reshape(100, 100)     #
        out1 = torch.matmul(proj_value, attention)  # 20x20 20x20 -> 20x20
        # out1 = F.relu(out1)
        out1 = F.relu(self.l6(out1))
        out1 = F.relu(self.l7(out1))
        out1 = F.relu(self.l8(out1))
        out1 = self.l9(out1)


        # out = F.relu(self.l2(out))
        # out = out.view(10,-1)
        return out1

class linear(nn.Module):
    def __init__(self):
        super(linear,self).__init__()
        self.l1 = nn.Linear(31,128)
        self.l2 = nn.Linear(128,64)
        self.l3 = nn.Linear(64,32)
        self.l4 = nn.Linear(32, 9)

    def forward(self,x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x

# Discriminator
class Discriminator(nn.Module):
    '''
   Discriminator Layer
   '''


    def __init__(self, in_dim):
        super(Discriminator, self).__init__()
        self.chanel_in = in_dim
        self.l10 = nn.Linear(65536, hidden_size) #20xhidden_size 256
        self.batch1 = nn.BatchNorm1d(hidden_size) #256
        self.relu = nn.LeakyReLU(0.1)

        self.l12 = nn.Linear(hidden_size, 31) #20x31
        self.batch2 = nn.BatchNorm1d(31)  # 256
        self.relu = nn.LeakyReLU(0.1)
        self.l13 = nn.Linear(30, 128)
        self.batch3 = nn.BatchNorm1d(128)
        self.sig = nn.Sigmoid()

        self.l14 = nn.Linear(128, 64)
        self.batch4 = nn.BatchNorm1d(64)
        self.l15 = nn.Linear(64, 31)
        self.l22 = nn.Linear(31,20)
        self.batch5 = nn.BatchNorm1d(31)
        self.l16 = nn.Linear(30, 128)
        self.l17 = nn.Linear(128,64)
        self.l18 = nn.Linear(64,32)
        self.l19 = nn.Linear(32,16)
        self.l20 = nn.Linear(16,8)
        self.l21 = nn.Linear(8,20)
        self.batch6 = nn.BatchNorm1d(30)
        self.attn1 = Self_Attn(3)
        self.attn2 = Self_Attn(3)

    def forward(self, x1):
        x1 = self.l10(x1)
        x1 = self.batch1(x1)
        x1 = self.relu(x1)

        x1 = self.l12(x1)
        x1 = self.batch2(x1)
        x1 = self.relu(x1) #20x31

        x1 = self.attn1(x1) #20x30
        # print(x1.shape)

        x1 = self.l13(x1)  #20x128
        x1 = self.batch3(x1)
        x1 = self.relu(x1)

        x1 = self.l14(x1)
        x1 = self.batch4(x1)
        x1 = self.relu(x1)

        x1 = self.l15(x1)  #20x31
        x1 = self.batch5(x1)
        x1 = self.relu(x1)

        x1 = self.l22(x1)

        # x1 = self.attn2(x1)   #20x9
        # x1 = self.batch6(x1)
        # x1 = self.relu(x1)

        # x1 = self.l16(x1)
        # x1 = self.l17(x1)
        # x1 = self.l18(x1)
        # x1 = self.l19(x1)
        # x1 = self.l20(x1)
        # x1 = self.l21(x1)
        x1 = self.sig(x1)

        return x1

# Discriminator
class Generator(nn.Module):

    '''
   Discriminator Layer
   '''

    def __init__(self, in_dim):
        super(Generator, self).__init__()
        self.chanel_in = in_dim
        self.l10 = nn.Linear(special_dimension, hidden_size) #20xhidden_size 256
        self.batch1 = nn.BatchNorm1d(hidden_size) #256
        self.relu = nn.ReLU()

        self.l12 = nn.Linear(hidden_size, 31) #20x31
        self.batch2 = nn.BatchNorm1d(31)  # 256
        self.relu = nn.ReLU()
        self.l13 = nn.Linear(30, 128)
        self.batch3 = nn.BatchNorm1d(128)
        self.sig = nn.Sigmoid()

        self.l14 = nn.Linear(128, 64)
        self.batch4 = nn.BatchNorm1d(64)
        self.l15 = nn.Linear(64, 31)
        self.l22 = nn.Linear(31,special_dimension)
        # self.l221 = nn.Linear(31,31)
        # self.l222 = nn.Linear(31,31)
        self.batch5 = nn.BatchNorm1d(31)
        self.l16 = nn.Linear(30, 128)
        self.l17 = nn.Linear(128,64)
        self.l18 = nn.Linear(64,32)
        self.l19 = nn.Linear(32,25)
        self.l20 = nn.Linear(25,8)
        self.l21 = nn.Linear(8,9)
        self.batch6 = nn.BatchNorm1d(30)
        self.attn1 = Self_Attn(3)
        self.attn2 = Self_Attn(3)

    def forward(self, x1):
        x1 = self.l10(x1)
        x1 = self.batch1(x1)
        x1 = self.relu(x1)

        x1 = self.l12(x1)
        x1 = self.batch2(x1)
        x1 = self.relu(x1) #20x31

        x1 = self.attn1(x1) #20x30
        # print(x1.shape)

        x1 = self.l13(x1)  #20x128
        x1 = self.batch3(x1)
        x1 = self.relu(x1)

        x1 = self.l14(x1)
        x1 = self.batch4(x1)
        x1 = self.relu(x1)

        x1 = self.l15(x1)  #20x31
        x1 = self.batch5(x1)
        x1 = self.relu(x1)

        x1 = self.l22(x1)

        # x1 = self.attn2(x1)   #20x9
        # x1 = self.batch6(x1)
        # x1 = self.relu(x1)

        # x1 = self.l16(x1)
        # x1 = self.l17(x1)
        # x1 = self.l18(x1)
        # x1 = self.l19(x1)
        # x1 = self.l20(x1)
        # x1 = self.l21(x1)

        return x1

modelD = Discriminator(in_dim=31).to(device)
modelG = Generator(in_dim=31).to(device)
# model = rnn().to(device)
model18 = resnet18().to(device)

conv_liner = ConvNet(input_channels=128, output_size=784).to(device)

# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(modelD.parameters(), lr=0.0002)
g_optimizer = optim.Adam(modelG.parameters(), lr=0.0002)


criterionll = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model18.parameters(), lr=0.01)


def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

total_step = len(train_loader)
a=[]
b=[]

modelG.load_state_dict(torch.load('generator.pth'))
tensorList = []
labeList = []
# 进行推理
with torch.no_grad():
    for i, (data_dict) in enumerate(train_loader):
        samples = data_dict['eeg'].to(device)
        labels = data_dict['label']

        batch_size_for = samples.shape[0]

        if (batch_size_for == batch_size):
            output_conv_liner = conv_liner(samples)
            fake = modelG(output_conv_liner)
            tensorList.append(fake)
            labeList.append(labels)

numpy_array_label = []
for tensor in labeList:
    number = tensor.shape[0]
    for i in range(0, number):
        numpy_array_label.append(tensor[i])

numpy_array_ = numpy_array_label
numpy_array_label_ = np.array([label.numpy() for label in numpy_array_])
np.save('combined_tensors_label.npy', numpy_array_label)


numpy_array = []
for tensor in tensorList:
    number = tensor.shape[0]
    tensor = tensor.to('cpu').view(100, 128, 512)
    for i in range(0, number):
        numpy_array.append(tensor[i:i+1, :, :])

numpy_array_bug = numpy_array
numpy_array_sc = []
for numpy_array_bug_bug in numpy_array_bug:
    numpy_array_sc.append(numpy_array_bug_bug.view(128,512))

numpy_array = np.array([tensor.numpy() for tensor in numpy_array_sc])
np.save('combined_tensors.npy', numpy_array)



# for tensor in tensorList:
#     tensor = tensor.to('cpu').view(100,128,512)
#     numpy_array.append(tensor)



# numpy_array = np.array([tensor.numpy() for tensor in numpy_array])
# numpy_array_label = np.array([label.numpy() for label in labeList])
#
# np.save('combined_tensors.npy', numpy_array)
# np.save('combined_tensors_label.npy', numpy_array_label)




# for epoch in range(600):
#     for i, (data_dict) in enumerate(train_loader):
#         samples = data_dict['eeg'].to(device)
#         labels = data_dict['label'].to(device)
#
#         # print(samples.shape)
#         # print(labels.shape)
#         batch_size_for = samples.shape[0]
#
#         if(batch_size_for == batch_size):
#             output_conv_liner = conv_liner(samples)
#
#             noise = torch.randn(batch_size, special_dimension).to(device)
#             fake = modelG(noise)
#
#             Dreal = modelD(output_conv_liner)  # .view(-1)
#             real_score = Dreal
#
#             lossDreal = criterion(Dreal, torch.ones_like(Dreal))
#             Dfake = modelD(fake).view(-1)
#             fake_score = Dfake
#
#             lossDfake = criterion(Dfake, torch.zeros_like(Dfake))
#             lossD = (lossDfake + lossDreal) / 2
#
#             reset_grad()
#             lossD.backward(retain_graph=True)
#             d_optimizer.step()
#
#             # Train the generator
#             fake = modelG(noise)
#             output = modelD(fake).view(-1)
#             lossG = criterion(output, torch.ones_like(output))
#             reset_grad()
#             lossG.backward()
#             g_optimizer.step()
#
#             print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
#                   .format(epoch, num_epochs, i + 1, total_step, lossD.item(), lossG.item(), real_score.mean().item(),
#                           fake_score.mean().item()))
# #
# torch.save(modelG.state_dict(), 'generator.pth')











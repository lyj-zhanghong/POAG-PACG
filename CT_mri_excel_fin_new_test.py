import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models
from PIL import Image
import pandas as pd
import os
import cv2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib import pyplot as plt
#from keras.models import load_model
#import keras.backend as K
from torchvision.models import densenet121
#import tensorflow as tf
import os
import cv2
import numpy as np
from sklearn.metrics import f1_score
#from keras.regularizers import l2
from torch.nn import DataParallel
from sklearn.metrics import roc_curve, auc
import sys


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MultiModalNet(nn.Module):
    def __init__(self, cnn_net, reduction=16):
        super(MultiModalNet, self).__init__()

        # DenseNet121 for CT image
        densenet_ct = densenet121(pretrained=True)
        self.densenet_ct = nn.Sequential(*list(densenet_ct.features.children())[:-1])
        self.se1 = SELayer(1024, reduction=reduction)
        self.fusion_conv1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.fusion_bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU()
        self.se2 = SELayer(512, reduction=reduction)
        self.fusion_conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.fusion_bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()
        self.se3 = SELayer(256, reduction=reduction)
        self.fc1 = nn.Linear(256, 128)

        # DenseNet121 for MRI image
        densenet_mri = densenet121(pretrained=True)
        self.densenet_mri = nn.Sequential(*list(densenet_mri.features.children())[:-1])
        self.fusion_conv3 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.se4 = SELayer(1024, reduction=reduction)
        self.fusion_bn3 = nn.BatchNorm2d(512)
        self.relu3 = nn.ReLU()
        self.se5 = SELayer(512, reduction=reduction)
        self.fusion_conv4 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.fusion_bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.se6 = SELayer(256, reduction=reduction)
        self.fc2 = nn.Linear(256, 128)

        densenet_th2 = densenet121(pretrained=True)
        self.densenet_th2 = nn.Sequential(*list(densenet_th2.features.children())[:-1])
        self.fusion_conv5 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.se7 = SELayer(1024, reduction=reduction)
        self.fusion_bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()
        self.se8 = SELayer(512, reduction=reduction)
        self.fusion_conv6 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.fusion_bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU()
        self.se9 = SELayer(256, reduction=reduction)
        self.fc3 = nn.Linear(256, 128)

        self.cnn_net = cnn_net

        # Classification
        self.f3 = nn.Linear(512, 128)  # Concatenated feature size: 512 (256 from fc1 + 256 from fc2

        self.f4 = nn.Linear(128, 2)

    def forward(self, a, b, x_ct, x_th1,x_th2, feature):
        # Forward pass for CT image
        ct_x = self.se1(self.densenet_ct(x_ct))
        #ct_x =  self.densenet_ct(x_ct)
        ct_x = self.se2(self.relu1(self.fusion_bn1(self.fusion_conv1(ct_x))))
        #ct_x = self.relu1(self.fusion_bn1(self.fusion_conv1(ct_x)))
        ct_x = F.max_pool2d(ct_x, 2)
        ct_x = self.se3(self.relu2(self.fusion_bn2(self.fusion_conv2(ct_x))))
        #ct_x = self.relu2(self.fusion_bn2(self.fusion_conv2(ct_x)))
        ct_x = F.max_pool2d(ct_x, 2)
        ct_x = ct_x.view(ct_x.size(0), -1)  # Flatten the features
        # ct_x = 0.97*ct_x
        ct_x = self.fc1(ct_x)

        # Forward pass for MRI image
        th1_x = self.se4(self.densenet_mri(x_th1))
        #th1_x =  self.densenet_mri(x_th1)
        th1_x = self.se5(self.relu3(self.fusion_bn3(self.fusion_conv3(th1_x))))
        #th1_x = self.relu3(self.fusion_bn3(self.fusion_conv3(th1_x)))
        th1_x = F.max_pool2d(th1_x, 2)
        th1_x = self.se6(self.relu4(self.fusion_bn4(self.fusion_conv4(th1_x))))
        #th1_x = self.relu4(self.fusion_bn4(self.fusion_conv4(th1_x)))
        th1_x = F.max_pool2d(th1_x, 2)
        th1_x = th1_x.view(th1_x.size(0), -1)  # Flatten the features
        th1_x = self.fc2(th1_x)
        # mri_x = 0.97*mri_x

        th2_x = self.se7(self.densenet_th2(x_th2))
        #th2_x = self.densenet_th2(x_th2)
        th2_x = self.se8(self.relu5(self.fusion_bn5(self.fusion_conv5(th2_x))))
        #th2_x = self.relu5(self.fusion_bn5(self.fusion_conv5(th2_x)))
        th2_x = F.max_pool2d(th2_x, 2)
        th2_x = self.se9(self.relu6(self.fusion_bn6(self.fusion_conv6(th2_x))))
        #th2_x = self.relu6(self.fusion_bn6(self.fusion_conv6(th2_x)))
        th2_x = F.max_pool2d(th2_x, 2)
        th2_x = th2_x.view(th2_x.size(0), -1)  # Flatten the features
        th2_x = self.fc2(th2_x)

        cnn_x = self.cnn_net(feature)

        # Concatenate the features
        concat_features = torch.cat((ct_x, th1_x,th2_x,cnn_x), dim=1)

        # Classification
        out = self.f3(concat_features)
        out = self.f4(out)
        return out


class CNNNeuralNetwork(torch.nn.Module):
    def __init__(self, dropout_prob=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(32, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv1d(128, 512, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(512)
        self.relu4 = nn.ReLU()

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(512, 128)

    def forward(self, x):
        x = self.relu1(self.conv1(x.unsqueeze(1)))
        x = self.pool(x)
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, data_column_names, transform=None):
        self.data = pd.read_excel(csv_file)  # Assuming your file is in excel format
        self.data = self.data[self.data['图片'] == 1]
        self.root_dir = root_dir
        self.transform = transform
        # Fill missing values with zeros
        self.data[data_column_names] = self.data[data_column_names].fillna(0)

        # Calculate mean and std for each column
        self.column_means = self.data[data_column_names].mean()
        self.column_stds = self.data[data_column_names].std()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        feature = self.data.iloc[idx][data_column_names].values.astype(np.float32)
        feature = (feature - self.column_means) / self.column_stds
        feature[np.isnan(feature)] = 0
        feature_tensor = torch.Tensor(feature.values)
        img_folder = str(self.data.iloc[idx, 1])  # Assuming the ID is in the third column
        eye_num = str(self.data.iloc[idx, -2])
        img_folder_path = f"{self.root_dir}/{img_folder}/{eye_num}"  # Modify the image file extension accordingly
        img_folder_path1 = f"{self.root_dir}/{img_folder}"
        label = int(self.data.iloc[idx, 4])  # Assuming the label is in the sixth column
        # img_ct = torch.zeros((3, 190, 190))  # Change the shape according to your image size
        # img_mri = torch.zeros((3, 224, 224))  # Change the shape according to your image size
        img_filenames1 = [f for f in os.listdir(img_folder_path1) if f.endswith('.jpeg') and eye_num in f]
        #print(img_filenames1)
        if not img_filenames1:
            raise FileNotFoundError(f"No .jpeg images found for {img_folder}/{eye_num}")
        img_path_th1 = os.path.join(img_folder_path1,img_filenames1[0])
        img_filenames2 = [f for f in os.listdir(img_folder_path1) if f.endswith('BScan.jpg') and eye_num in f]
        #print(img_filenames2)
        if not img_filenames2:
            raise FileNotFoundError(f"No BScan.jpg images found for {img_folder}/{eye_num}")
        img_path_th2 = os.path.join(img_folder_path1, img_filenames2[0])
        # Get the list of image filenames in the folder
        img_filenames = os.listdir(img_folder_path)
        # Sort the image filenames based on the numerical values extracted from filenames
        img_filenames.sort(key=lambda x: int(x[-10:-5]))  # Assuming filenames have two-digit numerical values

        # Construct paths to CT and MRI images based on the sorted filenames
        img_path_ct = os.path.join(img_folder_path, img_filenames[0])
        #img_path_mri = os.path.join(img_folder_path, img_filenames[1])

        # print(f"img_path_ct: {img_path_ct}")
        # print(f"img_path_mri: {img_path_mri}")

        # Load images
        img_ct = Image.open(img_path_ct)
        img_th1 = Image.open(img_path_th1)
        img_th2 = Image.open(img_path_th2)
        #img_mri = Image.open(img_path_mri)

        # img_ct = Image.open(img_path_ct).convert('L')  # Convert to grayscale
        # img_mri = Image.open(img_path_mri).convert('L')  # Convert to grayscale
        if eye_num == 'OS':
            # 水平翻转图像
            img_ct = img_ct.transpose(Image.FLIP_LEFT_RIGHT)
            img_th1 = img_th1.transpose(Image.FLIP_LEFT_RIGHT)
            img_th2 = img_th2.transpose(Image.FLIP_LEFT_RIGHT)
            #img_mri = img_mri.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform:
            # print(11111111111111)
            img_ct = self.transform(img_ct)
            img_th1 = self.transform(img_th1)
            img_th2 = self.transform(img_th2)
            #img_mri = self.transform(img_mri)

        return feature_tensor, img_ct,img_th1,img_th2,label


def train(model, device, train_loader, optimizer, epoch, output_file, train_losses, train_accuracies, a, b):
    model.train()
    total_loss = 0.0  # Initialize the total loss
    correct = 0
    total_samples = 0

    for batch_idx, (feature,  ct_data, th1_data, th2_data,target) in enumerate(train_loader):
        feature,  ct_data,th1_data,th2_data, target = feature.to(device),  ct_data.to(device), th1_data.to(device),th2_data.to(device),target.to(
            device)
        optimizer.zero_grad()
        output = model(a, b, ct_data, th1_data,th2_data,feature)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate batch accuracy
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += ct_data.size(0)

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(ct_data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

            output_file.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                epoch, batch_idx * len(ct_data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    # Calculate average loss and accuracy for the epoch
    average_loss = total_loss / len(train_loader.dataset)
    train_losses.append(average_loss)  # Append the average loss to the list

    accuracy = 100. * correct / total_samples
    train_accuracies.append(accuracy)  # Append the accuracy to the list

    print('Train Epoch: {} \tAverage Loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)'.format(
        epoch, average_loss, correct, total_samples, accuracy))
    output_file.write('Train Epoch: {} \tAverage Loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        epoch, average_loss, correct, total_samples, accuracy))


def vaild(model, device, test_loader, epoch, test_accuracy_list, output_file, test_losses, a, b):
    model.eval()
    test_loss = 0
    correct = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, (feature, ct_data,th1_data,th2_data, target) in enumerate(test_loader):
            feature, ct_data, th1_data, th2_data, target = feature.to(device), ct_data.to(device), th1_data.to(device), th2_data.to(device), target.to(device)
            output = model(a, b, ct_data, th1_data,th2_data, feature)
            loss = F.cross_entropy(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Collect labels and predictions for F1 score calculation
            all_labels.extend(target.cpu().numpy())
            # print("pred shape:", pred.shape)
            '''
            if pred.ndim == 0:
                all_preds.append(pred.item())
            else:
                all_preds.extend(pred.cpu().numpy().squeeze())
            test_loss += loss.item()
            '''

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    test_accuracy_list.append(accuracy)
    test_losses.append(test_loss)

    # Calculate F1 score
    #f1 = f1_score(all_labels, all_preds)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    output_file.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    return accuracy, test_loss


def test(best_model, device, val_dataset, epoch, output_file, a, b):
    best_model.eval()
    test_loss = 0
    correct = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for feature,  ct_data,th1_data,th2_data, target in val_dataset:
            feature, ct_data, th1_data, th2_data, target = feature.to(device), ct_data.to(device), th1_data.to(device), th2_data.to(device), target.to(
                device)
            output = best_model(a, b, ct_data, th1_data,th2_data, feature)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            probs = torch.softmax(output, dim=1)  # 获取预测的概率
            all_probs.extend(probs.cpu().numpy()[:, 1])  # 只存储正类的概率
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Collect labels and predictions for F1 score calculation
            all_labels.extend(target.cpu().numpy())
            # print("pred shape:", pred.shape)
            '''
            if pred.ndim == 0:
                all_preds.append(pred.item())
            else:
                all_preds.extend(pred.cpu().numpy().squeeze())
            '''

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    # Calculate F1 score
    #f1 = f1_score(all_labels, all_preds)

    print('\nValidation set using best model: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(val_dataset.dataset), accuracy))

    output_file.write('\nValidation set using best model: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(val_dataset.dataset), accuracy))

    return accuracy,all_probs,all_labels


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据增强
data_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=30),  # 随机旋转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 随机颜色调整
    transforms.RandomVerticalFlip(),  # 随机垂直翻转
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机仿射变换
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Provide the path to your CSV file and image folder
csv_file = '../data_change/no_eye_112_all_shuttle_renew.xlsx'
root_dir = '../data_change/data/all_picture'
data_column_names = ['年龄', 'BCVA', '眼压', 'RNFL', '视野VFI', '视野MD', '视野PSD']
# 存储测试准确率的列表
test_accuracy_list = []
train_losses = []
test_losses = []
train_accuracies = []
# Instantiate MultiModalNet and CNNNeuralNetwork

# Initialize the custom dataset
custom_dataset = CustomDataset(csv_file=csv_file, root_dir=root_dir, data_column_names=data_column_names,
                               transform=data_transform)

# Split the dataset into training and testing sets (you can use a more sophisticated split)
train_size = int(0.6 * len(custom_dataset))
val_size = int(0.2 * len(custom_dataset))
test_size = len(custom_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(custom_dataset,
                                                                         [train_size, val_size, test_size],
                                                                         generator=torch.Generator().manual_seed(0))

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # 作为训练
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # 作为验证
val_dataset = DataLoader(val_dataset, batch_size=1, shuffle=False)  # 作为测试
# multi_modal_net = MultiModalNet().to(device)
# cnn_net = CNNNeuralNetwork().to(device)

# Instantiate FusionModel
# model = FusionModel(multi_modal_net, cnn_net).to(device)
model1 = CNNNeuralNetwork().to(device)
model = MultiModalNet(model1).to(device)
model = DataParallel(model)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=2, verbose=True)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
output_file_path = '1111'
best_val_loss = 100.0
best_epoch = 0

best_model = MultiModalNet(CNNNeuralNetwork()).to(device)
best_model = DataParallel(best_model)
best_accuracy = 80
a = 0.97
b = 0.97
epoch = 1
with open(output_file_path, 'w') as output_file:
    while a == 0.97 and b == 0.97:
        model_save_path = '../saved_models/for_a_b_fin_112_610_vaild_excel_linear_concat_best_model_a0.97_b0.97.pth'
        best_model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
        test_acc,all_probs,all_labels=test(best_model, device, val_dataset, epoch, output_file, a, b)
        test_acc, all_probs, all_labels = test(best_model, device, val_dataset, epoch, output_file, a, b)
        test_acc, all_probs, all_labels = test(best_model, device, val_dataset, epoch, output_file, a, b)
        test_acc, all_probs, all_labels = test(best_model, device, val_dataset, epoch, output_file, a, b)
        test_acc, all_probs, all_labels = test(best_model, device, val_dataset, epoch, output_file, a, b)
        test_acc, all_probs, all_labels = test(best_model, device, val_dataset, epoch, output_file, a, b)
        test_acc, all_probs, all_labels = test(best_model, device, val_dataset, epoch, output_file, a, b)
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve_vaild.png')  # 保存图片
        plt.show()
        a -= 0.1
        b -= 0.1


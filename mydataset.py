import random
import torch
import torch.nn as nn


class LabelDataset(torch.utils.data.Dataset):
    '''
    data: List of paths
    label: List of labels
    '''
    def __init__(self, data, label, transform=None):
        self.transform = transform
        self.data = data
        self.data_num = len(data)
        self.label = label

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.transform(torch.load(self.data[idx], weights_only=True))
        out_label = torch.tensor(self.label[idx], dtype=torch.long)
        return out_data, out_label


class NormalizeTensor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: torch.Tensor (C, W)
        xmin = torch.min(x)
        xmax = torch.max(x)
        return (2 * x - xmax - xmin) / (xmax - xmin)


class InjectSignalIntoNoise(nn.Module):
    def __init__(self, std=1.0):
        super().__init__()
        self.std = std

    def forward(self, x):
        # x: torch.Tensor (C, W)
        noise = torch.empty_like(x, dtype=torch.float32).normal_(0.0, std=self.std)
        return x + noise


class MimicTimeTranslation(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.length = length

    def forward(self, x):
        # x: torch.Tensor(C, W)
        length_original = x.size()[1]
        dl = length_original - self.length
        kstart = random.randint(0, dl - 1)
        return x[:, kstart: kstart + self.length]


class MyDataset(torch.utils.data.Dataset):
    '''
    data: List of paths
    label: List of labels
    '''
    def __init__(self, signaldata, transform=None):
        self.transform = transform
        self.signaldata = signaldata
        self.nulldata = torch.zeros_like(signaldata, dtype=torch.float32)
        self.data = torch.cat((self.signaldata, self.nulldata))
        self.data_num = len(self.data)
        label_cbc = torch.ones((self.data_num // 2, 1), dtype=torch.long)
        label_null = torch.zeros((self.data_num // 2, 1), dtype=torch.long)
        self.label = torch.cat((label_cbc, label_null))[:, 0]

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.transform(self.data[idx])
        out_label = self.label[idx]
        return out_data, out_label

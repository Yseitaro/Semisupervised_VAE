import sys
sys.path.append('..')
from src.data.EMG_DataLoader import EMG_DataLoader
from src.utils import train_epoch
import os,sys
import torch
import torch.optim as optim
from torch import nn

from src.utils import fix_seed
from src.models.model import SemisupervisedVAE
from src.utils import *
from torch.utils.data import TensorDataset, DataLoader


if __name__ == "__main__":

    labeled_data_manipulator = EMG_DataLoader(1,[0])
    labeled_data_manipulator.set_dataset()
    labeled_data_manipulator.set_dataloader(30)
    labeled_data_loader = labeled_data_manipulator.dataloader

    unlabeled_data_manipulator = EMG_DataLoader(1,[1,2,3,4,5])
    unlabeled_data_manipulator.set_dataset()
    unlabeled_data_manipulator.set_dataloader(30)
    unlabeled_data_loader = unlabeled_data_manipulator.dataloader

    lx = labeled_data_manipulator.data_x
    ly = labeled_data_manipulator.data_y
    ux = unlabeled_data_manipulator.data_x
    uy = unlabeled_data_manipulator.data_y

    lx, ly = torch.Tensor(lx), torch.Tensor(ly)
    ux, uy = torch.Tensor(ux), torch.Tensor(uy)
    lx, ly = lx.to('cpu'), ly.to('cpu')
    ux, uy = ux.to('cpu'), uy.to('cpu')
    print(f'lx {lx.size()}')
    print(f'ux {ux.size()}')

    dataset = TensorDataset(lx, ly, ux, uy)
    data_loader = DataLoader(dataset, batch_size=30, shuffle=True)

    for lx, ly, ux, uy in data_loader:
        break
    print(lx.size())
    print(ly.size())
    print(ux.size())
    print(uy.size())
    
    # model = SemisupervisedVAE(z_dim=4,y_dim=6,input_shape=180,device='cpu')
    
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # loss_value = []
    # for epoch in range(50):
    #     loss,_ = train_epoch(model,data_loader,optimizer,epoch,device='cpu')
    #     loss_value.append(loss)


    
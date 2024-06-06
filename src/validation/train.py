import os, sys
sys.path.append('..')
import numpy as np
import torch
import torch.nn.functional as nnf
from torch.utils.data import TensorDataset
from tqdm import tqdm
from sklearn.metrics import *
from src.models.loss import LossFunctions


def fix_seed(seed: int, verbose: bool=False) -> None:
    """乱数シードを固定する関数
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    if verbose:
        print('fixed seeds')


def reshape_dataset_to_tensor(X, y=None, structure='static'):
    """Reshape dataset to tensor
    """
    X_tensor = torch.from_numpy(X).float()
    if structure == 'dynamic':
        X_tensor = torch.unsqueeze(X_tensor, dim=1)
    if y is not None:
        y_tensor = torch.from_numpy(y).long()
        return TensorDataset(X_tensor, y_tensor)
    else:
        return X_tensor

def split_train_valid(whole_train_set, valid_ratio=0.2):
    """Split train and valudation set
    """
    n_samples = len(whole_train_set)
    val_size = int(len(whole_train_set) * valid_ratio)
    train_size = n_samples - val_size

    train_dataset, val_dataset = \
        torch.utils.data.random_split(whole_train_set, [train_size, val_size])

    return train_dataset, val_dataset


def train_epoch(model, data_loader, optimizer, epoch, device):
    """モデルの学習に関する機能を集約した関数
    """
    # Set model to train mode
    model.train()

    epoch_loss = 0.0
    epoch_corrects = 0.0
    loss_fn = LossFunctions()
    # n_size = 0

    # Batch loop
    for data, targets in data_loader:
        # n_size+=1
        data, targets = data.to(device), targets.to(device)

        data = data.view(len(data),-1)
        one_hot_targets = nnf.one_hot(targets, num_classes=6).float()

        outputs = model(data,one_hot_targets) # get output
        loss = 0.0
        #再構成誤差
        # print(outputs['x_hat'].size())
        # print(f'data.shape : {data.size()}')
        loss_recon = loss_fn.reconstruction_loss(data,outputs['x_hat'])
        loss_gaussian = loss_fn.gaussian_loss(outputs['z'], outputs['z_mu'], outputs['z_logvar'], outputs['z_mean_prior'], outputs['z_logvar_prior'])
        loss_cat = loss_fn.entropy(outputs['y_logits'], one_hot_targets) 
        # print(f'loss_recon : {loss_recon.size()}')
        # print(f'loss_gaussian : {loss_gaussian.size()}')
        # print(f'loss_cat : {loss_cat.size()}') 
        # loss_py = - np.log(0.1)
        # print(f'loss_recon : {loss_recon}')
        # print(f'loss_gaussian : {loss_gaussian}')
        # print(f'loss_cat : {loss_cat}')
        loss = torch.mean(loss_recon + 100*loss_gaussian + loss_cat) - np.log(0.1)
        preds = torch.argmax(outputs['y_prb'], 1) # calculate predicted label

        epoch_loss += loss.item()*data.size(0)
        epoch_corrects += torch.sum(preds == targets)

        optimizer.zero_grad()
        loss.backward() # backpropagation
        optimizer.step() # update parameters

    epoch_loss = epoch_loss/len(data_loader.dataset)
    epoch_acc = epoch_corrects/len(data_loader.dataset)
    print(f'Epoch #{epoch+1}: train loss = {epoch_loss:.4f}, train acc = {epoch_acc:.4f}')

    return epoch_loss, epoch_acc


    # epoch_loss = epoch_loss / (len(data_loader.dataset))
    # epoch_acc = epoch_corrects / (len(data_loader.dataset))

    # print(f'Epoch #{epoch+1}: train loss = {epoch_loss:.4f}, train acc = {epoch_acc:.4f}')

    # return epoch_loss, epoch_acc


def val_epoch(model, data_loader, criterion, device):
    """検証データに関する機能を集約した関数
    """
    # Set model to evaluation mode
    model.eval()

    epoch_loss = 0.0
    epoch_corrects = 0.0

    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)

            outputs = model(data) # Get output

            loss = criterion(outputs, targets) # calculate loss
            _, preds = torch.max(outputs, 1) # calculate predicted label

            epoch_loss += loss.item() * data.size(0)
            epoch_corrects += torch.sum(preds == targets.data)

    epoch_loss = epoch_loss / (len(data_loader.dataset))
    epoch_acc = epoch_corrects / (len(data_loader.dataset))

    # print(f' -> val loss: {epoch_loss:.4f}, val acc: {epoch_acc:.4f}')

    return epoch_loss, epoch_acc



def eval_epoch(model, data_loader, criterion, device, save_path=None):
    """学習済みモデルを用いた識別に関する機能を集約した関数
    """
    # Set model to evaluation mode
    model.eval()

    epoch_loss = 0.0
    epoch_corrects = 0.0
    
    if save_path:
        with open(f'{save_path}/chunk_result.csv', 'w') as file:
            pass

    with torch.no_grad():
        for data, targets in tqdm(data_loader):
            data, targets = data.to(device), targets.to(device)

            outputs = model(data) # Get output
            probs = nnf.softmax(outputs, dim=1) # calculate posterior probabilities

            loss = criterion(outputs, targets) # calculate loss
            _, preds = torch.max(outputs, 1) # calculate predicted label

            # Calculate loss & accuracy
            epoch_loss += loss.item() * data.size(0)
            epoch_corrects += torch.sum(preds == targets.data)

            # Save chunk results
            if save_path:
                with open(f'{save_path}/chunk_result.csv', 'a') as file:
                    probs = probs.to('cpu').clone().numpy().copy()
                    targets = targets.data.to('cpu').clone().numpy().copy()
                    preds = preds.to('cpu').clone().numpy().copy()
                    for i in range(len(targets)):
                        class_probs = f'{probs[i,0]:.5f}'
                        for j in range(probs.shape[1]-1):
                            class_probs += f',{probs[i,j+1]:.5f}'
                        file.write(f'{class_probs},{preds[i]},{targets[i]}\n')

    epoch_loss = epoch_loss / len(data_loader.dataset)
    epoch_acc = epoch_corrects.double() / len(data_loader.dataset)

    print(f' -> Test loss: {epoch_loss:.4f}, Test acc: {epoch_acc:.4f}')

    return epoch_loss, epoch_acc.to('cpu').clone().numpy().copy()


def checkpoint(acc, epoch, model, save_path):
    # Save checkpoint.
    print('Saving..')
    state = {
        'epoch': epoch,
        'val_acc': acc, 
        'model_state_dict':  model.state_dict(),
        'rng_state': torch.get_rng_state()
    }
    torch.save(state, save_path)
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader,TensorDataset
import torch
import numpy as np



def get_dataloader(num_training=50000,num_labeled=3000,batch_size=200):
    train=MNIST(root='./data/mnist',download=True)
    train_data=(train.train_data.view(-1,784).float()/255.0)
    train_label=train.train_labels

    print(train_data.size())
    print(train_label.size())

    dataset={}
    dataset['data']=[]
    dataset['label']=[]

    dataset['test_data']=[]
    dataset['test_label']=[]

    num_per_class=num_training//10
    num_labeled_per_class=num_labeled//10
    # print(f'num_per_class : {num_per_class}, num_labeled_per_class : {num_labeled_per_class}')
    # print(train_label[:10])

    for i in range(10):
        ind_i=torch.nonzero(train_label==i)[:,0].numpy()
        np.random.shuffle(ind_i)
        # print(ind_i)
        dataset['data'].append(train_data[ind_i[:num_per_class],:])
        dataset['label'].append(train_label[ind_i[:num_per_class]])
        dataset['test_data'].append(train_data[ind_i[num_per_class:],:])
        dataset['test_label'].append(train_label[ind_i[num_per_class:]])

    # print(f'dataset["data"] : {dataset["data"].size()}')
    datas=torch.cat(dataset['data'],0)
    print(f'datas : {datas.size()}')
    labels=torch.cat(dataset['label'],0)
    labels=torch.zeros(labels.size(0),10).scatter_(1, labels.view(-1,1), 1)

    # dataset={}

    dataset['labeled_data']=datas[torch.Tensor(np.concatenate([np.arange(i*num_per_class,i*num_per_class+num_labeled_per_class) for i in range(10)],0)).long(),:]
    print(f'dataset["labeled_data"] : {dataset["labeled_data"].size()}')
    dataset['labeled_label']=labels[torch.Tensor(np.concatenate([np.arange(i*num_per_class,i*num_per_class+num_labeled_per_class) for i in range(10)],0)).long(),:]
    print(f'dataset["labeled_label"] : {dataset["labeled_label"].size()}')

    dataset['unlabeled_data']=datas[torch.Tensor(np.concatenate([np.arange(i*num_per_class+num_labeled_per_class,(i+1)*num_per_class) for i in range(10)],0)).long(),:]
    print(f'dataset["unlabeled_data"] : {dataset["unlabeled_data"].size()}')
    dataset['unlabeled_label']=labels[torch.Tensor(np.concatenate([np.arange(i*num_per_class+num_labeled_per_class,(i+1)*num_per_class) for i in range(10)],0)).long(),:]
    print(f'dataset["unlabeled_label"] : {dataset["unlabeled_label"].size()}')
    
    dataset['test_data']=torch.cat(dataset['test_data'],0)
    dataset['test_label']=torch.cat(dataset['test_label'],0)

    dataloader={}
    dataloader['labeled'] = DataLoader(TensorDataset(dataset['labeled_data'], dataset['labeled_label']),
                                       batch_size=num_labeled // (num_training // batch_size), shuffle=True,
                                       num_workers=4)

    dataloader['unlabeled'] = DataLoader(TensorDataset(dataset['unlabeled_data'], dataset['unlabeled_label']),
                                       batch_size=batch_size-num_labeled // (num_training // batch_size), shuffle=True,
                                       num_workers=4)

    dataloader['test'] = DataLoader(TensorDataset(dataset['test_data'],dataset['test_label']),
                                    batch_size=500,shuffle=False,num_workers=4)

    return dataloader


if __name__=='__main__':
    get_dataloader()
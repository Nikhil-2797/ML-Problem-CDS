import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from utility import class_map_dict


def download_dataset(train_pth=None, val_pth=None):
    if train_pth is None:
        train_pth = '../data/CIFAR10/train'
    cifar10 = datasets.CIFAR10(train_pth, train=True, download=True)
    tr_images = cifar10.data
    tr_targets = cifar10.targets
    if val_pth is None:
        val_pth = '../data/CIFAR10/val'
    cifar10_val = datasets.CIFAR10(val_pth, train=False, download=True)
    val_images = cifar10_val.data
    val_targets = cifar10_val.targets
    return (tr_images, tr_targets), (val_images, val_targets)


class Cifar10_dnn(Dataset):
    def __init__(self, input_images, targets, aug=None):
        self.sample = input_images
        self.targets = targets
        self.aug = aug
        self.transform = transforms.Compose([transforms.Resize(size=(32, 32)), transforms.ToTensor()])

    def __getitem__(self, ndx):
        sample = self.sample[ndx]
        sample = transforms.ToPILImage()(sample)
        if self.aug is not None:
            num_objects = random.randint(0, len(self.aug))
            aug_lst = random.sample(self.aug, num_objects)
            sample = transforms.Compose(aug_lst)(sample)

        sample = self.transform(sample).to(torch.float32)
        data_sample = sample.view(sample.shape[0] * sample.shape[1] * sample.shape[2])
        class_ndx = torch.tensor(self.targets[ndx]).to(torch.long)
        return data_sample, class_ndx

    def __len__(self):
        return len(self.targets)


class Cifar10_cnn(Dataset):
    def __init__(self, input_images, targets, aug=None):
        self.sample = input_images
        self.targets = targets
        self.aug = aug
        self.transform = transforms.Compose([transforms.Resize(size=(32, 32)), transforms.ToTensor()])

    def __getitem__(self, ndx):
        sample = self.sample[ndx]
        sample = transforms.ToPILImage()(sample)
        if self.aug is not None:
            num_objects = random.randint(0, len(self.aug))
            aug_lst = random.sample(self.aug, num_objects)
            sample = transforms.Compose(aug_lst)(sample)

        sample = self.transform(sample).to(torch.float32)
        data_sample = sample
        class_ndx = torch.tensor(self.targets[ndx]).to(torch.long)
        return data_sample, class_ndx

    def __len__(self):
        return len(self.targets)


if __name__ == '__main__':

    ## Information from data

    (tr_images, tr_targets),(val_images,val_targets) = download_dataset()
    unique_values = set(tr_targets)
    print(
        f'tr_images & tr_targets:\n\tX : {tr_images.shape}\n\tY \: {len(tr_targets)}\n\tY-Unique Values : {unique_values}')
    print(f'TASK:\n\t{len(unique_values)} class Classification')
    print(f'UNIQUE CLASSES:\n\t{list(class_map_dict.values())}')

    ## Single batch extraction
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    cifer10_dnn = Cifar10_dnn(tr_images, tr_targets)
    train_dataloader = DataLoader(cifer10_dnn, batch_size=64, shuffle=True, pin_memory=use_cuda)

    # Single data point 
    image, label = cifer10_dnn[10]
    print(f'Sample type and shape : {type(image), image.size()} , Label type and shape: {type(label), label.size()}')

    # Single batch extraction 
    batch_sample, batch_target = next(iter(train_dataloader))
    print(batch_sample.shape, batch_target.shape)
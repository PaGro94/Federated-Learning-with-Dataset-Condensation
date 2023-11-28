import os
import copy

import torch
import torch.nn as nn

from typing import Tuple, Dict

import utils


def load_data(settings: Dict, device):
    if not os.path.exists(settings['data_path']):
        os.mkdir(settings['data_path'])

    if settings['method'] == "normal":
        channel, num_classes, im_size, trainloader, testloader, datasets_size = utils.get_dataset(settings['dataset'], settings['data_path'], settings['method'])

    elif settings['method'] == "syn_set":
        channel, num_classes, im_size, dst_train, testloader, datasets_size = utils.get_dataset(settings['dataset'], settings['data_path'], settings['method'])

        ''' organize the real dataset '''
        image_all, label_all, indices_class = utils.organize_real_dataset(dst_train=dst_train, num_classes=num_classes, channel=channel, device=device)

        ''' initialize the synthetic data '''
        image_syn, label_syn = utils.initialize_synthetic_data(images_all=image_all, indices_class=indices_class, num_classes=num_classes, channel=channel, im_size=im_size, device=device, settings=settings)

        image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
        dst_syn_train = utils.TensorDataset(image_syn_train, label_syn_train)
        trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=settings['batch_size'], shuffle=True, num_workers=0)

    elif settings['method'] == "pre_opt_syn_set":

        # e.g.: 'datasets/res_DSA_CIFAR10_ConvNet_10ipc.pt'
        filename = "%s/res_%s_%s_%s_%dipc.pt" % (
            settings['datasets_path'],
            settings['method_DC'],
            settings['dataset'],
            settings['model'],
            settings['ipc']
        )

        dataloader = torch.load(filename)
        image_syn_train = dataloader['data'][0][0]
        label_syn_train = dataloader['data'][0][1]
        dst_syn_train = utils.TensorDataset(image_syn_train, label_syn_train)
        trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=settings['batch_size'], shuffle=True, num_workers=0)
        print(len(trainloader.dataset))

        channel, num_classes, im_size, testloader, datasets_size = utils.get_dataset(
            settings['dataset'], settings['data_path'], settings['method'])

    else:
        exit('cannnot load dataset for unkown method: %s' % settings['method'])

    return channel, num_classes, im_size, trainloader, testloader, datasets_size


def load_network(
    model: str,
    channel: int,
    num_classes: int,
    im_size
) -> nn.Module:
    return utils.get_network(model=model, channel=channel, num_classes=num_classes, im_size=im_size)


def train(
    net: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    settings: Dict,
    epochs: int,
    device: torch.device
) -> None:
    """Train the network."""
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=settings['lr'], momentum=0.9)

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")

    # Train the network
    for epoch in range(epochs):  # loop over the dataset multiple times
        for i, data in enumerate(trainloader, 0):
            
            images, labels = data[0].to(device), data[1].to(device)

            if settings['method'] == 'syn_set' or settings['method'] == 'pre_opt_syn_set':

                if settings['method_DC'] == 'DSA':
                    dsa_params = utils.ParamDiffAug()
                    images = utils.DiffAugment(images, settings['dsa_strategy'], param=dsa_params)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def test(
    net: nn.Module,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float]:

    """Validate the network on the entire test set."""
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return loss, accuracy

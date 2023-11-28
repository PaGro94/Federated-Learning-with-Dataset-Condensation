from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

import time
import os

import flwr as fl

import dc_net


DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DCClient(fl.client.NumPyClient):
    """Flower client implementing CIFAR-10 image classification using
    PyTorch."""

    def __init__(
        self,
        net: nn.Module,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        datasets_size: Dict,
        settings: Dict,
        loss_list: list,
        accuracy_list: list,
    ) -> None:

        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.datasets_size = datasets_size
        self.settings = settings
        self.loss_list = loss_list
        self.accuracy_list = accuracy_list

    def get_parameters(
        self,
        config
    ) -> List[np.ndarray]:

        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(
        self,
        parameters: List[np.ndarray]
    ) -> None:

        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:

        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)

        dc_net.train(self.net, trainloader=self.trainloader, settings=self.settings, device=DEVICE, epochs=self.settings['epochs_client'])

        return self.get_parameters(config={}), self.datasets_size["trainset"], {}

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:

        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = dc_net.test(self.net, self.testloader, device=DEVICE)
        self.loss_list.append(loss)
        self.accuracy_list.append(accuracy)
        print("Loss: ", float(loss))
        print("Accuracy: ", float(accuracy))
        return float(loss), self.datasets_size["testset"], {"accuracy": float(accuracy)}


    def save_results(self, runtime) -> None:
        method = np.array(self.settings['method'])
        method_DC = np.array(self.settings['method_DC'])
        dataset = np.array(self.settings['dataset'])
        model = np.array(self.settings['model'])

        ipc = np.array(self.settings['ipc'])

        epochs_server = np.array(self.settings['epochs_server'])
        epochs_client = np.array(self.settings['epochs_client'])

        batch_size = np.array(self.settings['batch_size'])
        lr = np.array(self.settings['lr'])

        dsa_strategy = np.array(self.settings['dsa_strategy'])

        time_arr = np.array(runtime)

        loss_arr = np.vstack((
            np.arange(len(self.loss_list)),
            np.array(self.loss_list)
        )).T

        accuracy_arr = np.vstack((
            np.arange(len(self.accuracy_list)),
            np.array(self.accuracy_list)
        )).T

        if not os.path.exists(self.settings['save_path']):
            os.mkdir(self.settings['save_path'])

        if self.settings['method'] == 'normal':
            filename = "%s/res_%s_%s_epoch_s%d-epoch_c%d_lr%f_%s.npz" % (
                self.settings['save_path'],
                self.settings['model'],
                self.settings['dataset'],
                self.settings['epochs_server'],
                self.settings['epochs_client'],
                self.settings['lr'],
                self.settings['method'],
            )
        else:
            filename = "%s/res_%s_%s_ipc%d_epoch_s%d-epoch_c%d_lr%f_%s_%s.npz" % (
                self.settings['save_path'],
                self.settings['model'],
                self.settings['dataset'],
                self.settings['ipc'],
                self.settings['epochs_server'],
                self.settings['epochs_client'],
                self.settings['lr'],
                self.settings['method'],
                self.settings['method_DC'],
            )

        np.savez(
            file=filename,
            method=method,
            method_DC=method_DC,
            dataset=dataset,
            model=model,
            ipc=ipc,
            epochs_server=epochs_server,
            epochs_client=epochs_client,
            batch_size=batch_size,
            lr=lr,
            dsa_strategy=dsa_strategy,
            time=time_arr,
            loss=loss_arr,
            accuracy=accuracy_arr
        )


def main() -> None:
    """Load data, start CifarClient."""

    with open('config.yml', 'r') as file:
        settings = yaml.safe_load(file)

    # Load model and data
    channel, num_classes, im_size, trainloader, testloader, datasets_size = dc_net.load_data(settings=settings, device=DEVICE)

    net = dc_net.load_network(model=settings['model'], channel=channel, num_classes=num_classes, im_size=im_size)
    net.to(DEVICE)

    # create client
    client = DCClient(net, trainloader=trainloader, testloader=testloader, datasets_size=datasets_size, settings=settings, loss_list=[], accuracy_list=[])

    start_time = time.time()
    # Start FL-Client and connect to Server
    fl.client.start_numpy_client(server_address=settings['ip_address'], client=client)

    end_time = time.time()
    runtime = end_time - start_time
    print("time: ", runtime)
    print("mean: ", np.mean(np.array(client.accuracy_list)))
    print("std: ", np.std(np.array(client.accuracy_list)))

    client.save_results(runtime)


if __name__ == "__main__":
    main()

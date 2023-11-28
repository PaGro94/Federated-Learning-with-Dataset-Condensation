
import torch
import yaml

import flwr as fl

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main() -> None:

    with open('../../client/client/config.yml', 'r') as file:
        settings = yaml.safe_load(file)

    strategy = fl.server.strategy.FedAvg(min_fit_clients=3, min_evaluate_clients=3, min_available_clients=3)

    """"Start FL-Server."""
    fl.server.start_server(server_address=settings['ip_address'], strategy=strategy, config=fl.server.ServerConfig(num_rounds=settings['epochs_server']))


if __name__ == "__main__":
    main()
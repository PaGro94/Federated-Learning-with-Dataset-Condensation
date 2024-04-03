# Federated-Learning-with-Dataset-Condensation

 ---
Abstract:
To increase the security in a Federated Learning environment this work proposes a
combination of Federated Learning with Dataset Condensation. A technique to synthesize
the information of a large dataset into a smaller dataset. We evaluate federated trained
models with and without Dataset Condensation with good performance but
not similar accurate results. Further, we compare the accuracy of the proposed algorithm
with Dataset Condensation with Differentiable Siamese Augmentation in a nonfederated
environment. And conclude that Dataset Condensation performs similarly
in both environments.
---

# Introduction

In Federated Learning the training process is split between clients and a server. Each
client trains a copy of the same initial neural network on its device. After several
training rounds (called *epochs*), each client distributes the resulting weight to the
server for an optimization process. This process is happening in rounds
(called *communication rounds*) as well.

In this project, the server is represented by the *server.py* is a
[*flower*-framework server](https://flower.ai/). It depends on a configuration file
located in the client directory (*config.yml*).

The client part is split into multiple files. The neural network is defined in
*network.py*. *client.py* represents a [*flower*-framework client](https://flower.ai/).
The training and testing process itself is implemented in the *dc_net.py* file
and the *utils.py* file keeps many utility functions.
To configure the clients you have to use the given *config.yml*.
Further information is given in my [Thesis](documentation/Thesis.pdf)
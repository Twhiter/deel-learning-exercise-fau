import numpy as np
import pandas as pd
import torch as t
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.nn import BCELoss

import model
from data import ChallengeDataset
from trainer import Trainer

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename='7.log',
    filemode='w',
    format='%(asctime)s - %(message)s'
)

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
data = pd.read_csv('data.csv', sep=';')

# create an instance of our ResNet model

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion

critic = BCELoss()


def train(configs):
    training_set, vali_set = train_test_split(data, train_size=0.8)
    net = model.ResNet()

    # set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
    training_data = t.utils.data.DataLoader(ChallengeDataset(training_set, mode='train'),batch_size=32)
    val_data = t.utils.data.DataLoader(ChallengeDataset(vali_set, mode='val'), batch_size=32)

    optimizer = t.optim.Adam(net.parameters(), weight_decay=configs['weight_decay'], lr=configs['lr'])

    trainer = Trainer(model=net, crit=critic, optim=optimizer, train_dl=training_data, val_test_dl=val_data,
                      early_stopping_patience=40)

    logging.info(f"\n\n{configs} starts:\n")

    # go, go, go... call fit on trainer
    res = trainer.fit(1000)

    # plot the results
    plt.clf()
    plt.plot(np.arange(len(res[0])), res[0], label='train loss')
    plt.plot(np.arange(len(res[1])), res[1], label='val loss')
    plt.plot(np.arange(len(res[2])), res[2], label='f1 training score')
    plt.plot(np.arange(len(res[3])), res[3], label='f1 test score')
    plt.title(str(configs))
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'losses/{idx}/{configs}.png')


weight_decay_domain = [1e-2]
lr_domain = [1e-4]

idx = 7

for lr in lr_domain:
    for weight_decay in weight_decay_domain:
        for i in range(5):
            config = {
                'weight_decay': weight_decay,
                'lr': lr,
                'iteartion': i
            }
            train(config)

import logging

import numpy as np
import pandas as pd
import ray
import torch
import torch as t
from matplotlib import pyplot as plt
from ray import tune
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import BCELoss

logging.basicConfig(
    level=logging.INFO,
    filename='log2.log',
    filemode='w',
    format='%(asctime)s - %(message)s'
)

import model
from data import ChallengeDataset
from trainer import Trainer

data = pd.read_csv('data.csv', sep=';')
train, test = train_test_split(data, test_size=0.2)

training_data = ChallengeDataset(train, mode='train')
val_data = ChallengeDataset(test, mode='val')


critic = BCELoss()




def train(config):
    logging.info(f"""
        {config} starts
    \n:""")

    net = model.ResNet()

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    optimizer = t.optim.SGD(net.parameters(), weight_decay=config['weight_decay'], lr=config['lr'])
    train_loader = t.utils.data.DataLoader(training_data, batch_size=int(config["batch_size"]))

    vali_loader = torch.utils.data.DataLoader(val_data, batch_size=int(config["batch_size2"]), shuffle=True)

    trainer = Trainer(model=net, crit=critic, optim=optimizer, train_dl=train_loader, val_test_dl=vali_loader,
                      early_stopping_patience=20)

    # go, go, go... call fit on trainer
    res = trainer.fit(500)

    # plot the results
    plt.clf()
    plt.plot(np.arange(len(res[0])), res[0], label='train loss')
    plt.plot(np.arange(len(res[1])), res[1], label='val loss')
    plt.plot(np.arange(len(res[2])), res[2], label='f1 training score')
    plt.plot(np.arange(len(res[3])), res[3], label='f1 test score')
    plt.title(str(config))
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'losses/{config}.png')


ray.init()
analysis = tune.run(
    train,
    config={
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16, 32]),
        "batch_size2": tune.choice([2, 4, 6, 8, 10]),
        "weight_decay": tune.loguniform(0.01, 0.1)
    },
    num_samples=10,
)

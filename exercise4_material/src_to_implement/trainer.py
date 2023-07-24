import logging
import random
import string

import numpy as np
import torch
import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm


model_directory = '/proj/ciptmp/ky10pida/models'


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, '{}/checkpoint_{}.ckp'.format(model_directory,epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('{}/checkpoint_{}.ckp'.format(model_directory,epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss

        self._optim.zero_grad()

        y_hat = self._model(x).double()

        loss = self._crit(y_hat,y)
        loss.backward()
        self._optim.step()

        y_hat[y_hat < 0.5] = 0
        y_hat[y_hat >= 0.5] = 1

        return loss.item(),y_hat
        
        
    
    def val_test_step(self, x, y):
        
        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions


        y_hat = self._model(x).double()
        loss = self._crit(y_hat,y)

        y_hat[y_hat >= 0.5] = 1
        y_hat[y_hat < 0.5] = 0
        return loss.item(),y_hat
        
    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it

        losses = []
        y_hats = []
        ys = []


        self._model.train()
        with torch.enable_grad():
            training_set = self._train_dl

            for x,y in training_set:
                if self._cuda:
                    x = x.cuda()
                    y = y.cuda()
                    loss,y_hat = self.train_step(x,y)
                    losses.append(loss)

                    y_hats.extend(y_hat.cpu().detach().numpy())
                    ys.extend(y.cpu().detach().numpy())

        f1 = f1_score(np.array(ys).reshape((-1, 2)), np.array(y_hats).reshape((-1, 2)), average='weighted')

        return sum(losses) / len(losses),f1


    
    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore. 
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics

        losses = []
        y_hats = []
        ys = []
        self._model.eval()

        with torch.no_grad():
            validation_set = self._val_test_dl

            for x,y in validation_set:
                if self._cuda:
                    x = x.cuda()
                    y = y.cuda()
                loss,y_hat = self.val_test_step(x,y)

                y_hats.extend(y_hat.cpu().numpy())
                ys.extend(y.cpu().numpy())

                losses.append(loss)

        f1 = f1_score(np.array(ys).reshape((-1,2)),np.array(y_hats).reshape((-1,2)),average='weighted')
        return sum(losses) / len(losses), f1

    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch

        train_losses = []
        val_losses = []

        metrics = []
        metrics_training = []

        count = 0
        patience_count = 0
        min_validation_loss = 100000000
        max_f1 = 0

        delta = 0.02

        prefix = str("".join(random.choice(string.ascii_lowercase) for _ in range(6)))
        logging.info("epoch file prefix {}\n".format(prefix))
        
        while True:
            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation

            if count >= epochs:
                break

            train_loss,f1_training = self.train_epoch()
            train_losses.append(train_loss)
            metrics_training.append(f1_training)

            val_loss,f1 = self.val_test()
            val_losses.append(val_loss)
            metrics.append(f1)

            self.save_checkpoint(f"{prefix}_{count}")
            logging.info(
                f"train loss:{train_loss}, val_loss:{val_loss}, f1 training score:{f1_training},f1_score:{f1},model@{prefix}_{count}")

            if val_loss > min_validation_loss + delta and f1 + delta < max_f1:
                patience_count += 1

            if val_loss < min_validation_loss or f1 > max_f1:
                min_validation_loss = min(val_loss,min_validation_loss)
                max_f1 = max(max_f1,f1)
                patience_count = 0

            if patience_count >= self._early_stopping_patience:
                break

            count += 1

        return train_losses,val_losses,metrics_training,metrics
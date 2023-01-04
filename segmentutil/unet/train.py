import sys
import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from datetime import datetime
from segmentutil.unet.utils.DataSet import TestData
from segmentutil.unet.utils.lovasz_losses import lovasz_hinge
from segmentutil.unet.utils.Loss import Compute
from segmentutil.unet.utils.ScoreMatrix import Stats, Compute_IOU
from segmentutil.unet.models import UNET25D_Atrous
from segmentutil.unet.config import Config
# from utils import CE_and_Dice


def Change_Lr(optimizer, epoch, init_lr, decay_rate):
    lr = init_lr - decay_rate*(epoch//2)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


class Train_Session:
    """
    Create a training session.
    """

    def __init__(self, model, train_val_ds, test_ds=None):
        assert isinstance(train_val_ds, Dataset), 'the input must be a dataset'
        assert isinstance(model, nn.Module), 'the model input needs to be defined'
        self.model = model
        self.train_val_ds = train_val_ds
        if test_ds and isinstance(test_ds, Dataset):
            self.test_ds = test_ds
            self._additional_test = True
        else:
            self._additional_test = False
        self.saving_dir = ''
        self.log_dir = ''

    def Create_Saving_Logs(self, saving_dir, msg, printing=True):

        if not os.path.isdir(saving_dir):
            os.makedirs(saving_dir)

        session_folder_name = datetime.now().strftime('%Y%m%d%H%M%S')
        session_dir = os.path.join(saving_dir, session_folder_name)
        os.mkdir(session_dir)
        self.saving_dir = os.path.join(session_dir, 'trained_network')
        os.mkdir(self.saving_dir)
        self.log_dir = os.path.join(session_dir, 'log.txt')

        with open(self.log_dir, 'w+') as f:
            f.write(msg + '\n')

        if printing:
            print('Training folder created:{}'.format(session_dir))

    def Logging(self, content='stats', printing=True, **kwargs):
        """
        Print and write the training log
        :param content: stats of each epoch or the Hyper-parameters
        :param printing:
        :param kwargs:
        :return:
        """
        if content == 'hyper_parameters':
            with open(self.log_dir, 'a') as f:
                line = ','.join(['{}:{}'.format(keys, values) for keys, values in kwargs.items()])
                f.write(line + '\n')
                if printing:
                    print(line)
        elif content == 'stats':
            with open(self.log_dir, 'a') as f:
                line = ','.join(['{}:{}'.format(keys, values) if type(values) == int else '{}:{:.6f}'.format(keys, values) for keys, values in kwargs.items()])
                f.write(line + '\n')
                if printing:
                    print(line)
        else:
            with open(self.log_dir, 'a') as f:
                f.write(content + '\n')
                if printing:
                    print(content)

    def Freeze(self, unfreeze_layers):
        """
        Unfreeze certain layers for transfer learning
        :param unfreeze_layers: last # layers, for instance 3 for last 3 layers
        :return: model
        """
        unfreeze_idx = len(list(self.model.children())) - unfreeze_layers
        for i, (name, layer) in enumerate(self.model.named_children()):
            if i < unfreeze_idx:
                self.model.__getattr__(name).requires_grad_(False)
        return self.model

    def Train(self, config, loss_type=nn.CrossEntropyLoss, split=5, save_model=True):
        """
        Train the model.
        :param save_model: If need to save the trained model
        :param split: split of train and validation
        :param config: a config file specifying all the training info
        :param loss_type: the loss function
        :return: the model
        """

        if self._additional_test:
            stats_name = ['train_loss', 'val_loss', 'test_loss', 'train_IOU', 'val_IOU', 'test_IOU']
        else:
            stats_name = ['train_loss', 'val_loss', 'train_IOU', 'val_IOU']
        stats = Stats(stats_name)
        best_IOU = 0.0

        self.model.float()
        self.model.cuda()
        epoch = config.epoch
        if loss_type == nn.CrossEntropyLoss:
            wt = torch.from_numpy(config.weight).float()
            cri = loss_type(weight=wt).cuda()
        else:

            cri = loss_type

        lr = config.lr
        weight_decay = config.weight_decay
        opm = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Record the hyper parameters from the config file
        all_hyper_dict = {}
        all_hyper_dict.update({attr: config.__getattribute__(attr) for attr in config.__dir__() if not attr.endswith('__')})
        self.Logging(content='hyper_parameters', **all_hyper_dict)

        # Loading dataset
        total_samples = len(self.train_val_ds)
        split_train, split_val = int(total_samples*split/(split + 1)), total_samples - int(total_samples*split/(split + 1))
        train_ds, val_ds = random_split(self.train_val_ds, [split_train, split_val])
        train_dataLoader = DataLoader(dataset=train_ds, batch_size=config.batch, num_workers=config.num_workers, drop_last=False, shuffle=True)  # Loading training data
        val_dataLoader = DataLoader(dataset=val_ds, batch_size=1, num_workers=config.num_workers, drop_last=False)  # Loading validation data
        if self._additional_test:
            test_dataLoader = DataLoader(dataset=self.test_ds, batch_size=1, num_workers=config.num_workers, drop_last=False)  # Loading testing data
            self.Logging(content='training:{}, validation:{}, testing:{}'.format(split_train, split_val, self.test_ds.__len__()), printing=True)
        else:
            self.Logging(content='training:{}, validation:{}'.format(split_train, split_val), printing=True)
        print('Training starts '.ljust(200, '-'))
        cross_entropy = nn.CrossEntropyLoss(weight=torch.from_numpy(config.weight).float(), reduction='mean').cuda()
        for i in range(epoch):

            train_loss = val_loss = test_loss = 0
            opm = Change_Lr(opm, i, config.lr, 0.0001)
            # Training
            self.model.train()
            train_pred_vec, train_lab_vec = [], []
            for j, (ori_images, ori_labels) in enumerate(train_dataLoader):
                print(ori_images.shape, ori_labels.shape)
                loss, result = Compute(self.model, cross_entropy, ori_images, ori_labels)
                result = torch.argmax(result, dim=1).view(-1)
                train_pred_vec.extend(result.cpu().numpy())
                train_lab_vec.extend(ori_labels.view(-1).numpy())
                train_loss += loss.item()
                opm.zero_grad()
                loss.backward()
                opm.step()

            # Validation
            self.model.eval()
            val_pred_vec = []
            val_lab_vec = []
            with torch.no_grad():
                for k, (val_images, val_labels) in enumerate(val_dataLoader):
                    loss, result = Compute(self.model, cross_entropy, val_images, val_labels)
                    val_loss += loss.item()
                    result = torch.argmax(result, dim=1).view(-1)
                    val_labels = val_labels.view(-1)
                    val_pred_vec.extend(result.cpu().numpy())
                    val_lab_vec.extend(val_labels.cpu().numpy())

            # Additional test
            test_pred_vec = []
            test_lab_vec = []
            if self._additional_test:
                with torch.no_grad():
                    for m, (test_images, test_labels) in enumerate(test_dataLoader):
                        loss, result = Compute(self.model, cross_entropy, test_images, test_labels)
                        test_loss += loss.item()
                        result = torch.argmax(result, dim=1).view(-1)
                        test_labels = test_labels.view(-1)
                        test_pred_vec.extend(result.cpu().numpy())
                        test_lab_vec.extend(test_labels.cpu().numpy())

            # Calculate stats
            epoch_stats = {'epoch': i}
            if self._additional_test:
                stats_list = [train_loss/len(train_dataLoader), val_loss/len(val_dataLoader), test_loss/len(test_dataLoader), Compute_IOU(train_pred_vec, train_lab_vec), Compute_IOU(val_pred_vec, val_lab_vec), Compute_IOU(test_pred_vec, test_lab_vec)]
                this_epoch_stats = dict(zip(stats_name, stats_list))
                stats.Update(this_epoch_stats)

            else:
                stats_list = [train_loss/len(train_dataLoader), val_loss/len(val_dataLoader), Compute_IOU(train_pred_vec, train_lab_vec), Compute_IOU(val_pred_vec, val_lab_vec)]
                this_epoch_stats = dict(zip(stats_name, stats_list))
                stats.Update(this_epoch_stats)
            epoch_stats.update(this_epoch_stats)
            self.Logging(content='stats', printing=True, **epoch_stats)

            # Saving
            if save_model and self._additional_test:
                if epoch_stats['test_IOU'] > best_IOU:
                    trained_name = os.path.join(self.saving_dir, '{}.pt'.format(i))
                    torch.save(self.model.state_dict(), trained_name)
                    best_IOU = epoch_stats['test_IOU']

            torch.cuda.empty_cache()



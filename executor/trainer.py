import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn

from utils.logger import get_logger


LOG = get_logger(__name__)

class Trainer:

    def __init__(self, **kwargs):
        self.device = kwargs['device']
        self.model = kwargs['model']
        self.trainloader, self.testloader = kwargs['dataloaders']
        self.dataloaders_dict = {"train": self.trainloader, "val": self.testloader}
        self.epochs = kwargs['epochs']
        self.optimizer = kwargs['optimizer']
        self.criterion = kwargs['criterion']
        # self.metric = kwargs['metrics']
        self.save_ckpt_interval = kwargs['save_ckpt_interval']
        self.ckpt_dir = kwargs['ckpt_dir']
        # self.writer = SummaryWriter(str(kwargs['summary_dir']))

    def train(self):
        # ネットワークがある程度固定であれば、高速化させる
        torch.backends.cudnn.benchmark = True

        iteration = 1 # イテレーションカウンタ
        epoch_train_loss = 0.0  # epochの損失和
        best_loss = np.inf
        best_epoch = 0

        for epoch in range(self.epochs+1):
            LOG.info(f'\n==================== Epoch: {epoch} ====================')
            LOG.info('\n Train:')

            # 開始時刻を保存
            t_epoch_start = time.time()

            self.model.train()
            with tqdm(self.dataloaders_dict['train'], ncols=100) as pbar:
                for images, targets in pbar:

                    images = images.to(self.device)
                    targets = [ann.to(self.device) for ann in targets]  # リストの各要素のテンソルをGPUへ

                    self.optimizer.zero_grad()

                    outputs = self.model(images)

                    loss_l, loss_c = self.criterion(outputs, targets)
                    loss = loss_l + loss_c

                    loss.backward()  # 勾配の計算

                    # 勾配が大きくなりすぎると計算が不安定になるので、clipで最大でも勾配2.0に留める
                    nn.utils.clip_grad_value_(
                        self.model.parameters(), clip_value=2.0)

                    self.optimizer.step()  # パラメータ更新

                    epoch_train_loss += loss.item()
                    iteration += 1

                    if (iteration % 10 == 0):  # 10iterに1度、lossを表示
                        description = f"iteration：{iteration}, time：{time.time() - t_epoch_start:.4f}, mean loss:{epoch_train_loss / (iteration):.16f}"
                        pbar.set_description(description)

                    pbar.set_description(f'train epoch: {epoch}')
            
            epoch_val_loss = self.eval(epoch)

            # save ckpt
            if epoch != 0 and epoch % self.save_ckpt_interval == 0:
                LOG.info(' Saving Checkpoint...')
                self._save_ckpt(epoch)

            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                self._save_ckpt(epoch, mode='best')

    def eval(self, epoch):
        # ネットワークがある程度固定であれば、高速化させる
        torch.backends.cudnn.benchmark = True

        epoch_val_loss = 0.0  # epochの損失和

        self.model.eval()   # モデルを検証モードに
        print('-------------')
        print('（val）')
        with tqdm(self.dataloaders_dict['train'], ncols=100) as pbar:
            with torch.no_grad():
                for images, targets in self.dataloaders_dict['val']:

                    # GPUが使えるならGPUにデータを送る
                    images = images.to(self.device)
                    targets = [ann.to(self.device)
                            for ann in targets]  # リストの各要素のテンソルをGPUへ

                    self.optimizer.zero_grad()

                    outputs = self.model(images)

                    # 損失の計算
                    loss_l, loss_c = self.criterion(outputs, targets)
                    loss = loss_l + loss_c

                    epoch_val_loss += loss.item()

                    pbar.set_description(f'eval epoch: {epoch}')
                
        return epoch_val_loss

    def _save_ckpt(self, epoch, mode=None, zfill=4):
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model
    
        if mode == 'best':
            ckpt_path = self.ckpt_dir / 'best_acc_ckpt.pth'
        else:
            ckpt_path = self.ckpt_dir / f'epoch{str(epoch).zfill(zfill)}_ckpt.pth'

        torch.save({
            'epoch': epoch,
            'model': model,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, ckpt_path)
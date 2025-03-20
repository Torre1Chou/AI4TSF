from models.Leddam.data_provider.data_factory import data_provider
from models.Leddam.exp.exp_basic import Exp_Basic
from models.Leddam.utils.tools import EarlyStopping, adjust_learning_rate
from models.Leddam.utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import json
import pandas as pd

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

        #生成的预测数据
        self.preds_df = None

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'MSE' or self.args.loss == 'mse':
            criterion = nn.MSELoss()
        elif self.args.loss == 'MAE' or self.args.loss == 'mae':
            criterion = nn.L1Loss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            preds=[]
            trues=[]
            
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                
                batch_x = batch_x.float().to(self.device,non_blocking=True)
                batch_y = batch_y[:, -self.args.pred_len:,:].float()
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:

                    outputs = self.model(batch_x)
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().numpy()
                preds.append(pred)
                trues.append(true)
        # if len(preds) == 0:
        #     # 提示错误或返回一个默认的损失值
        #     print("Warning: No validation predictions were generated. Please check your validation dataset.")
        #     return float('nan')  # 或者 raise 一个明确的异常
        #if len(preds)>0:
        preds=np.concatenate(preds, axis=0)
        trues=np.concatenate(trues, axis=0)
        # else:
        #     preds=preds[0]
        #     trues=trues[0]

        mse,mae= metric(preds, trues)
        vali_loss=mae if criterion == 'MAE' or criterion == 'mae' else mse
        self.model.train()
        torch.cuda.empty_cache()
        return vali_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        #打印test_loader的长度
        #print('test_loader的大小:', len(test_loader))

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad(set_to_none=True)
                batch_x = batch_x.float().to(self.device,non_blocking=True)
                batch_y = batch_y[:, -self.args.pred_len:,:].float().to(self.device,non_blocking=True)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                torch.cuda.empty_cache()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss= self.vali(vali_data, vali_loader, self.args.loss)
            test_loss = self.vali(test_data, test_loader, self.args.loss)

            print("Epoch: {}, Steps: {} | Train Loss: {:.3f}  vali_loss: {:.3f}   test_loss: {:.3f} ".format(epoch + 1, train_steps, train_loss,  vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
        torch.cuda.empty_cache()

    

    def test(self, setting, test=1):
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))

        head = f'./test_dict/{self.args.data_path[:-4]}/'
        tail = f'{self.args.model}'
        dict_path = head + tail

        if not os.path.exists(dict_path):
            os.makedirs(dict_path)

        self.model.eval()
        with torch.no_grad():
            last_inputs = None
            last_preds = None
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y[:, -self.args.pred_len:, :].float()
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)
                outputs = outputs.detach().cpu().numpy()
                batch_x = batch_x.detach().cpu().numpy()  # 将 CUDA 张量移动到 CPU

                last_preds = outputs[-1]
                last_inputs = batch_x[-1]

        if last_preds is not None and last_inputs is not None:
            print('test shape:', last_preds.shape)

            #去掉时间列
            original_columns = test_data.columns[1:]


            #print('test_data的所有列名称:', test_data.columns)

            date_col = test_data.get_dates()

            
            #print('预测长度:', len(outputs))

            # 确保 last_preds 和 last_inputs 的形状与原始数据的列数匹配
            num_features = len(original_columns)
            last_preds = last_preds.reshape(last_preds.shape[0], -1)[:, :num_features]
            last_inputs = last_inputs.reshape(last_inputs.shape[0], -1)[:, :num_features]

            # 调整形状以匹配逆标准化的要求
            last_preds = last_preds.reshape(-1, num_features)
            last_inputs = last_inputs.reshape(-1, num_features)


            # 逆标准化
            last_preds = test_data.inverse_transform(last_preds)
            last_inputs = test_data.inverse_transform(last_inputs)

            # 获取日期列
            #dates = date_col[:last_preds.shape[0]]

            # 获取最后一个已知的日期
            last_date = pd.to_datetime(date_col.iloc[-1])

            # 计算预测步长（pred_len）
            pred_len = last_preds.shape[0]

            # 生成从 last_date 开始的未来 pred_len 天
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=pred_len, freq='D')

            # 转换成数组
            dates = future_dates.strftime('%Y-%m-%d').to_numpy()

            # 将日期列添加回预测和输入数据
            self.preds_df = pd.DataFrame(last_preds, columns=original_columns)
            #inputs_df = pd.DataFrame(last_inputs, columns=original_columns)
            self.preds_df.insert(0, 'date', dates)
            #inputs_df.insert(0, 'date', dates)

            # # Save predictions and inputs as CSV files
            # preds_df = pd.DataFrame(last_preds.reshape(last_preds.shape[0], -1), columns=original_columns)
            # inputs_df = pd.DataFrame(last_inputs.reshape(last_inputs.shape[0], -1), columns=original_columns)

            #preds_df.to_csv(os.path.join(dict_path, 'predictions.csv'), index=False)
            #inputs_df.to_csv(os.path.join(dict_path, 'inputs.csv'), index=False)

        torch.cuda.empty_cache()
        return
    
    def get_predictions_as_dataframe(self):
        return self.preds_df

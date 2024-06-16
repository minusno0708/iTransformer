from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

weather_columns = ['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
       'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
       'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
       'wd (deg)', 'rain (mm)', 'raining (s)', 'SWDR (W/m�)',
       'PAR (�mol/m�/s)', 'max. PAR (�mol/m�/s)', 'Tlog (degC)', 'OT']

class Loss_Analyzer():
    def __init__(self):
        self.train_loss = np.array([])
        self.vali_loss = np.array([])
        self.test_loss = np.array([])

        self.train_details = np.array([])
        self.test_details = np.array([])

    def loss_append(self, flag, loss):
        if flag == 'train':
            self.train_loss = np.append(self.train_loss, loss)
        elif flag == 'vali':
            self.vali_loss = np.append(self.vali_loss, loss)
        elif flag == 'test':
            self.test_loss = np.append(self.test_loss, loss)

    def details_append(self, flag, details):
        if flag == 'train':
            if self.train_details.size == 0:
                self.train_details = details.reshape(details.shape[0], 1)
            else:
                self.train_details = np.concatenate((self.train_details, details.reshape(details.shape[0], 1)), axis=1)
        elif flag == 'test':
            if self.test_details.size == 0:
                self.test_details = details.reshape(details.shape[0], 1)
            else:
                self.test_details = np.concatenate((self.test_details, details.reshape(details.shape[0], 1)), axis=1)

    def print_loss(self):
        print("Train Loss: ", self.train_loss)
        print("Vali Loss: ", self.vali_loss)
        print("Test Loss: ", self.test_loss)
    
    def print_details(self, flag):
        if flag == 'train':
            print("Train Details: ", self.train_details)
            print("shape: ", self.train_details.shape)
        elif flag == 'test':
            print("Test Details: ", self.test_details)
            print("shape: ", self.test_details.shape)

    def write_loss(self, path, title=""):
        plt.plot(self.train_loss, label='Train Loss')
        plt.plot(self.vali_loss, label='Vali Loss')
        plt.plot(self.test_loss, label='Test Loss')

        if title != "":
            plt.title(title) 

        plt.legend()
        plt.savefig(path)

    def write_details(self, path, title=""):
        fig, ax = plt.subplots(7, 3)
        for i, train, test in zip(range(self.train_details.shape[0]), self.train_details, self.test_details):
            ax[int(i/3), i%3].plot(train, label="Train")
            ax[int(i/3), i%3].plot(test, label="Test")

            ax[int(i/3), i%3].set_title(f"[{str(i+1)}]{weather_columns[i]}", fontsize=6, x=0.5, y=0.6)

            ax[int(i/3), i%3].tick_params(axis='x', labelsize=6)
            ax[int(i/3), i%3].tick_params(axis='y', labelsize=6)
        plt.legend(bbox_to_anchor=(0.8, 0), loc='upper left', borderaxespad=0, fontsize=10)
    
        if title != "":
            plt.title(title)
            
        plt.savefig(path)
        plt.close()

    def get_detail_loss(self, pred, true):
        diff = (pred - true) ** 2
        diff = np.transpose(diff, (1, 2, 0))
        diff = diff.reshape(diff.shape[0], diff.shape[1] * diff.shape[2])
        diff = diff.mean(axis=1)
        return diff

    def write_pred_true(pred, true):
        tmp_dir = "results/output_compare_train/"

        for j in range(len(pred)):

            fig, ax = plt.subplots(7, 3)

            for i in range(len(pred_t)):
                ax[int(i/3), i%3].plot(pred[j, i], label="pred")
                ax[int(i/3), i%3].plot(true[j, i], label="true")

                ax[int(i/3), i%3].set_title(f"[{str(i+1)}]{weather_columns[i]}", fontsize=6, x=0.5, y=0.6)

                ax[int(i/3), i%3].tick_params(axis='x', labelsize=6)
                ax[int(i/3), i%3].tick_params(axis='y', labelsize=6)
            plt.legend(bbox_to_anchor=(0.8, 0), loc='upper left', borderaxespad=0, fontsize=10)

            filename = f"{j}.jpg"
            print(tmp_dir + filename)
            plt.savefig(tmp_dir + filename)
            plt.close()


class Results_Inverser():
    def __init__(self, pred, true, scaler):
        self.pred = pred
        self.true = true
        self.scaler = scaler

        self.inverse()

    def inverse(self):
        self.pred = self.scaler.inverse_transform(self.pred)
        self.true = self.scaler.inverse_transform(self.true)

    def write(self, path, title=""):
        fig, ax = plt.subplots(7, 3)

        for i, pred_i, true_i in zip(range(self.pred.shape[1]), self.pred.T, self.true.T):
            ax[int(i/3), i%3].plot(pred_i, label="pred")
            ax[int(i/3), i%3].plot(true_i, label="true")

            ax[int(i/3), i%3].set_title(f"[{str(i+1)}]{weather_columns[i]}", fontsize=6, x=0.5, y=0.6)

            ax[int(i/3), i%3].tick_params(axis='x', labelsize=6)
            ax[int(i/3), i%3].tick_params(axis='y', labelsize=6)
        plt.legend(bbox_to_anchor=(0.8, 0), loc='upper left', borderaxespad=0, fontsize=10)

        if title != "":
            plt.title(title)
        
        plt.savefig(path)
        plt.close()

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

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
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, loss_analyzer=None, scaler=None):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            tmp = 0

            pred_t = np.array([])
            true_t = np.array([])

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                if loss_analyzer != None:
                    if pred_t.size == 0:
                        pred_t = pred.permute(0, 2, 1).numpy()
                        true_t = true.permute(0, 2, 1).numpy()
                    else:
                        pred_t = np.concatenate((pred_t, pred.permute(0, 2, 1).numpy()), axis=0)
                        true_t = np.concatenate((true_t, true.permute(0, 2, 1).numpy()), axis=0)
                    tmp += 1

                    if tmp == 32:
                        diff = loss_analyzer.get_detail_loss(pred_t, true_t)
                        loss_analyzer.details_append('test', diff)

                loss = criterion(pred, true)

                total_loss.append(loss)

            if scaler != None:
                test_output = Results_Inverser(pred[0].numpy(), true[0].numpy(), scaler)
                test_output.write(f"results/pred_true/vali.jpg")

        total_loss = np.average(total_loss)
        
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        loss_analyzer = Loss_Analyzer()

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        print("step:")
        print(train_steps)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        #for epoch in range(self.args.train_epochs):
        for epoch in range(10):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        pred = outputs.detach().cpu()
                        true = batch_y.detach().cpu()

                        pred_t = pred.permute(0, 2, 1).numpy()[0]
                        true_t = true.permute(0, 2, 1).numpy()[0]

                        tmp_dir = "results/output_compare_train/"

                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

            train_output = Results_Inverser(pred[0].numpy(), true[0].numpy(), train_data.scaler)
            train_output.write(f"results/pred_true/train.jpg")

            pred_t = pred.permute(0, 2, 1).numpy()
            true_t = true.permute(0, 2, 1).numpy()

            diff_t = loss_analyzer.get_detail_loss(pred_t, true_t)
            loss_analyzer.details_append('train', diff_t)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)  
            test_loss = self.vali(test_data, test_loader, criterion, loss_analyzer, test_data.scaler)

            # Lossを記録
            loss_analyzer.loss_append('train', train_loss)
            loss_analyzer.loss_append('vali', vali_loss)
            loss_analyzer.loss_append('test', test_loss)

            loss_analyzer.print_loss()

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            #early_stopping(vali_loss, self.model, path)
            #if early_stopping.early_stop:
            #    print("Early stopping")
            #    break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        try:
            self.model.load_state_dict(torch.load(best_model_path))
        except:
            print("no best model")

        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}'.format(
            epoch + 1, train_steps, train_loss, vali_loss, test_loss) + '\n')
        f.close()

        loss_analyzer.write_loss('results/' + setting + "_remove_0.1" + '.jpg')
        loss_analyzer.write_details('results/' + setting + "_remove_0.1" + '_details.jpg')

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
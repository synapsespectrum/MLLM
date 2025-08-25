import os
import torch
import torch.nn as nn
from torch import optim
import time
import warnings
import numpy as np
import datetime
from benchmarks.unimodal_models.data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, MetricsTracker
from utils.metrics import metric
from torch.utils.tensorboard import SummaryWriter
from benchmarks.unimodal_models.models import Autoformer, Transformer, Informer, Crossformer, iTransformer, PatchTST

warnings.filterwarnings('ignore')


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()

        self.model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'PatchTST': PatchTST,
            'Crossformer': Crossformer,
            'iTransformer': iTransformer
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

        self.__setup_logging()
        self.__setup_tracking()

    def __setup_logging(self):
        setting = '{}_{}_{}_ft{}sl{}ll{}pl{}dm{}nh{}el{}dl{}df{}fc{}eb{}dt{}{}{}'.format(
            self.args.model,
            self.args.model_id,
            self.args.data,
            self.args.features,
            self.args.seq_len,
            self.args.label_len,
            self.args.pred_len,
            self.args.d_model,
            self.args.n_heads,
            self.args.e_layers,
            self.args.d_layers,
            self.args.d_ff,
            self.args.factor,
            self.args.embed,
            self.args.distil,
            self.args.des, self.args.run_id)

        import platform
        if platform.system() == 'Windows' and len(setting) > 100:
            from datetime import datetime
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            setting = f"{self.args.model}_pl{self.args.pred_len}_{now}"
            print(f"⚠️  Setting name shortened for Windows compatibility")

        self.args.setting = setting
        # ./logs/experiment_name/dataset_name
        self.args.output_path = os.path.join("benchmarks/logs", self.args.data)
        if not os.path.exists(self.args.output_path):
            os.makedirs(self.args.output_path)
        # ./logs/experiment_name/dataset_name/settings
        self.args.log_path = os.path.join(self.args.output_path, self.args.setting)
        if not os.path.exists(self.args.log_path):
            os.makedirs(self.args.log_path)

    def __setup_tracking(self):
        # Setup TensorBoard Writer ./logs/{exp_name}/{dataset_name}/{settings}/tensorboard
        tensorboard_path = os.path.join(self.args.log_path, 'tensorboard')
        os.makedirs(tensorboard_path, exist_ok=True)
        self.writer = SummaryWriter(tensorboard_path)
        print(f"📊 TensorBoard logs: {tensorboard_path}")
        # Setup metrics tracker
        log_iteration_freq = getattr(self.args, 'log_iteration_freq', 10)  # Default 100
        self.metrics_tracker = MetricsTracker(
            writer=self.writer,
            log_iteration_freq=log_iteration_freq,
            tracking_mlflow=False
        )

    def close_tracking(self):
        """Đóng tất cả tracking connections"""
        if hasattr(self, 'writer'):
            self.writer.close()
            print("📊 TensorBoard writer closed")

        if hasattr(self, 'metrics_tracker'):
            best_metrics = self.metrics_tracker.get_best_metrics_summary()
            print("🏆 Best metrics summary displayed")

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass


class Exp_Unimodal_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Unimodal_Forecast, self).__init__(args)
        self.args = args
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
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

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # forward
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        time_now = time.time()

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
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # forward
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
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

                # Log to TensorBoard
                self.writer.add_scalar('train_loss', loss.item(), epoch * train_steps + i)
                self.metrics_tracker.log_iteration_metrics({
                    'train_loss': loss.item(),
                    'learning_rate': model_optim.param_groups[0]['lr']
                })

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            # Log to TensorBoard
            self.metrics_tracker.log_epoch_metrics({
                'train_loss': train_loss,
                'validation_loss': vali_loss,
                'test_loss': test_loss
            })

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, os.path.join(self.args.output_path, self.args.setting))
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = self.args.output_path + '/' + self.args.setting + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, test=0):
        if test:
            print('Loading model from checkpoint: ', self.args.checkpoints)
            self.model.load_state_dict(torch.load(self.args.checkpoints))

        preds = []
        trues = []

        self.model.eval()

        print('Start testing phase...')
        test_data, test_loader = self._get_data(flag='test')

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # forward
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

                # Visualize some examples
                if i % 1 == 0:
                    try:
                        input = batch_x.detach().cpu().numpy()
                        if test_data.scale and self.args.inverse:
                            shape = input.shape
                            input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                        gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(self.args.log_path, str(i) + '.pdf'))
                    except Exception as e:
                        print(f"Visualization skipped due to error: {e}")

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        # Calculate metrics
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))

        # Save results
        result_path = os.path.join(self.args.output_path, 'result.txt')
        with open(result_path, 'a') as f:
            f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ")
            f.write(self.args.setting + "  \n")
            f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
            f.write('\n\n')

        # Save metrics and predictions
        np.save(os.path.join(self.args.log_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(self.args.log_path, 'pred.npy'), preds)
        np.save(os.path.join(self.args.log_path, 'true.npy'), trues)

        return mse

    def close_tracking(self):
        """Close tracking connections"""
        if hasattr(self, 'writer'):
            self.writer.close()
            print("📊 TensorBoard writer closed")

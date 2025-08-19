import os
import torch
from ts import PatchTST, iTransformer


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            # 'TimesNet': TimesNet,
            # 'Autoformer': Autoformer,
            # 'Transformer': Transformer,
            # 'Nonstationary_Transformer': Nonstationary_Transformer,
            # 'DLinear': DLinear,
            # 'FEDformer': FEDformer,
            # 'Informer': Informer,
            # 'LightTS': LightTS,
            # 'Reformer': Reformer,
            # 'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            # 'Pyraformer': Pyraformer,
            # 'MICN': MICN,
            # 'Crossformer': Crossformer,
            # 'FiLM': FiLM,
            'iTransformer': iTransformer,
            # 'Koopa': Koopa,
            # 'TiDE': TiDE,
            # 'FreTS': FreTS,
            # 'TimeMixer': TimeMixer,
            # 'TSMixer': TSMixer,
            # 'SegRNN': SegRNN
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.args = args
        self.__setup_logging()

    def __setup_logging(self):
        # checking the output path
        output_path = './logs' + self.args.experiment_name
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            self.args.task_name,
            self.args.model_id,
            self.args.model,
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
            self.args.expand,
            self.args.d_conv,
            self.args.factor,
            self.args.embed,
            self.args.distil,
            self.args.des, self.args.run_id)
        self.args.setting = setting
        # ./logs/experiment_name/dataset_name
        self.args.output_path = os.path.join("./logs", self.args.experiment_name, self.args.data)
        if not os.path.exists(self.args.output_path):
            os.makedirs(self.args.output_path)
        # ./logs/experiment_name/dataset_name/settings
        if not os.path.exists(os.path.join(self.args.output_path, self.args.setting)):
            os.makedirs(os.path.join(self.args.output_path, self.args.setting))


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

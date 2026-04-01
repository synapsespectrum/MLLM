import os

# Import tracking libraries
import mlflow
import mlflow.pytorch
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.tools import MetricsTracker


class Exp_Basic:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.__setup_logging()
        self.__setup_tracking()

    def __setup_logging(self):
        # checking the output path
        output_path = os.path.join("logs/", self.args.experiment_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        setting = f"{self.args.model}_{self.args.llm_model}_{self.args.data}_ft{self.args.features}sl{self.args.seq_len}ll{self.args.label_len}pl{self.args.pred_len}pw{self.args.prompt_weight}dm{self.args.d_model}nh{self.args.n_heads}el{self.args.e_layers}dl{self.args.d_layers}df{self.args.d_ff}expand{self.args.expand}dc{self.args.d_conv}fc{self.args.factor}eb{self.args.embed}dt{self.args.distil}{self.args.des}{self.args.run_id}"

        import platform

        if platform.system() == "Windows" and len(setting) > 100:
            from datetime import datetime

            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            setting = f"{self.args.model}_pl{self.args.pred_len}_{now}"
            print("⚠️  Setting name shortened for Windows compatibility")

        self.args.setting = setting
        # ./logs/experiment_name/dataset_name
        self.args.output_path = os.path.join(
            "logs", self.args.experiment_name, self.args.data
        )
        if not os.path.exists(self.args.output_path):
            os.makedirs(self.args.output_path)
        # ./logs/experiment_name/dataset_name/settings
        self.args.log_path = os.path.join(self.args.output_path, self.args.setting)
        if not os.path.exists(self.args.log_path):
            os.makedirs(self.args.log_path)

    def __setup_tracking(self):
        # Setup TensorBoard Writer ./logs/{exp_name}/{dataset_name}/{settings}/tensorboard
        tensorboard_path = os.path.join(self.args.log_path, "tensorboard")
        os.makedirs(tensorboard_path, exist_ok=True)
        self.writer = SummaryWriter(tensorboard_path)
        print(f"📊 TensorBoard logs: {tensorboard_path}")

        # Setup MLflow: ./mlflow/{exp_name}/{dataset_name}/{settings}/
        mlflow_base_path = os.path.join(
            "./mlflow_track", self.args.data, self.args.experiment_name
        )
        os.makedirs(mlflow_base_path, exist_ok=True)
        # mlflow.set_tracking_uri(f"file://{os.path.abspath(mlflow_base_path)}")

        mlflow_db_path = os.path.join(mlflow_base_path, "mlflow.db")
        mlflow_db_path = os.path.abspath(mlflow_db_path)

        mlflow_tracking_path = f"sqlite:///{mlflow_db_path.replace(os.sep, '/')}"
        mlflow.set_tracking_uri(mlflow_tracking_path)
        print(f"📈 MLflow tracking path: {mlflow_tracking_path}")

        # Create experiment name từ args
        experiment_name = self.args.setting
        try:
            mlflow.set_experiment(experiment_name)
            print(f"🧪 MLflow experiment: {experiment_name}")
        except Exception as e:
            print(f"⚠️  MLflow experiment error: {e}")

        # Create run name với thông tin chi tiết để dễ phân biệt
        run_name = f"{self.args.model}_run{self.args.run_id}"

        # Start MLflow run
        self.mlflow_run = mlflow.start_run(run_name=run_name)
        print(f"🏃 MLflow run: {run_name}")

        # Setup MetricsTracker
        log_iteration_freq = getattr(self.args, "log_iteration_freq", 10)  # Default 100
        self.metrics_tracker = MetricsTracker(
            writer=self.writer,
            log_iteration_freq=log_iteration_freq,
            tracking_mlflow=self.args.tracking_mlflow,
        )

        # Log experiment parameters
        self._log_experiment_params()
        self._set_mlflow_tags()

    def _log_experiment_params(self):
        """Log tất cả parameters vào MLflow"""
        # Model parameters
        mlflow.log_param("model", self.args.model)
        mlflow.log_param("model_id", self.args.model_id)
        mlflow.log_param("task_name", self.args.task_name)

        # Data parameters
        mlflow.log_param("data", self.args.data)
        mlflow.log_param("features", self.args.features)
        mlflow.log_param("seq_len", self.args.seq_len)
        mlflow.log_param("label_len", self.args.label_len)
        mlflow.log_param("pred_len", self.args.pred_len)

        # Model architecture
        mlflow.log_param("d_model", self.args.d_model)
        mlflow.log_param("n_heads", self.args.n_heads)
        mlflow.log_param("e_layers", self.args.e_layers)
        mlflow.log_param("d_layers", self.args.d_layers)
        mlflow.log_param("d_ff", self.args.d_ff)
        mlflow.log_param("dropout", self.args.dropout)

        # Training parameters
        mlflow.log_param("learning_rate", self.args.learning_rate)
        mlflow.log_param("batch_size", self.args.batch_size)
        mlflow.log_param("train_epochs", self.args.train_epochs)
        mlflow.log_param("patience", self.args.patience)
        mlflow.log_param("seed", self.args.seed)

        # LLM parameters (nếu có)
        if hasattr(self.args, "llm_model"):
            mlflow.log_param("llm_model", self.args.llm_model)
            mlflow.log_param("llm_dim", self.args.llm_dim)
            mlflow.log_param("prompt_weight", self.args.prompt_weight)

        # Device info
        mlflow.log_param("device", str(self.device))
        mlflow.log_param("use_gpu", self.args.use_gpu)

        print("✅ Experiment parameters logged to MLflow")

    def _set_mlflow_tags(self):
        """Set tags để dễ dàng filter và search experiments"""
        # Basic tags
        mlflow.set_tag("model_type", self.args.model)
        mlflow.set_tag("dataset", self.args.data)
        mlflow.set_tag("task", self.args.task_name)
        mlflow.set_tag("experiment_group", self.args.experiment_name)

        # Training configuration tags
        mlflow.set_tag("seq_len", str(self.args.seq_len))
        mlflow.set_tag("pred_len", str(self.args.pred_len))
        mlflow.set_tag("features", self.args.features)

        # Model architecture tags
        mlflow.set_tag("d_model", str(self.args.d_model))
        mlflow.set_tag("layers", f"e{self.args.e_layers}_d{self.args.d_layers}")

        # Special tags
        if hasattr(self.args, "llm_model") and self.args.llm_model:
            mlflow.set_tag("uses_llm", "true")
            mlflow.set_tag("llm_type", self.args.llm_model)
        else:
            mlflow.set_tag("uses_llm", "false")

        # Version tag (có thể dùng để track code version)
        import datetime

        mlflow.set_tag("created_date", datetime.datetime.now().strftime("%Y-%m-%d"))

        print("🏷️  MLflow tags set for easy filtering")

    def close_tracking(self):
        """Đóng tất cả tracking connections"""
        if hasattr(self, "writer"):
            self.writer.close()
            print("📊 TensorBoard writer closed")

        if hasattr(self, "mlflow_run"):
            mlflow.end_run()
            print("📈 MLflow run ended")

        if hasattr(self, "metrics_tracker"):
            best_metrics = self.metrics_tracker.get_best_metrics_summary()
            print("🏆 Best metrics summary displayed")

    def log_final_metrics(self, test_mse=None, **kwargs):
        """Log final metrics khi kết thúc experiment"""
        if test_mse is not None:
            mlflow.log_metric("final_test_mse", test_mse)

        # Log any additional metrics
        for key, value in kwargs.items():
            mlflow.log_metric(f"final_{key}", value)

        # Log best metrics từ MetricsTracker
        if hasattr(self, "metrics_tracker"):
            best_metrics = self.metrics_tracker.best_metrics
            for metric_name, info in best_metrics.items():
                mlflow.log_metric(f"best_{metric_name}", info["value"])
                mlflow.log_metric(f"best_{metric_name}_epoch", info["epoch"])

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device(f"cuda:{self.args.gpu}")
            print(f"Use GPU: cuda:{self.args.gpu}")
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import datetime
import glob
import os
import pathlib
import random
import threading

import dill
import hydra
import numpy as np
import torch
import tqdm
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

try:
    import huggingface_hub

    if not hasattr(huggingface_hub, "cached_download") and hasattr(huggingface_hub, "hf_hub_download"):
        huggingface_hub.cached_download = huggingface_hub.hf_hub_download
except Exception:
    pass

from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.model.goal.pose_utils import pose9d_to_matrix, transform_point_cloud
from diffusion_policy_3d.policy.goal_pose_diffuser import GoalPoseDiffuser


OmegaConf.register_new_resolver("eval", eval, replace=True)


class _NoOpWandbRun:
    def log(self, *args, **kwargs):
        return None


def _prefix_metrics(metrics, prefix):
    return {f"{prefix}/{k}": v for k, v in metrics.items()}


def _mean_metric_dict(metric_dicts):
    if len(metric_dicts) == 0:
        return {}
    keys = metric_dicts[0].keys()
    result = {}
    for key in keys:
        values = [x[key] for x in metric_dicts if key in x]
        if len(values) > 0:
            result[key] = float(np.mean(values))
    return result


class TrainGoalPoseWorkspace:
    include_keys = ["global_step", "epoch", "_output_dir"]
    exclude_keys = tuple()

    def _infer_task_save_name(self, cfg) -> str:
        output_name = getattr(cfg.task, "output_name", None)
        if output_name is not None and str(output_name).strip():
            return str(output_name)

        data_dirs = OmegaConf.to_container(cfg.task.dataset.data_dirs, resolve=True)
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        names = []
        for path in data_dirs:
            if path is None:
                continue
            name = os.path.basename(os.path.normpath(str(path)))
            if name:
                names.append(name)

        if len(names) == 0:
            return str(cfg.task.task_name)
        if len(names) == 1:
            return names[0]
        return "multitask__" + "__".join(names)

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        base_dir = output_dir if output_dir is not None else HydraConfig.get().runtime.output_dir
        self.task_save_name = self._infer_task_save_name(cfg)
        self.task_base_dir = os.path.join(base_dir, f"{self.task_save_name}_seed{cfg.training.seed}")

        if cfg.training.resume:
            subdirs = sorted(
                path for path in glob.glob(os.path.join(self.task_base_dir, "20*")) if os.path.isdir(path)
            )
            if len(subdirs) > 0:
                self._output_dir = subdirs[-1]
                print(f"\n[*] GoalPose resume: using latest workspace -> {self._output_dir}")
            else:
                self._output_dir = self.task_base_dir
                os.makedirs(self._output_dir, exist_ok=True)
                print(f"\n[*] GoalPose resume: no timestamp workspace found, using -> {self._output_dir}")
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self._output_dir = os.path.join(self.task_base_dir, timestamp)
            os.makedirs(self._output_dir, exist_ok=True)
            print(f"\n[*] GoalPose train: created workspace -> {self._output_dir}")
        self._update_latest_link()
        self._saving_thread = None

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: GoalPoseDiffuser = hydra.utils.instantiate(cfg.policy)
        self.ema_model: GoalPoseDiffuser = copy.deepcopy(self.model) if cfg.training.use_ema else None
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
        self.global_step = 0
        self.epoch = 0

    def _update_latest_link(self):
        latest_path = os.path.join(self.task_base_dir, "latest")
        try:
            if os.path.islink(latest_path) or os.path.exists(latest_path):
                if os.path.realpath(latest_path) == os.path.realpath(self._output_dir):
                    return
                if os.path.islink(latest_path):
                    os.unlink(latest_path)
                else:
                    return
            os.symlink(os.path.basename(self._output_dir), latest_path)
        except Exception as exc:
            print(f"[GoalPose] failed to update latest link {latest_path}: {exc}")

    @property
    def output_dir(self):
        return self._output_dir

    def _save_resolved_config(self, cfg):
        output_dir = pathlib.Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        resolved_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        OmegaConf.save(config=resolved_cfg, f=output_dir.joinpath("config_resolved.yaml"))

    def _configure_run_names(self, cfg):
        cfg.exp_name = f"goal_pose_{self.task_save_name}"
        cfg.logging.group = cfg.exp_name
        cfg.logging.name = f"{self.task_save_name}_seed{cfg.training.seed}"

    def _prepare_wandb_dirs(self):
        output_dir = pathlib.Path(self.output_dir)
        wandb_cache_dir = output_dir.joinpath("wandb_cache")
        wandb_config_dir = output_dir.joinpath("wandb_config")
        wandb_data_dir = output_dir.joinpath("wandb_data")
        for path in (wandb_cache_dir, wandb_config_dir, wandb_data_dir):
            path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("WANDB_CACHE_DIR", str(wandb_cache_dir))
        os.environ.setdefault("WANDB_CONFIG_DIR", str(wandb_config_dir))
        os.environ.setdefault("WANDB_DATA_DIR", str(wandb_data_dir))

    def _device(self, cfg):
        device = torch.device(cfg.training.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print(f"[GoalPose] CUDA requested ({device}) but unavailable; falling back to CPU.")
            device = torch.device("cpu")
        return device

    def _print_epoch_summary(self, metrics):
        print(f"\n[GoalPose Epoch {self.epoch}]")
        ordered_keys = [
            "train/loss",
            "train/loss_pose9d",
            "train/goal_pos_err_cm",
            "train/goal_rot_err_deg",
            "val/loss",
            "val/loss_pose9d",
            "val/goal_pos_err_cm",
            "val/goal_rot_err_deg",
            "val_sample/goal_pos_err_cm",
            "val_sample/goal_rot_err_deg",
            "train_sample/goal_pos_err_cm",
            "train_sample/goal_rot_err_deg",
            "train_sample/goal_cloud_mse",
        ]
        for key in ordered_keys:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        self._configure_run_names(cfg)
        self.cfg = cfg
        if cfg.training.debug:
            cfg.training.num_epochs = min(int(cfg.training.num_epochs), 2)
            cfg.training.max_train_steps = 2 if cfg.training.max_train_steps is None else cfg.training.max_train_steps
            cfg.training.max_val_steps = 1 if cfg.training.max_val_steps is None else cfg.training.max_val_steps
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            cfg.logging.mode = "disabled"

        self._save_resolved_config(cfg)
        print(
            "[GoalPose Config] "
            f"task_save_name={self.task_save_name}, "
            f"batch_size={cfg.dataloader.batch_size}, "
            f"shuffle={cfg.dataloader.shuffle}, "
            f"max_train_steps={cfg.training.max_train_steps}, "
            f"num_epochs={cfg.training.num_epochs}, "
            f"logging.mode={cfg.logging.mode}"
        )

        dataset: BaseDataset = hydra.utils.instantiate(cfg.task.dataset)
        if len(dataset) == 0:
            raise RuntimeError("GoalPoseSE3Dataset is empty; check configs/model/task/goal_pose_multitask.yaml data_dirs.")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        if cfg.training.resume:
            ckpt_path = self.get_checkpoint_path()
            if ckpt_path.is_file():
                print(f"Resuming from checkpoint {ckpt_path}")
                self.load_checkpoint(path=ckpt_path)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=max(1, (len(train_dataloader) * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every),
            last_epoch=self.global_step - 1,
        )
        ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model) if cfg.training.use_ema else None

        self._prepare_wandb_dirs()
        try:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging,
            )
            wandb.config.update({"output_dir": self.output_dir})
        except Exception as exc:
            print(f"[GoalPose] wandb init failed; continuing with no-op logging. Error: {exc}")
            wandb_run = _NoOpWandbRun()
        topk_manager = TopKCheckpointManager(save_dir=os.path.join(self.output_dir, "checkpoints"), **cfg.checkpoint.topk)

        device = self._device(cfg)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        train_sampling_batch = None
        for _ in range(cfg.training.num_epochs):
            self.model.train()
            epoch_log = {
                "epoch": self.epoch,
                "train/epoch": self.epoch,
            }
            train_step_logs = []
            with tqdm.tqdm(train_dataloader, desc=f"GoalPose Train epoch {self.epoch}", leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch
                    raw_loss, loss_dict = self.model.compute_loss(batch)
                    if not torch.isfinite(raw_loss):
                        raise RuntimeError(f"Non-finite GoalPose loss at step {self.global_step}: {raw_loss.item()}")
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    if ema is not None:
                        ema.step(self.model)

                    raw_loss_cpu = raw_loss.item()
                    train_log = {
                        "train/loss": raw_loss_cpu,
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/epoch": self.epoch,
                        "train/global_step": self.global_step,
                    }
                    train_log.update(_prefix_metrics(loss_dict, "train"))
                    train_step_logs.append(train_log)
                    tepoch.set_postfix(
                        loss=raw_loss_cpu,
                        pos_err_cm=loss_dict.get("goal_pos_err_cm", float("nan")),
                        rot_err_deg=loss_dict.get("goal_rot_err_deg", float("nan")),
                        refresh=False,
                    )
                    wandb_run.log(train_log, step=self.global_step)
                    self.global_step += 1

                    if cfg.training.max_train_steps is not None and batch_idx >= cfg.training.max_train_steps - 1:
                        break

            epoch_train_log = _mean_metric_dict(train_step_logs)
            epoch_log.update(epoch_train_log)
            epoch_log["train/global_step"] = self.global_step
            epoch_log["global_step"] = self.global_step
            policy = self.ema_model if self.ema_model is not None else self.model
            policy.eval()

            if self.epoch % cfg.training.val_every == 0:
                epoch_log.update(self.validate(policy, val_dataloader, device, cfg))

            if self.epoch % cfg.training.sample_every == 0 and train_sampling_batch is not None:
                epoch_log.update(self.print_train_samples(policy, train_sampling_batch, device))

            if self.epoch % cfg.training.checkpoint_every == 0 and cfg.checkpoint.save_ckpt:
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint()
                metric_dict = {k.replace("/", "_"): v for k, v in epoch_log.items()}
                if cfg.checkpoint.topk.monitor_key in metric_dict:
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

            wandb_run.log(epoch_log, step=self.global_step)
            self._print_epoch_summary(epoch_log)
            self.epoch += 1

    @torch.no_grad()
    def validate(self, policy, val_dataloader, device, cfg):
        denoise_logs = []
        sample_pos_errs, sample_rot_errs, sample_cloud_losses = [], [], []
        if len(val_dataloader.dataset) == 0:
            return {
                "val/loss": float("nan"),
                "val/loss_pose9d": float("nan"),
                "val/loss_trans": float("nan"),
                "val/loss_rot_rad": float("nan"),
                "val/loss_rot_deg": float("nan"),
                "val/loss_cloud": float("nan"),
                "val/goal_pos_err_cm": float("nan"),
                "val/goal_rot_err_deg": float("nan"),
                "val_sample/goal_pos_err_cm": float("nan"),
                "val_sample/goal_rot_err_deg": float("nan"),
                "val_sample/goal_cloud_mse": float("nan"),
            }
        with tqdm.tqdm(val_dataloader, desc=f"GoalPose Val epoch {self.epoch}", leave=False) as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                loss, loss_dict = policy.compute_loss(batch)
                denoise_log = {"val/loss": loss.item()}
                denoise_log.update(_prefix_metrics(loss_dict, "val"))
                denoise_logs.append(denoise_log)
                result = policy.sample_goal(batch["obs"])
                pred9 = result["goal_pose9d"]
                gt9 = batch["goal_pose9d"]
                T_pred = pose9d_to_matrix(pred9)
                T_gt = pose9d_to_matrix(gt9)
                pc_man = GoalPoseDiffuser._squeeze_pc(batch["obs"]["pc_manipulated"])
                P_pred = transform_point_cloud(pc_man, T_pred)
                P_gt = transform_point_cloud(pc_man, T_gt)
                pos_err = torch.linalg.norm(pred9[:, :3] - gt9[:, :3], dim=-1) * 100.0
                rot_err = GoalPoseDiffuser._rotation_error_per_sample(T_pred[:, :3, :3], T_gt[:, :3, :3]) * 180.0 / torch.pi
                cloud_err = torch.mean((P_pred - P_gt) ** 2, dim=(1, 2))
                sample_pos_errs.extend(pos_err.detach().cpu().tolist())
                sample_rot_errs.extend(rot_err.detach().cpu().tolist())
                sample_cloud_losses.extend(cloud_err.detach().cpu().tolist())
                if cfg.training.max_val_steps is not None and batch_idx >= cfg.training.max_val_steps - 1:
                    break
        result = _mean_metric_dict(denoise_logs)
        result.update({
            "val_sample/goal_pos_err_cm": float(np.mean(sample_pos_errs)),
            "val_sample/goal_rot_err_deg": float(np.mean(sample_rot_errs)),
            "val_sample/goal_cloud_mse": float(np.mean(sample_cloud_losses)),
        })
        return result

    @torch.no_grad()
    def print_train_samples(self, policy, batch, device):
        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        result = policy.sample_goal(batch["obs"])
        pred9 = result["goal_pose9d"]
        pred7 = result["goal_pose7d"]
        gt9 = batch["goal_pose9d"]
        gt7 = batch["goal_pose7d"]
        T_pred = pose9d_to_matrix(pred9)
        T_gt = pose9d_to_matrix(gt9)
        R_pred = T_pred[:, :3, :3]
        R_gt = T_gt[:, :3, :3]
        pc_man = GoalPoseDiffuser._squeeze_pc(batch["obs"]["pc_manipulated"])
        P_pred = transform_point_cloud(pc_man, T_pred)
        P_gt = transform_point_cloud(pc_man, T_gt)
        pos_err = torch.linalg.norm(pred9[:, :3] - gt9[:, :3], dim=-1) * 100.0
        rot_err = GoalPoseDiffuser._rotation_error_per_sample(R_pred, R_gt) * 180.0 / torch.pi
        cloud_err = torch.mean((P_pred - P_gt) ** 2, dim=(1, 2))
        print(f"\n[GoalPose sample epoch {self.epoch}]")
        for i in range(min(3, pred9.shape[0])):
            print(f"  sample {i}:")
            print(f"    pred xyz {pred9[i, :3].detach().cpu().numpy()} | gt xyz {gt9[i, :3].detach().cpu().numpy()} | err {pos_err[i].item():.2f} cm")
            print(f"    pred quat {pred7[i, 3:7].detach().cpu().numpy()} | gt quat {gt7[i, 3:7].detach().cpu().numpy()} | rot {rot_err[i].item():.2f} deg")
        return {
            "train_sample/goal_pos_err_cm": float(pos_err.mean().item()),
            "train_sample/goal_rot_err_deg": float(rot_err.mean().item()),
            "train_sample/goal_cloud_mse": float(cloud_err.mean().item()),
        }

    def save_checkpoint(self, path=None, tag="latest", use_thread=False):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")
        else:
            path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cfg": self.cfg,
            "state_dicts": {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            "pickles": {
                "global_step": dill.dumps(self.global_step),
                "epoch": dill.dumps(self.epoch),
                "_output_dir": dill.dumps(self._output_dir),
            },
            "normalizer": self.model.normalizer.state_dict(),
            "output_dir": self.output_dir,
        }
        if self.ema_model is not None:
            payload["state_dicts"]["ema_model"] = self.ema_model.state_dict()
        if use_thread:
            self._saving_thread = threading.Thread(target=lambda: torch.save(payload, path.open("wb"), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open("wb"), pickle_module=dill)
        return str(path.absolute())

    def get_checkpoint_path(self, tag="latest"):
        return pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")

    def load_payload(self, payload, **kwargs):
        self.model.load_state_dict(payload["state_dicts"]["model"], **kwargs)
        self.optimizer.load_state_dict(payload["state_dicts"]["optimizer"])
        if self.ema_model is not None and "ema_model" in payload["state_dicts"]:
            self.ema_model.load_state_dict(payload["state_dicts"]["ema_model"], **kwargs)
        if "normalizer" in payload:
            self.model.normalizer.load_state_dict(payload["normalizer"])
            if self.ema_model is not None:
                self.ema_model.normalizer.load_state_dict(payload["normalizer"])
        for key, value in payload.get("pickles", {}).items():
            self.__dict__[key] = dill.loads(value)

    def load_checkpoint(self, path=None, tag="latest", **kwargs):
        path = self.get_checkpoint_path(tag=tag) if path is None else pathlib.Path(path)
        payload = torch.load(path.open("rb"), pickle_module=dill, map_location="cpu")
        self.load_payload(payload, **kwargs)
        return payload


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).resolve().parents[2].joinpath("configs/model")),
    config_name="train_goal_pose_diffusion",
)
def main(cfg):
    workspace = TrainGoalPoseWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()

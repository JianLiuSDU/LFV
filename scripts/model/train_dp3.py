# if __name__ == "__main__":
#     import sys
#     import os
#     import pathlib

#     ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[2])
#     sys.path.append(ROOT_DIR)
#     os.chdir(ROOT_DIR)

# import os
# import hydra
# import torch
# import dill
# from omegaconf import OmegaConf
# import pathlib
# from torch.utils.data import DataLoader
# import copy
# import random
# import wandb
# import tqdm
# import numpy as np
# from termcolor import cprint
# import shutil
# import time
# import threading
# from hydra.core.hydra_config import HydraConfig
# from diffusion_policy_3d.policy.dp3 import DP3
# from diffusion_policy_3d.dataset.base_dataset import BaseDataset
# from diffusion_policy_3d.env_runner.base_runner import BaseRunner
# from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager
# from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
# from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
# from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler


# OmegaConf.register_new_resolver("eval", eval, replace=True)

# class TrainDP3Workspace:
#     include_keys = ['global_step', 'epoch']
#     exclude_keys = tuple()

#     # def __init__(self, cfg: OmegaConf, output_dir=None):
#     #     self.cfg = cfg
#     #     self._output_dir = output_dir
#     #     self._saving_thread = None
        
#     #     # set seed
#     #     seed = cfg.training.seed
#     #     torch.manual_seed(seed)
#     #     np.random.seed(seed)
#     #     random.seed(seed)

#     #     # configure model
#     #     self.model: DP3 = hydra.utils.instantiate(cfg.policy)

#     #     self.ema_model: DP3 = None
#     #     if cfg.training.use_ema:
#     #         try:
#     #             self.ema_model = copy.deepcopy(self.model)
#     #         except: 
#     #             self.ema_model = hydra.utils.instantiate(cfg.policy)

#     #     # configure training state
#     #     self.optimizer = hydra.utils.instantiate(
#     #         cfg.optimizer, params=self.model.parameters())

#     #     self.global_step = 0
#     #     self.epoch = 0
#     def _infer_task_save_name(self, cfg) -> str:
#         """
#         从 cfg.task.dataset.data_dirs 自动推断当前训练任务的保存目录名。
#         单任务：
#             ["/media/ljian/lj/data_3d/drawer_open"] -> "drawer_open"
#         多任务：
#             ["/media/.../drawer_open", "/media/.../pickNplace"] -> "multitask__drawer_open__pickNplace"
#         """
#         data_dirs = OmegaConf.to_container(cfg.task.dataset.data_dirs, resolve=True)

#         if isinstance(data_dirs, str):
#             data_dirs = [data_dirs]

#         if not data_dirs:
#             # 兜底，避免空配置时报错
#             return str(cfg.task.task_name)

#         names = []
#         for p in data_dirs:
#             if p is None:
#                 continue
#             name = os.path.basename(os.path.normpath(str(p)))
#             if len(name) > 0:
#                 names.append(name)

#         if len(names) == 0:
#             return str(cfg.task.task_name)

#         if len(names) == 1:
#             return names[0]

#         return "multitask__" + "__".join(names)
#     # def __init__(self, cfg: OmegaConf, output_dir=None):
#     #         self.cfg = cfg
            
#     #         # ==========================================
#     #         # 💡 修改开始：按时间戳自动隔离训练文件夹
#     #         # ==========================================
#     #         import datetime
#     #         import glob
            
#     #         # 获取 Hydra 默认的根目录 (例如: data/outputs/multitask_8d_seed0)
#     #         base_dir = output_dir if output_dir is not None else HydraConfig.get().runtime.output_dir
            
#     #         if cfg.training.resume:
#     #             # 如果是断点续训，自动寻找最新的时间戳文件夹
#     #             subdirs = sorted([d for d in glob.glob(os.path.join(base_dir, "20*")) if os.path.isdir(d)])
#     #             if len(subdirs) > 0:
#     #                 self._output_dir = subdirs[-1]
#     #                 print(f"\n[*] Resume 模式: 自动挂载到最新训练记录 -> {self._output_dir}")
#     #             else:
#     #                 self._output_dir = base_dir
#     #         else:
#     #             # 如果是重新训练，新建一个时间戳文件夹，永远不用再手动 rm -rf
#     #             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     #             self._output_dir = os.path.join(base_dir, timestamp)
#     #             os.makedirs(self._output_dir, exist_ok=True)
#     #             print(f"\n[*] New Train 模式: 已创建全新时间戳工作区 -> {self._output_dir}")
#     #         # ==========================================
#     #         # 💡 修改结束
#     #         # ==========================================
            
#     #         self._saving_thread = None
            
#     #         # set seed
#     #         seed = cfg.training.seed
#     #         torch.manual_seed(seed)
#     #         np.random.seed(seed)
#     #         random.seed(seed)

#     #         # configure model
#     #         self.model: DP3 = hydra.utils.instantiate(cfg.policy)

#     #         self.ema_model: DP3 = None
#     #         if cfg.training.use_ema:
#     #             try:
#     #                 self.ema_model = copy.deepcopy(self.model)
#     #             except: 
#     #                 self.ema_model = hydra.utils.instantiate(cfg.policy)

#     #         # configure training state
#     #         self.optimizer = hydra.utils.instantiate(
#     #             cfg.optimizer, params=self.model.parameters())

#     #         self.global_step = 0
#     #         self.epoch = 0
#     def __init__(self, cfg: OmegaConf, output_dir=None):
#             self.cfg = cfg

#             import datetime
#             import glob

#             # Hydra 传进来的总根目录，比如 data/outputs
#             base_dir = output_dir if output_dir is not None else HydraConfig.get().runtime.output_dir

#             # 自动从 data_dirs 推断当前任务名
#             task_save_name = self._infer_task_save_name(cfg)

#             # 你想保留 seed 区分的话，这里一起加上
#             task_base_dir = os.path.join(base_dir, f"{task_save_name}_seed{cfg.training.seed}")

#             if cfg.training.resume:
#                 subdirs = sorted(
#                     [d for d in glob.glob(os.path.join(task_base_dir, "20*")) if os.path.isdir(d)]
#                 )
#                 if len(subdirs) > 0:
#                     self._output_dir = subdirs[-1]
#                     print(f"\n[*] Resume 模式: 自动挂载到最新训练记录 -> {self._output_dir}")
#                 else:
#                     self._output_dir = task_base_dir
#                     os.makedirs(self._output_dir, exist_ok=True)
#             else:
#                 timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#                 self._output_dir = os.path.join(task_base_dir, timestamp)
#                 os.makedirs(self._output_dir, exist_ok=True)
#                 print(f"\n[*] New Train 模式: 已创建全新任务工作区 -> {self._output_dir}")

#             self._saving_thread = None

#             # set seed
#             seed = cfg.training.seed
#             torch.manual_seed(seed)
#             np.random.seed(seed)
#             random.seed(seed)

#             # configure model
#             self.model: DP3 = hydra.utils.instantiate(cfg.policy)

#             self.ema_model: DP3 = None
#             if cfg.training.use_ema:
#                 try:
#                     self.ema_model = copy.deepcopy(self.model)
#                 except:
#                     self.ema_model = hydra.utils.instantiate(cfg.policy)

#             # configure training state
#             self.optimizer = hydra.utils.instantiate(
#                 cfg.optimizer, params=self.model.parameters())

#             self.global_step = 0
#             self.epoch = 0
#             self._last_eval_traj_metrics = {}



#     def run(self):
#         cfg = copy.deepcopy(self.cfg)
        
#         if cfg.training.debug:
#             cfg.training.num_epochs = 100
#             cfg.training.max_train_steps = 10
#             cfg.training.max_val_steps = 3
#             cfg.training.rollout_every = 20
#             cfg.training.checkpoint_every = 1
#             cfg.training.val_every = 1
#             cfg.training.sample_every = 1
#             RUN_ROLLOUT = True
#             RUN_CKPT = False
#             verbose = True
#         else:
#             RUN_ROLLOUT = True
#             RUN_CKPT = True
#             verbose = False
        
#         RUN_VALIDATION = True 
        
#         # resume training
#         if cfg.training.resume:
#             lastest_ckpt_path = self.get_checkpoint_path()
#             if lastest_ckpt_path.is_file():
#                 print(f"Resuming from checkpoint {lastest_ckpt_path}")
#                 self.load_checkpoint(path=lastest_ckpt_path)

#         # configure dataset
#         dataset: BaseDataset
#         dataset = hydra.utils.instantiate(cfg.task.dataset)

#         assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
#         train_dataloader = DataLoader(dataset, **cfg.dataloader)
#         normalizer = dataset.get_normalizer()

#         val_dataset = dataset.get_validation_dataset()
#         val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

#         self.model.set_normalizer(normalizer)
#         if cfg.training.use_ema:
#             self.ema_model.set_normalizer(normalizer)

#         # configure lr scheduler
#         lr_scheduler = get_scheduler(
#             cfg.training.lr_scheduler,
#             optimizer=self.optimizer,
#             num_warmup_steps=cfg.training.lr_warmup_steps,
#             num_training_steps=(
#                 len(train_dataloader) * cfg.training.num_epochs) \
#                     // cfg.training.gradient_accumulate_every,
#             last_epoch=self.global_step-1
#         )

#         ema: EMAModel = None
#         if cfg.training.use_ema:
#             ema = hydra.utils.instantiate(
#                 cfg.ema,
#                 model=self.ema_model)

#         env_runner = None
        
#         cfg.logging.name = str(cfg.logging.name)
#         cprint("-----------------------------", "yellow")
#         cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
#         cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
#         cprint("-----------------------------", "yellow")

#         # configure logging
#         wandb_run = wandb.init(
#             dir=str(self.output_dir),
#             config=OmegaConf.to_container(cfg, resolve=True),
#             **cfg.logging
#         )
#         wandb.config.update({"output_dir": self.output_dir})

#         topk_manager = TopKCheckpointManager(
#             save_dir=os.path.join(self.output_dir, 'checkpoints'),
#             **cfg.checkpoint.topk
#         )

#         device = torch.device(cfg.training.device)
#         if device.type == "cuda" and not torch.cuda.is_available():
#             cprint(f"[TrainDP3] CUDA device {cfg.training.device} unavailable; falling back to CPU.", "yellow")
#             device = torch.device("cpu")
#         self.model.to(device)
#         if self.ema_model is not None:
#             self.ema_model.to(device)
#         optimizer_to(self.optimizer, device)

#         train_sampling_batch = None

#         for local_epoch_idx in range(cfg.training.num_epochs):
#             step_log = dict()
#             train_losses = list()
#             with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
#                     leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
#                 for batch_idx, batch in enumerate(tepoch):
#                     t1 = time.time()
#                     batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
#                     if train_sampling_batch is None:
#                         train_sampling_batch = batch
                
#                     t1_1 = time.time()
#                     raw_loss, loss_dict = self.model.compute_loss(batch)
#                     loss = raw_loss / cfg.training.gradient_accumulate_every
#                     loss.backward()
                    
#                     t1_2 = time.time()

#                     if self.global_step % cfg.training.gradient_accumulate_every == 0:
#                         self.optimizer.step()
#                         self.optimizer.zero_grad()
#                         lr_scheduler.step()
#                     t1_3 = time.time()
                    
#                     if cfg.training.use_ema:
#                         ema.step(self.model)
#                     t1_4 = time.time()
                    
#                     raw_loss_cpu = raw_loss.item()
#                     tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
#                     train_losses.append(raw_loss_cpu)
                    
#                     step_log = {
#                         'train_loss': raw_loss_cpu,
#                         'global_step': self.global_step,
#                         'epoch': self.epoch,
#                         'lr': lr_scheduler.get_last_lr()[0]
#                     }
                    
#                     t1_5 = time.time()
#                     step_log.update(loss_dict)
                    
#                     is_last_batch = (batch_idx == (len(train_dataloader)-1))
#                     if not is_last_batch:
#                         wandb_run.log({k: v for k, v in step_log.items() if k not in ['test_mean_score', 'global_step', 'epoch']}, step=self.global_step)
#                         self.global_step += 1

#                     if (cfg.training.max_train_steps is not None) and batch_idx >= (cfg.training.max_train_steps-1):
#                         break

#             train_loss = np.mean(train_losses)
#             step_log['train_loss'] = train_loss

#             policy = self.model
#             if cfg.training.use_ema:
#                 policy = self.ema_model
#             policy.eval()

#             if (self.epoch % cfg.training.rollout_every) == 0 and RUN_ROLLOUT and env_runner is not None:
#                 runner_log = env_runner.run(policy)
#                 step_log.update(runner_log)

#             # ==========================================
#             # 💡 修复验证集逻辑：不仅测去噪，还做端到端推演
#             # ==========================================
#             if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
#                 val_losses = self.eval_traj_denoise(val_dataloader=val_dataloader, device=device)
#                 if len(val_losses) > 0:
#                     val_loss = torch.mean(torch.tensor(val_losses)).item()
#                     step_log['val_loss'] = val_loss
                
#                 val_action_losses = self.eval_traj(val_dataloader=val_dataloader, device=device)
#                 if len(val_action_losses) > 0:
#                     val_action_loss = torch.mean(torch.tensor(val_action_losses)).item()
#                     step_log['val_action_loss'] = val_action_loss
#                     step_log.update(self._last_eval_traj_metrics)

#             # ==========================================
#             # 🔍 训练集端到端性能评估与拆解
#             # ==========================================
#             if (self.epoch % cfg.training.sample_every) == 0:
#                 with torch.no_grad():
#                     batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
#                     obs_dict = batch['obs']
#                     gt_action = batch['action']
                    
#                     result = policy.predict_action(obs_dict)
#                     pred_action = result['action_pred']
                    
#                     # 💡 强制符号对齐后的旋转 MSE (避免 Q 和 -Q 导致的虚假爆炸)
#                     p_quat = pred_action[..., 3:7]
#                     g_quat = gt_action[..., 3:7]
#                     sign = torch.sign((p_quat * g_quat).sum(dim=-1, keepdim=True))
#                     sign = torch.where(sign == 0, torch.ones_like(sign), sign) # 防止符号为0
#                     p_quat_aligned = p_quat * sign
                    
#                     mse_pos = torch.nn.functional.mse_loss(pred_action[..., :3], gt_action[..., :3])
#                     mse_rot = torch.nn.functional.mse_loss(p_quat_aligned, g_quat)
#                     # mse_gripper = torch.nn.functional.mse_loss(pred_action[..., 7:], gt_action[..., 7:])
                    
#                     # # 💡 计算夹爪的物理准确率 (Accuracy, 0.5 为阈值)
#                     # gripper_acc = ((pred_action[..., 7:] > 0.5) == (gt_action[..., 7:] > 0.5)).float().mean()
                    
#                     # step_log['train_action_mse_pos'] = mse_pos.item()
#                     # step_log['train_action_mse_rot'] = mse_rot.item()
#                     # step_log['train_action_mse_gripper'] = mse_gripper.item()
#                     # step_log['train_gripper_accuracy'] = gripper_acc.item()
#                     step_log['train_action_mse_pos'] = mse_pos.item()
#                     step_log['train_action_mse_rot'] = mse_rot.item()

#                     try:
#                         metrics = self._compute_traj_metrics(pred_action, gt_action)
#                         step_log.update({f"train_sample_{k}": v for k, v in metrics.items()})
#                         self._print_traj_metrics(
#                             title=f"[Epoch {self.epoch} | Train sample full64 middle metrics]",
#                             metrics=metrics,
#                             color="yellow"
#                         )
#                     except Exception as e:
#                         print(f"Print error: {e}")

#                     del batch, obs_dict, gt_action, result, pred_action

#             if env_runner is None:
#                 step_log['test_mean_score'] = - train_loss
                
#             if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
#                 if cfg.checkpoint.save_last_ckpt:
#                     if self.epoch > 0 and (self.epoch % 200) == 0 and cfg.checkpoint.save_ckpt:
#                         self.save_checkpoint(tag=f'epoch={self.epoch:04d}')
#                     self.save_checkpoint()
#                 if cfg.checkpoint.save_last_snapshot:
#                     self.save_snapshot()

#                 metric_dict = {k.replace('/', '_'): v for k, v in step_log.items()}
#                 topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
#                 if topk_ckpt_path is not None:
#                     self.save_checkpoint(path=topk_ckpt_path)
                    
#             policy.train()

#             wandb_log = {k: v for k, v in step_log.items() if k not in ['test_mean_score', 'global_step', 'epoch']}
#             wandb_run.log(wandb_log, step=self.global_step)
            
#             self.global_step += 1
#             self.epoch += 1
#             del step_log

#     def _rotation_error_deg(self, pred_quat, gt_quat):
#         pred_quat = torch.nn.functional.normalize(pred_quat, dim=-1)
#         gt_quat = torch.nn.functional.normalize(gt_quat, dim=-1)
#         dot = torch.sum(pred_quat * gt_quat, dim=-1).abs().clamp(0.0, 1.0)
#         return torch.rad2deg(2.0 * torch.acos(dot))

#     def _compute_traj_metrics(self, pred_action, gt_action):
#         pred_action = pred_action.detach()
#         gt_action = gt_action.detach().to(pred_action.device)

#         pos_err_cm = torch.linalg.norm(pred_action[..., :3] - gt_action[..., :3], dim=-1) * 100.0
#         rot_err_deg = self._rotation_error_deg(pred_action[..., 3:7], gt_action[..., 3:7])

#         T = pred_action.shape[1]
#         if T > 2:
#             middle = slice(1, T - 1)
#         else:
#             middle = slice(0, T)

#         metrics = {
#             "middle_traj_pos_err_cm_mean": pos_err_cm[:, middle].mean().item(),
#             "middle_traj_rot_err_deg_mean": rot_err_deg[:, middle].mean().item(),
#             "full_traj_pos_err_cm_mean": pos_err_cm.mean().item(),
#             "full_traj_rot_err_deg_mean": rot_err_deg.mean().item(),
#             "boundary_start_pos_err_cm": pos_err_cm[:, 0].mean().item(),
#             "boundary_start_rot_err_deg": rot_err_deg[:, 0].mean().item(),
#             "boundary_end_pos_err_cm": pos_err_cm[:, -1].mean().item(),
#             "boundary_end_rot_err_deg": rot_err_deg[:, -1].mean().item(),
#         }

#         for step in (16, 32, 48):
#             if step < T:
#                 metrics[f"step{step}_pos_err_cm"] = pos_err_cm[:, step].mean().item()
#                 metrics[f"step{step}_rot_err_deg"] = rot_err_deg[:, step].mean().item()
#         return metrics

#     def _print_traj_metrics(self, title, metrics, color="cyan"):
#         cprint(f"\n{title}", color)
#         print("  ▶ Main learning metric excludes hard-clamped boundary frames 0 and 63.")
#         print(f"  ▶ middle[1:62] pos: {metrics['middle_traj_pos_err_cm_mean']:.2f} cm | rot: {metrics['middle_traj_rot_err_deg_mean']:.2f} deg")
#         for step in (16, 32, 48):
#             pos_key = f"step{step}_pos_err_cm"
#             rot_key = f"step{step}_rot_err_deg"
#             if pos_key in metrics:
#                 print(f"  ▶ step {step:02d} pos: {metrics[pos_key]:.2f} cm | rot: {metrics[rot_key]:.2f} deg")
#         print(f"  ▶ boundary check start pos: {metrics['boundary_start_pos_err_cm']:.4f} cm | rot: {metrics['boundary_start_rot_err_deg']:.4f} deg")
#         print(f"  ▶ boundary check end   pos: {metrics['boundary_end_pos_err_cm']:.4f} cm | rot: {metrics['boundary_end_rot_err_deg']:.4f} deg")
#         print("  ▶ Step 63 is hard-clamped and is not used as the primary learning metric.")
#         print("-" * 50)
    
#     def eval_traj_denoise(self, val_dataloader=None, device=None, load_checkpoint=False):
#         cfg = copy.deepcopy(self.cfg)    

#         if load_checkpoint:
#             lastest_ckpt_path = self.get_checkpoint_path(tag="latest")
#             if lastest_ckpt_path.is_file():
#                 self.load_checkpoint(path=lastest_ckpt_path)

#         if device is None:
#             device = torch.device(cfg.training.device)

#         if val_dataloader is None:
#             dataset: BaseDataset = hydra.utils.instantiate(cfg.task.dataset)
#             val_dataset = dataset.get_validation_dataset()
#             val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
#             self.model.set_normalizer(dataset.get_normalizer())

#         with torch.no_grad():
#             val_losses = list()
#             with tqdm.tqdm(val_dataloader, desc=f"Validation Denoise {self.epoch}", leave=False) as tepoch:
#                 for batch_idx, batch in enumerate(tepoch):
#                     batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
#                     loss, loss_dict = self.model.compute_loss(batch)
#                     val_losses.append(loss)
#                     if (cfg.training.max_val_steps is not None) and batch_idx >= (cfg.training.max_val_steps-1):
#                         break
#             return val_losses
    
#     def eval_traj(self, val_dataloader=None, device=None, load_checkpoint=False):
#         cfg = copy.deepcopy(self.cfg)    
        
#         if load_checkpoint:
#             lastest_ckpt_path = self.get_checkpoint_path(tag="latest")
#             if lastest_ckpt_path.is_file():
#                 self.load_checkpoint(path=lastest_ckpt_path)

#         if device is None:
#             device = torch.device(cfg.training.device)

#         if val_dataloader is None:
#             dataset: BaseDataset = hydra.utils.instantiate(cfg.task.dataset)
#             val_dataset = dataset.get_validation_dataset()
#             val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
#             self.model.set_normalizer(dataset.get_normalizer())

#         with torch.no_grad():
#             val_losses = list()
#             metric_accumulator = []
            
#             with tqdm.tqdm(val_dataloader, desc=f"Validation Action {self.epoch}", leave=False) as tepoch:
#                 for batch_idx, batch in enumerate(tepoch):
#                     batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
#                     result = self.model.predict_action(batch['obs'], target=batch['action'])
                    
#                     val_losses.append(result['loss'])
#                     pred_action = result['action_pred']
#                     gt_action = batch['action']
#                     metric_accumulator.append(self._compute_traj_metrics(pred_action, gt_action))
                    
#                     if batch_idx == 0:
#                         try:
#                             self._print_traj_metrics(
#                                 title=f"[Epoch {self.epoch} | Val sample full64 middle metrics]",
#                                 metrics=metric_accumulator[-1],
#                                 color="cyan"
#                             )
#                         except Exception as e:
#                             print(f"Val Print error: {e}")

#                     if (cfg.training.max_val_steps is not None) and batch_idx >= (cfg.training.max_val_steps-1):
#                         break
#             if len(metric_accumulator) > 0:
#                 keys = metric_accumulator[0].keys()
#                 self._last_eval_traj_metrics = {
#                     f"val_sample_{k}": float(np.mean([m[k] for m in metric_accumulator]))
#                     for k in keys
#                 }
#             else:
#                 self._last_eval_traj_metrics = {}
#             return val_losses

#     def eval(self):
#         # 原有代码...
#         pass

#     @property
#     def output_dir(self):
#         output_dir = self._output_dir
#         if output_dir is None:
#             output_dir = HydraConfig.get().runtime.output_dir
#         return output_dir
    

#     def save_checkpoint(self, path=None, tag='latest', 
#             exclude_keys=None, include_keys=None, use_thread=False):
#         if path is None:
#             path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
#         else:
#             path = pathlib.Path(path)
#         if exclude_keys is None:
#             exclude_keys = tuple(self.exclude_keys)
#         if include_keys is None:
#             include_keys = tuple(self.include_keys) + ('_output_dir',)

#         path.parent.mkdir(parents=False, exist_ok=True)
#         payload = {
#             'cfg': self.cfg,
#             'state_dicts': dict(),
#             'pickles': dict()
#         } 

#         for key, value in self.__dict__.items():
#             if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
#                 if key not in exclude_keys:
#                     if use_thread:
#                         payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
#                     else:
#                         payload['state_dicts'][key] = value.state_dict()
#             elif key in include_keys:
#                 payload['pickles'][key] = dill.dumps(value)
#         if use_thread:
#             self._saving_thread = threading.Thread(
#                 target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
#             self._saving_thread.start()
#         else:
#             torch.save(payload, path.open('wb'), pickle_module=dill)
        
#         del payload
#         torch.cuda.empty_cache()
#         return str(path.absolute())
    
#     def get_checkpoint_path(self, tag='latest'):
#         if tag=='latest':
#             return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
#         elif tag=='best': 
#             checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
#             all_checkpoints = os.listdir(checkpoint_dir)
#             best_ckpt = None
#             best_score = -1e10
#             for ckpt in all_checkpoints:
#                 if 'latest' in ckpt:
#                     continue
#                 score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
#                 if score > best_score:
#                     best_ckpt = ckpt
#                     best_score = score
#             return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
#         else:
#             return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')

#     def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
#         if exclude_keys is None:
#             exclude_keys = tuple()
#         if include_keys is None:
#             include_keys = payload['pickles'].keys()

#         for key, value in payload['state_dicts'].items():
#             if key not in exclude_keys:
#                 self.__dict__[key].load_state_dict(value, **kwargs)
#         for key in include_keys:
#             if key in payload['pickles']:
#                 self.__dict__[key] = dill.loads(payload['pickles'][key])
    
#     def load_checkpoint(self, path=None, tag='latest',
#             exclude_keys=None, include_keys=None, **kwargs):
#         if path is None:
#             path = self.get_checkpoint_path(tag=tag)
#         else:
#             path = pathlib.Path(path)
#         payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
#         self.load_payload(payload, 
#             exclude_keys=exclude_keys, 
#             include_keys=include_keys)
#         return payload
    
#     @classmethod
#     def create_from_checkpoint(cls, path, exclude_keys=None, include_keys=None, **kwargs):
#         payload = torch.load(open(path, 'rb'), pickle_module=dill)
#         instance = cls(payload['cfg'])
#         instance.load_payload(
#             payload=payload, 
#             exclude_keys=exclude_keys,
#             include_keys=include_keys,
#             **kwargs)
#         return instance

#     def save_snapshot(self, tag='latest'):
#         path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
#         path.parent.mkdir(parents=False, exist_ok=True)
#         torch.save(self, path.open('wb'), pickle_module=dill)
#         return str(path.absolute())
    
#     @classmethod
#     def create_from_snapshot(cls, path):
#         return torch.load(open(path, 'rb'), pickle_module=dill)

# @hydra.main(
#     version_base=None,
#     config_path=str(pathlib.Path(__file__).resolve().parents[2].joinpath('configs/model'))
# )
# def main(cfg):
#     workspace = TrainDP3Workspace(cfg)
#     workspace.run()

# if __name__ == "__main__":
#     main()
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
from termcolor import cprint
from torch.utils.data import DataLoader

from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.policy.dp3 import DP3


OmegaConf.register_new_resolver("eval", eval, replace=True)


def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: _copy_to_cpu(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_copy_to_cpu(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_copy_to_cpu(v) for v in x)
    return x


class TrainDP3Workspace:
    include_keys = ["global_step", "epoch"]
    exclude_keys = tuple()

    def _infer_task_save_name(self, cfg) -> str:
        output_name = getattr(cfg.task, "output_name", None)
        if output_name is not None and str(output_name).strip():
            return str(output_name)

        data_dirs = OmegaConf.to_container(cfg.task.dataset.data_dirs, resolve=True)

        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        if not data_dirs:
            return str(cfg.task.task_name)

        names = []
        for p in data_dirs:
            if p is None:
                continue
            name = os.path.basename(os.path.normpath(str(p)))
            if len(name) > 0:
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
        self.run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        task_base_dir = os.path.join(
            base_dir,
            f"{self.task_save_name}_seed{cfg.training.seed}"
        )

        if cfg.training.resume:
            subdirs = sorted(
                d for d in glob.glob(os.path.join(task_base_dir, "20*"))
                if os.path.isdir(d)
            )
            if len(subdirs) > 0:
                self._output_dir = subdirs[-1]
                cprint(f"\n[*] Resume 模式: 自动挂载到最新训练记录 -> {self._output_dir}", "yellow")
            else:
                self._output_dir = os.path.join(task_base_dir, self.run_timestamp)
                os.makedirs(self._output_dir, exist_ok=True)
                cprint(f"\n[*] Resume 模式: 未找到历史记录，新建 -> {self._output_dir}", "yellow")
        else:
            self._output_dir = os.path.join(task_base_dir, self.run_timestamp)
            os.makedirs(self._output_dir, exist_ok=True)
            cprint(f"\n[*] New Train 模式: 已创建全新任务工作区 -> {self._output_dir}", "green")

        self.task_base_dir = task_base_dir
        self._update_latest_link()
        self._saving_thread = None

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: DP3 = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DP3 = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except Exception:
                self.ema_model = hydra.utils.instantiate(cfg.policy)

        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer,
            params=self.model.parameters()
        )

        self.global_step = 0
        self.epoch = 0

        self._last_eval_traj_metrics = {}
        self._best_val_metrics = {
            "epoch": -1,
            "middle_pos": float("inf"),
            "middle_rot": float("inf"),
        }

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
            cprint(f"[TrainDP3] failed to update latest link {latest_path}: {exc}", "yellow")

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def _save_resolved_config(self, cfg):
        output_dir = pathlib.Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        resolved_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        OmegaConf.save(config=resolved_cfg, f=output_dir.joinpath("config_resolved.yaml"))

    def _device(self, cfg):
        device = torch.device(cfg.training.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            cprint(f"[TrainDP3] CUDA device {cfg.training.device} unavailable; falling back to CPU.", "yellow")
            device = torch.device("cpu")
        return device

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        self._save_resolved_config(cfg)

        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 100
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            run_rollout = True
            run_ckpt = True
        else:
            run_rollout = True
            run_ckpt = True

        run_validation = True

        if cfg.training.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {latest_ckpt_path}")
                self.load_checkpoint(path=latest_ckpt_path)

        dataset: BaseDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseDataset), f"dataset must be BaseDataset, got {type(dataset)}"

        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema and self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=max(
                1,
                (len(train_dataloader) * cfg.training.num_epochs)
                // cfg.training.gradient_accumulate_every
            ),
            last_epoch=self.global_step - 1
        )

        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        cfg.logging.name = str(cfg.logging.name)

        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint(f"[Output] task_save_name: {self.task_save_name}", "yellow")
        cprint(f"[Output] timestamp: {self.run_timestamp}", "yellow")
        cprint(f"[Output] dir: {self.output_dir}", "yellow")
        cprint("-----------------------------", "yellow")

        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update({"output_dir": self.output_dir})

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk
        )

        device = self._device(cfg)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        env_runner = None
        train_sampling_batch = None

        checkpoint_every = int(getattr(cfg.training, "checkpoint_every", 100))
        if checkpoint_every <= 0:
            checkpoint_every = 100

        cprint(f"[Checkpoint] periodic checkpoint will be saved every {checkpoint_every} epochs.", "cyan")

        for _ in range(cfg.training.num_epochs):
            step_log = {}
            train_losses = []

            self.model.train()

            with tqdm.tqdm(
                train_dataloader,
                desc=f"Training epoch {self.epoch}",
                leave=False,
                mininterval=cfg.training.tqdm_interval_sec
            ) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    raw_loss, loss_dict = self.model.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()

                    if cfg.training.use_ema and ema is not None:
                        ema.step(self.model)

                    raw_loss_cpu = raw_loss.item()
                    train_losses.append(raw_loss_cpu)

                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)

                    step_log = {
                        "train_loss": raw_loss_cpu,
                        "global_step": self.global_step,
                        "epoch": self.epoch,
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    step_log.update(loss_dict)

                    is_last_batch = batch_idx == (len(train_dataloader) - 1)
                    if not is_last_batch:
                        wandb_run.log(
                            {
                                k: v for k, v in step_log.items()
                                if k not in ["test_mean_score", "global_step", "epoch"]
                            },
                            step=self.global_step
                        )
                        self.global_step += 1

                    if (
                        cfg.training.max_train_steps is not None
                        and batch_idx >= cfg.training.max_train_steps - 1
                    ):
                        break

            train_loss = float(np.mean(train_losses)) if len(train_losses) > 0 else float("nan")
            step_log["train_loss"] = train_loss

            policy = self.ema_model if (cfg.training.use_ema and self.ema_model is not None) else self.model
            policy.eval()

            if (self.epoch % cfg.training.rollout_every) == 0 and run_rollout and env_runner is not None:
                runner_log = env_runner.run(policy)
                step_log.update(runner_log)

            if (self.epoch % cfg.training.val_every) == 0 and run_validation:
                val_losses = self.eval_traj_denoise(
                    policy=policy,
                    val_dataloader=val_dataloader,
                    device=device
                )
                if len(val_losses) > 0:
                    step_log["val_loss"] = float(np.mean(val_losses))

                val_action_losses = self.eval_traj(
                    policy=policy,
                    val_dataloader=val_dataloader,
                    device=device
                )
                if len(val_action_losses) > 0:
                    step_log["val_action_loss"] = float(np.mean(val_action_losses))
                    step_log.update(self._last_eval_traj_metrics)

            if (self.epoch % cfg.training.sample_every) == 0 and train_sampling_batch is not None:
                with torch.no_grad():
                    batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                    obs_dict = batch["obs"]
                    gt_action = batch["action"]

                    result = policy.predict_action(obs_dict)
                    pred_action = result["action_pred"]

                    p_quat = pred_action[..., 3:7]
                    g_quat = gt_action[..., 3:7]
                    sign = torch.sign((p_quat * g_quat).sum(dim=-1, keepdim=True))
                    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
                    p_quat_aligned = p_quat * sign

                    mse_pos = torch.nn.functional.mse_loss(pred_action[..., :3], gt_action[..., :3])
                    mse_rot = torch.nn.functional.mse_loss(p_quat_aligned, g_quat)

                    step_log["train_action_mse_pos"] = mse_pos.item()
                    step_log["train_action_mse_rot"] = mse_rot.item()

                    try:
                        metrics = self._compute_traj_metrics(pred_action, gt_action)
                        step_log.update({f"train_sample_{k}": v for k, v in metrics.items()})
                        self._print_traj_metrics(
                            title=f"[Epoch {self.epoch} | Train sample full64 middle metrics]",
                            metrics=metrics,
                            color="yellow"
                        )
                    except Exception as exc:
                        print(f"Train metric print error: {exc}")

                    del batch, obs_dict, gt_action, result, pred_action

            if env_runner is None:
                step_log["test_mean_score"] = -train_loss

            self._maybe_update_best_val(step_log)
            self._print_epoch_decision_summary(step_log)

            if run_ckpt and cfg.checkpoint.save_ckpt:
                self._save_periodic_and_latest_checkpoints(
                    cfg=cfg,
                    topk_manager=topk_manager,
                    step_log=step_log,
                    checkpoint_every=checkpoint_every
                )

            policy.train()

            wandb_log = {
                k: v for k, v in step_log.items()
                if k not in ["test_mean_score", "global_step", "epoch"]
            }
            wandb_run.log(wandb_log, step=self.global_step)

            self.global_step += 1
            self.epoch += 1

            del step_log

    # def _save_periodic_and_latest_checkpoints(self, cfg, topk_manager, step_log, checkpoint_every: int):
    #     """
    #     checkpoint 规则：
    #     1. 第 100 / 200 / 300 ... 个 epoch 结束后保存 epoch=0100.ckpt / epoch=0200.ckpt ...
    #     2. 同时刷新 latest.ckpt
    #     3. 如果 top-k monitor key 存在，则额外保存 top-k checkpoint
    #     """
    #     completed_epoch = self.epoch + 1

    #     should_save_periodic = (
    #         completed_epoch > 0
    #         and completed_epoch % checkpoint_every == 0
    #     )

    #     if should_save_periodic:
    #         if cfg.checkpoint.save_last_ckpt:
    #             periodic_tag = f"epoch={completed_epoch:04d}"
    #             periodic_path = self.save_checkpoint(tag=periodic_tag)
    #             latest_path = self.save_checkpoint(tag="latest")

    #             cprint(f"[Checkpoint] saved periodic checkpoint: {periodic_path}", "green")
    #             cprint(f"[Checkpoint] updated latest checkpoint: {latest_path}", "green")

    #         metric_dict = {k.replace("/", "_"): v for k, v in step_log.items()}
    #         metric_dict["epoch"] = completed_epoch

    #         topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
    #         if topk_ckpt_path is not None:
    #             topk_path = self.save_checkpoint(path=topk_ckpt_path)
    #             cprint(f"[Checkpoint] saved top-k checkpoint: {topk_path}", "green")

    #         if cfg.checkpoint.save_last_snapshot:
    #             snapshot_path = self.save_snapshot(tag=f"epoch={completed_epoch:04d}")
    #             cprint(f"[Checkpoint] saved snapshot: {snapshot_path}", "green")

    #     elif self.epoch == 0 and cfg.checkpoint.save_last_ckpt:
    #         latest_path = self.save_checkpoint(tag="latest")
    #         cprint(f"[Checkpoint] epoch 0 latest checkpoint initialized: {latest_path}", "green")

    def _save_periodic_and_latest_checkpoints(self, cfg, topk_manager, step_log, checkpoint_every: int):
        """
        checkpoint 规则：
        1. 每 checkpoint_every 个 completed epoch 保存 epoch=0100.ckpt / epoch=0200.ckpt ...
        2. 每 checkpoint_every 个 completed epoch 同时刷新 latest.ckpt
        3. 只有当 top-k monitor key 存在于当前 step_log 时，才保存 top-k checkpoint
        """
        completed_epoch = self.epoch + 1

        should_save_periodic = (
            completed_epoch > 0
            and completed_epoch % checkpoint_every == 0
        )

        if should_save_periodic:
            periodic_tag = f"epoch={completed_epoch:04d}"
            periodic_path = self.save_checkpoint(tag=periodic_tag)

            if cfg.checkpoint.save_last_ckpt:
                latest_path = self.save_checkpoint(tag="latest")
                cprint(f"[Checkpoint] updated latest checkpoint: {latest_path}", "green")

            cprint(f"[Checkpoint] saved periodic checkpoint: {periodic_path}", "green")

            metric_dict = {k.replace("/", "_"): v for k, v in step_log.items()}
            metric_dict["epoch"] = completed_epoch

            monitor_key = cfg.checkpoint.topk.monitor_key

            if monitor_key in metric_dict:
                topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                if topk_ckpt_path is not None:
                    topk_path = self.save_checkpoint(path=topk_ckpt_path)
                    cprint(f"[Checkpoint] saved top-k checkpoint: {topk_path}", "green")
            else:
                cprint(
                    f"[Checkpoint] skip top-k: monitor key '{monitor_key}' not found in this epoch. "
                    f"This is normal when checkpoint epoch is not a validation epoch.",
                    "yellow"
                )

        elif self.epoch == 0 and cfg.checkpoint.save_last_ckpt:
            latest_path = self.save_checkpoint(tag="latest")
            cprint(f"[Checkpoint] epoch 0 latest checkpoint initialized: {latest_path}", "green")

    def _rotation_error_deg(self, pred_quat, gt_quat):
        pred_quat = torch.nn.functional.normalize(pred_quat, dim=-1)
        gt_quat = torch.nn.functional.normalize(gt_quat, dim=-1)
        dot = torch.sum(pred_quat * gt_quat, dim=-1).abs().clamp(0.0, 1.0)
        return torch.rad2deg(2.0 * torch.acos(dot))

    def _percentile(self, x, q: float):
        x = x.reshape(-1).float()
        if x.numel() == 0:
            return float("nan")
        return torch.quantile(x, q / 100.0).item()

    def _compute_traj_metrics(self, pred_action, gt_action):
        pred_action = pred_action.detach()
        gt_action = gt_action.detach().to(pred_action.device)

        pos_err_cm = torch.linalg.norm(
            pred_action[..., :3] - gt_action[..., :3],
            dim=-1
        ) * 100.0

        rot_err_deg = self._rotation_error_deg(
            pred_action[..., 3:7],
            gt_action[..., 3:7]
        )

        T = pred_action.shape[1]
        if T > 2:
            middle = slice(1, T - 1)
        else:
            middle = slice(0, T)

        middle_pos = pos_err_cm[:, middle]
        middle_rot = rot_err_deg[:, middle]

        middle_pos_per_sample = middle_pos.mean(dim=1)
        middle_rot_per_sample = middle_rot.mean(dim=1)

        gt_endpoint_delta_cm = torch.linalg.norm(
            gt_action[:, -1, :3] - gt_action[:, 0, :3],
            dim=-1
        ) * 100.0

        gt_step_delta_cm = torch.linalg.norm(
            gt_action[:, 1:, :3] - gt_action[:, :-1, :3],
            dim=-1
        ) * 100.0
        gt_path_length_cm = gt_step_delta_cm.sum(dim=1)

        middle_pos_err_over_endpoint_pct = (
            middle_pos_per_sample / gt_endpoint_delta_cm.clamp_min(1e-6) * 100.0
        ).mean().item()

        middle_pos_err_over_path_pct = (
            middle_pos_per_sample / gt_path_length_cm.clamp_min(1e-6) * 100.0
        ).mean().item()

        metrics = {
            "middle_traj_pos_err_cm_mean": middle_pos.mean().item(),
            "middle_traj_rot_err_deg_mean": middle_rot.mean().item(),
            "full_traj_pos_err_cm_mean": pos_err_cm.mean().item(),
            "full_traj_rot_err_deg_mean": rot_err_deg.mean().item(),

            "middle_traj_pos_err_cm_p50": self._percentile(middle_pos, 50),
            "middle_traj_pos_err_cm_p90": self._percentile(middle_pos, 90),
            "middle_traj_pos_err_cm_max": middle_pos.max().item(),

            "middle_traj_rot_err_deg_p50": self._percentile(middle_rot, 50),
            "middle_traj_rot_err_deg_p90": self._percentile(middle_rot, 90),
            "middle_traj_rot_err_deg_max": middle_rot.max().item(),

            "gt_endpoint_delta_cm_mean": gt_endpoint_delta_cm.mean().item(),
            "gt_path_length_cm_mean": gt_path_length_cm.mean().item(),
            "middle_pos_err_over_endpoint_pct": middle_pos_err_over_endpoint_pct,
            "middle_pos_err_over_path_pct": middle_pos_err_over_path_pct,

            "boundary_start_pos_err_cm": pos_err_cm[:, 0].mean().item(),
            "boundary_start_rot_err_deg": rot_err_deg[:, 0].mean().item(),
            "boundary_end_pos_err_cm": pos_err_cm[:, -1].mean().item(),
            "boundary_end_rot_err_deg": rot_err_deg[:, -1].mean().item(),

            "num_eval_samples": float(pred_action.shape[0]),
            "num_eval_middle_frames": float(pred_action.shape[0] * max(T - 2, 1)),
        }

        for step in (16, 32, 48):
            if step < T:
                metrics[f"step{step}_pos_err_cm"] = pos_err_cm[:, step].mean().item()
                metrics[f"step{step}_rot_err_deg"] = rot_err_deg[:, step].mean().item()

        return metrics

    def _print_traj_metrics(self, title, metrics, color="cyan"):
        cprint(f"\n{title}", color)
        print("  ▶ Main learning metric excludes hard-clamped boundary frames 0 and 63.")

        print(
            f"  ▶ middle[1:62] pos mean/p50/p90/max: "
            f"{metrics['middle_traj_pos_err_cm_mean']:.2f} / "
            f"{metrics.get('middle_traj_pos_err_cm_p50', float('nan')):.2f} / "
            f"{metrics.get('middle_traj_pos_err_cm_p90', float('nan')):.2f} / "
            f"{metrics.get('middle_traj_pos_err_cm_max', float('nan')):.2f} cm"
        )

        print(
            f"  ▶ middle[1:62] rot mean/p50/p90/max: "
            f"{metrics['middle_traj_rot_err_deg_mean']:.2f} / "
            f"{metrics.get('middle_traj_rot_err_deg_p50', float('nan')):.2f} / "
            f"{metrics.get('middle_traj_rot_err_deg_p90', float('nan')):.2f} / "
            f"{metrics.get('middle_traj_rot_err_deg_max', float('nan')):.2f} deg"
        )

        print(
            f"  ▶ GT scale: endpoint delta {metrics['gt_endpoint_delta_cm_mean']:.2f} cm | "
            f"path length {metrics['gt_path_length_cm_mean']:.2f} cm"
        )

        print(
            f"  ▶ relative middle pos err: "
            f"{metrics['middle_pos_err_over_endpoint_pct']:.1f}% of endpoint delta | "
            f"{metrics['middle_pos_err_over_path_pct']:.1f}% of path length"
        )

        for step in (16, 32, 48):
            pos_key = f"step{step}_pos_err_cm"
            rot_key = f"step{step}_rot_err_deg"
            if pos_key in metrics:
                print(
                    f"  ▶ step {step:02d} pos: {metrics[pos_key]:.2f} cm | "
                    f"rot: {metrics[rot_key]:.2f} deg"
                )

        print(
            f"  ▶ boundary start pos: {metrics['boundary_start_pos_err_cm']:.4f} cm | "
            f"rot: {metrics['boundary_start_rot_err_deg']:.4f} deg"
        )
        print(
            f"  ▶ boundary end   pos: {metrics['boundary_end_pos_err_cm']:.4f} cm | "
            f"rot: {metrics['boundary_end_rot_err_deg']:.4f} deg"
        )
        print("  ▶ Step 63 is hard-clamped and is not used as the primary learning metric.")
        print("-" * 70)

    def _maybe_update_best_val(self, step_log):
        pos_key = "val_sample_middle_traj_pos_err_cm_mean"
        rot_key = "val_sample_middle_traj_rot_err_deg_mean"

        if pos_key not in step_log:
            return

        val_pos = float(step_log[pos_key])
        val_rot = float(step_log.get(rot_key, float("nan")))

        if val_pos < self._best_val_metrics["middle_pos"]:
            self._best_val_metrics = {
                "epoch": int(self.epoch),
                "middle_pos": val_pos,
                "middle_rot": val_rot,
            }

    def _print_epoch_decision_summary(self, step_log):
        train_pos = step_log.get("train_sample_middle_traj_pos_err_cm_mean", None)
        train_rot = step_log.get("train_sample_middle_traj_rot_err_deg_mean", None)

        val_pos = step_log.get("val_sample_middle_traj_pos_err_cm_mean", None)
        val_rot = step_log.get("val_sample_middle_traj_rot_err_deg_mean", None)

        cprint(f"\n[Epoch {self.epoch} | Decision Summary]", "green")

        if train_pos is not None:
            print(f"  ▶ Train sample middle: {train_pos:.2f} cm | {train_rot:.2f} deg")

        if val_pos is not None:
            print(f"  ▶ Val middle:          {val_pos:.2f} cm | {val_rot:.2f} deg")

        if train_pos is not None and val_pos is not None:
            print(
                f"  ▶ Train-Val gap:       {val_pos - train_pos:+.2f} cm | "
                f"{val_rot - train_rot:+.2f} deg"
            )

        if self._best_val_metrics["epoch"] >= 0:
            print(
                f"  ▶ Best Val so far:     epoch {self._best_val_metrics['epoch']} | "
                f"{self._best_val_metrics['middle_pos']:.2f} cm | "
                f"{self._best_val_metrics['middle_rot']:.2f} deg"
            )

        print("  ▶ Checkpoint selection should use Val middle error, not boundary error.")
        print("-" * 70)

    def eval_traj_denoise(self, policy=None, val_dataloader=None, device=None, load_checkpoint=False):
        cfg = copy.deepcopy(self.cfg)

        if load_checkpoint:
            latest_ckpt_path = self.get_checkpoint_path(tag="latest")
            if latest_ckpt_path.is_file():
                self.load_checkpoint(path=latest_ckpt_path)

        if device is None:
            device = self._device(cfg)

        if policy is None:
            policy = self.ema_model if (cfg.training.use_ema and self.ema_model is not None) else self.model

        if val_dataloader is None:
            dataset: BaseDataset = hydra.utils.instantiate(cfg.task.dataset)
            val_dataset = dataset.get_validation_dataset()
            val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
            policy.set_normalizer(dataset.get_normalizer())

        policy.eval()

        with torch.no_grad():
            val_losses = []
            with tqdm.tqdm(
                val_dataloader,
                desc=f"Validation Denoise {self.epoch}",
                leave=False
            ) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    loss, _loss_dict = policy.compute_loss(batch)
                    val_losses.append(float(loss.item()))

                    if (
                        cfg.training.max_val_steps is not None
                        and batch_idx >= cfg.training.max_val_steps - 1
                    ):
                        break

            return val_losses

    def eval_traj(self, policy=None, val_dataloader=None, device=None, load_checkpoint=False):
        cfg = copy.deepcopy(self.cfg)

        if load_checkpoint:
            latest_ckpt_path = self.get_checkpoint_path(tag="latest")
            if latest_ckpt_path.is_file():
                self.load_checkpoint(path=latest_ckpt_path)

        if device is None:
            device = self._device(cfg)

        if policy is None:
            policy = self.ema_model if (cfg.training.use_ema and self.ema_model is not None) else self.model

        if val_dataloader is None:
            dataset: BaseDataset = hydra.utils.instantiate(cfg.task.dataset)
            val_dataset = dataset.get_validation_dataset()
            val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
            policy.set_normalizer(dataset.get_normalizer())

        policy.eval()

        with torch.no_grad():
            val_losses = []
            metric_accumulator = []

            with tqdm.tqdm(
                val_dataloader,
                desc=f"Validation Action {self.epoch}",
                leave=False
            ) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                    result = policy.predict_action(batch["obs"], target=batch["action"])

                    if "loss" in result:
                        val_losses.append(float(result["loss"]))

                    pred_action = result["action_pred"]
                    gt_action = batch["action"]
                    metric = self._compute_traj_metrics(pred_action, gt_action)
                    metric_accumulator.append(metric)

                    if batch_idx == 0:
                        try:
                            self._print_traj_metrics(
                                title=f"[Epoch {self.epoch} | Val sample full64 middle metrics]",
                                metrics=metric,
                                color="cyan"
                            )
                        except Exception as exc:
                            print(f"Val metric print error: {exc}")

                    if (
                        cfg.training.max_val_steps is not None
                        and batch_idx >= cfg.training.max_val_steps - 1
                    ):
                        break

            if len(metric_accumulator) > 0:
                keys = metric_accumulator[0].keys()
                self._last_eval_traj_metrics = {
                    f"val_sample_{k}": float(np.mean([m[k] for m in metric_accumulator]))
                    for k in keys
                }
            else:
                self._last_eval_traj_metrics = {}

            return val_losses

    def eval(self):
        pass

    def save_checkpoint(
        self,
        path=None,
        tag="latest",
        exclude_keys=None,
        include_keys=None,
        use_thread=False
    ):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")
        else:
            path = pathlib.Path(path)

        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ("_output_dir",)

        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "cfg": self.cfg,
            "state_dicts": {},
            "pickles": {},
            "output_dir": self.output_dir,
            "task_save_name": getattr(self, "task_save_name", None),
            "run_timestamp": getattr(self, "run_timestamp", None),
        }

        for key, value in self.__dict__.items():
            if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
                if key not in exclude_keys:
                    if use_thread:
                        payload["state_dicts"][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload["state_dicts"][key] = value.state_dict()
            elif key in include_keys:
                payload["pickles"][key] = dill.dumps(value)

        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda: torch.save(payload, path.open("wb"), pickle_module=dill)
            )
            self._saving_thread.start()
        else:
            torch.save(payload, path.open("wb"), pickle_module=dill)

        del payload
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return str(path.absolute())

    def get_checkpoint_path(self, tag="latest"):
        if tag == "latest":
            return pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")

        if tag == "best":
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath("checkpoints")
            all_checkpoints = os.listdir(checkpoint_dir)

            best_ckpt = None
            best_score = float("inf")

            for ckpt in all_checkpoints:
                if "latest" in ckpt:
                    continue

                if "val_mid_pos_cm=" in ckpt:
                    try:
                        score_str = ckpt.split("val_mid_pos_cm=")[1].split(".ckpt")[0]
                        score = float(score_str)
                    except Exception:
                        continue

                    if score < best_score:
                        best_ckpt = ckpt
                        best_score = score

            if best_ckpt is None:
                raise FileNotFoundError(f"No best checkpoint found in {checkpoint_dir}")

            return pathlib.Path(self.output_dir).joinpath("checkpoints", best_ckpt)

        return pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload["pickles"].keys()

        for key, value in payload["state_dicts"].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)

        for key in include_keys:
            if key in payload["pickles"]:
                self.__dict__[key] = dill.loads(payload["pickles"][key])

    def load_checkpoint(self, path=None, tag="latest", exclude_keys=None, include_keys=None, **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)

        payload = torch.load(path.open("rb"), pickle_module=dill, map_location="cpu")
        self.load_payload(
            payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs
        )
        return payload

    @classmethod
    def create_from_checkpoint(cls, path, exclude_keys=None, include_keys=None, **kwargs):
        payload = torch.load(open(path, "rb"), pickle_module=dill)
        instance = cls(payload["cfg"])
        instance.load_payload(
            payload=payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs
        )
        return instance

    def save_snapshot(self, tag="latest"):
        path = pathlib.Path(self.output_dir).joinpath("snapshots", f"{tag}.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, path.open("wb"), pickle_module=dill)
        return str(path.absolute())

    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, "rb"), pickle_module=dill)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).resolve().parents[2].joinpath("configs/model")),
    config_name="train_dp3_goal_full64",
)
def main(cfg):
    workspace = TrainDP3Workspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()

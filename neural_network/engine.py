# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 2022 Zhuang Liu (Meta) - liuzhuangthu@gmail.com

import math
import matplotlib.pyplot as plt
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    wandb_logger=None,
    start_steps=None,
    lr_schedule_values=None,
    wd_schedule_values=None,
    num_training_steps_per_epoch=None,
    update_freq=None,
    use_amp=False,
    train_history_file=None,
    iter_eval=False,
    data_loader_val=None,
    val_iter_history_file=None,
    num_validation_steps_per_epoch=None,
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "min_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10
    iter_print_freq = (len(data_loader) + 1) // 6
    if iter_eval == True:
        val_iter_step = 0

    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if (
            lr_schedule_values is not None
            or wd_schedule_values is not None
            and data_iter_step % update_freq == 0
        ):
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        ### DATASET VISUALIZATION BEFORE TRAINING ###
        # savepath = (
        #     "/nfsroot/home1/homedirs/gusso/Tesi_codici/Python/convnext_1d/plots/test"
        #     + str(epoch)
        #     + ".png"
        # )
        # y = samples[0][0]
        # fig, ax = plt.subplots()
        # plt.title(str(targets[0]))
        # ax.plot(y, "-c", linewidth=0.2)
        # ax.plot(y, ".b", markersize=1)
        # ax.grid(True, axis="both")
        # plt.draw()
        # plt.savefig(savepath)
        # plt.close()
        # print("# Figure saved at " + savepath)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        else:  # full precision
            output = model(samples)
            loss = criterion(output, targets)

        loss_value = loss.item()
        # print("loss {}, output {}, target {}".format(loss_value, output, targets))

        if not math.isfinite(loss_value):  # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = (
                hasattr(optimizer, "is_second_order") and optimizer.is_second_order
            )
            loss /= update_freq
            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=(data_iter_step + 1) % update_freq == 0,
            )
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else:  # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if train_history_file:
            train_iteration = start_steps * update_freq + data_iter_step
            train_history_file.write(
                "{:>7} {:>9} {:<13.10f} {:<13.8f} {:<32.30f}\n".format(
                    epoch,
                    train_iteration,
                    loss_value,
                    class_acc,
                    param_group["lr"],
                )
            )

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log(
                {
                    "Rank-0 Batch Wise/train_loss": loss_value,
                    "Rank-0 Batch Wise/train_max_lr": max_lr,
                    "Rank-0 Batch Wise/train_min_lr": min_lr,
                },
                commit=False,
            )
            if class_acc:
                wandb_logger._wandb.log(
                    {"Rank-0 Batch Wise/train_class_acc": class_acc}, commit=False
                )
            if use_amp:
                wandb_logger._wandb.log(
                    {"Rank-0 Batch Wise/train_grad_norm": grad_norm}, commit=False
                )
            wandb_logger._wandb.log({"Rank-0 Batch Wise/global_train_step": it})

        if iter_eval == True:
            if (step + 1) % iter_print_freq == 0:
                start_iter_eval_steps = (
                    epoch * ((len(data_loader) + 1) // iter_print_freq) + val_iter_step
                ) * num_validation_steps_per_epoch
                evaluate(
                    data_loader_val,
                    model,
                    criterion,
                    device,
                    use_amp=use_amp,
                    val_history_file=val_iter_history_file,
                    epoch=epoch,
                    start_steps=start_iter_eval_steps,
                )
                val_iter_step += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# @torch.no_grad()
# def evaluate(
#     data_loader,
#     model,
#     criterion,
#     device,
#     use_amp=False,
#     val_history_file=False,
#     epoch=None,
#     start_steps=None,
# ):

#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = "Test:"

#     # switch to evaluation mode
#     model.eval()
#     for data_iter_step, batch in enumerate(
#         metric_logger.log_every(data_loader, 10, header)
#     ):
#         samples = batch[0]
#         target = batch[-1]

#         samples = samples.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)

#         # compute output
#         if use_amp:
#             with torch.cuda.amp.autocast():
#                 output = model(samples)
#                 loss = criterion(output, target)
#         else:
#             output = model(samples)
#             loss = criterion(output, target)

#         acc1 = accuracy(output, target, topk=(1,))[0]

#         if val_history_file:
#             val_iteration = start_steps + data_iter_step
#             val_history_file.write(
#                 "{:>7} {:>9} {:<13.10f} {:<13.8f}\n".format(
#                     epoch, val_iteration, loss.item(), acc1.item()
#                 )
#             )

#         batch_size = samples.shape[0]
#         metric_logger.update(loss=loss.item())
#         metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print(
#         "* Acc1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}".format(
#             top1=metric_logger.acc1, losses=metric_logger.loss
#         )
#     )

#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(
    data_loader,
    model,
    criterion,
    device,
    use_amp=False,
    val_history_file=False,
    ROC_history_file=False,
    epoch=None,
    start_steps=None,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    model.eval()

    all_preds = []
    all_targets = []

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        samples = batch[0]
        target = batch[-1]

        samples = samples.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, target)
        else:
            output = model(samples)
            loss = criterion(output, target)

        acc1 = accuracy(output, target, topk=(1,))[0]

        preds = output.argmax(dim=1)
        all_preds.append(preds)
        all_targets.append(target)

        if val_history_file:
            val_iteration = start_steps + data_iter_step
            val_history_file.write(
                "{:>7} {:>9} {:<13.10f} {:<13.8f}\n".format(
                    epoch, val_iteration, loss.item(), acc1.item()
                )
            )

        if ROC_history_file:
            for i in range(target.size(0)):
                score = torch.softmax(output[i], dim=0)[1].item()
                label = target[i].item()
                ROC_history_file.write(f"{epoch} {score:.10f} {label}\n")

        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)

    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds).to(device)
    all_targets = torch.cat(all_targets).to(device)

    # Confusion matrix components for binary classification
    TP = ((all_preds == 1) & (all_targets == 1)).sum().item()
    TN = ((all_preds == 0) & (all_targets == 0)).sum().item()
    FP = ((all_preds == 1) & (all_targets == 0)).sum().item()
    FN = ((all_preds == 0) & (all_targets == 1)).sum().item()

    # Log to metric_logger
    metric_logger.update(TP=TP, TN=TN, FP=FP, FN=FN)

    # Optionally write to history file
    if val_history_file:
        #val_history_file.write(f"Confusion Matrix (Epoch {epoch}): TP={TP}, TN={TN}, FP={FP}, FN={FN}\n")
        val_history_file.write(
            "ConfMatr {:>7} {:>7} {:>7} {:>7} {:>7} \n".format(
                epoch, TP, TN, FP, FN
            )
        )

    metric_logger.synchronize_between_processes()

    print(
        "* Acc1 {top1.global_avg:.3f} loss {losses.global_avg:.3f} TP {TP} TN {TN} FP {FP} FN {FN}".format(
            top1=metric_logger.acc1,
            losses=metric_logger.loss,
            TP=TP,
            TN=TN,
            FP=FP,
            FN=FN
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

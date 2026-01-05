import os
import random

import numpy as np
import torch
import wandb

from utils import accuracy, AverageMeter


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


def val(model, dataloader, epoch=9999, test=False, criterion=None):
    acc1_meter = AverageMeter(name='accuracy top 1')
    acc5_meter = AverageMeter(name='accuracy top 5')
    loss_meter = AverageMeter(name='loss')
    n_iters = len(dataloader)
    model.eval()
    outputs_all = []
    image_names_all = []
    with torch.no_grad():
        for iter_idx, (images, labels, image_paths) in enumerate(dataloader):

            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            if test:
                outputs_all.append(outputs.detach().cpu().numpy())
                image_names_all.extend([os.path.splitext(os.path.basename(path))[0] for path in image_paths])
                continue

            loss = criterion(outputs, labels)
            loss_meter.update(loss.item(), images.shape[0])

            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            acc1_meter.update(acc1[0], images.shape[0])
            acc5_meter.update(acc5[0], images.shape[0])

            print(f"[Epoch {epoch}] iter {iter_idx} / {n_iters}: \tLoss {loss_meter.val:.4f}({loss_meter.avg:.4f}) \tAcc top-1 {acc1_meter.val:.2f}({acc1_meter.avg:.2f}) \tAcc top-5 {acc5_meter.val:.2f}({acc5_meter.avg:.2f})", end='\r')
    print("")
    print(f"Epoch {epoch} validation: top-1 acc {acc1_meter.avg} top-5 acc {acc5_meter.avg} loss {loss_meter.avg}")
    if not test:
        wandb.log({
            "epoch": epoch,
            "val/acc1": acc1_meter.avg,
            "val/acc5": acc5_meter.avg,
            "val/loss": loss_meter.avg
        }, step=epoch)
    if test:
        outputs_all = np.concatenate(outputs_all)
    return acc1_meter.avg, acc5_meter.avg, image_names_all, outputs_all, loss_meter.avg
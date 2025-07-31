import os

import numpy as np
import torch

from utils import accuracy, AverageMeter


def val(model, dataloader, epoch=9999, test=False):
    acc1_meter = AverageMeter(name='accuracy top 1')
    acc5_meter = AverageMeter(name='accuracy top 5')
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

            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            acc1_meter.update(acc1[0], images.shape[0])
            acc5_meter.update(acc5[0], images.shape[0])

            print(f"[Epoch {epoch}] iter {iter_idx} / {n_iters}: \tAcc top-1 {acc1_meter.val:.2f}({acc1_meter.avg:.2f}) \tAcc top-5 {acc5_meter.val:.2f}({acc5_meter.avg:.2f})", end='\r')
    print("")
    print(f"Epoch {epoch} validation: top-1 acc {acc1_meter.avg} top-5 acc {acc5_meter.avg}")
    if test:
        outputs_all = np.concatenate(outputs_all)
    return acc1_meter.avg, acc5_meter.avg, image_names_all, outputs_all

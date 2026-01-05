import random
import numpy as np
import torch
import wandb
from tqdm import tqdm
from utils import accuracy, AverageMeter

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def train(model, dataloader, criterion, optimizer, epoch=9999, strength=1.0):
    acc1_meter = AverageMeter(name='accuracy top 1')
    acc5_meter = AverageMeter(name='accuracy top 5')
    loss_meter = AverageMeter(name='loss')
    n_iters = len(dataloader)
    cutmix_prob = 0.5 * strength
    mixup_prob = 0.5 * strength
    cutmix_alpha = 1.0 * strength
    mixup_alpha = 1.0 * strength

    model.train()
    for iter_idx, (images, labels, _) in enumerate(tqdm(dataloader, desc=f"Train Epoch {epoch}")):

        images = images.cuda()
        labels = labels.cuda()

        r = np.random.rand()
        if r < cutmix_prob:
            # CutMix 적용
            lam = np.random.beta(cutmix_alpha, cutmix_alpha)
            batch_size = images.size(0)
            index = torch.randperm(batch_size).cuda()
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
            images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
            labels_a, labels_b = labels, labels[index]
            outputs = model(images)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        elif r < cutmix_prob + mixup_prob:
            # MixUp 적용
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            batch_size = images.size(0)
            index = torch.randperm(batch_size).cuda()
            images = lam * images + (1 - lam) * images[index]
            labels_a, labels_b = labels, labels[index]
            outputs = model(images)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            # 일반 학습
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        loss_meter.update(loss.item(), images.shape[0])
        acc1_meter.update(acc1[0], images.shape[0])
        acc5_meter.update(acc5[0], images.shape[0])

    print("")
    print(f"Epoch {epoch} training finished")

    wandb.log({
        "epoch": epoch,
        "train/loss": loss_meter.avg,
        "train/acc1": acc1_meter.avg,
        "train/acc5": acc5_meter.avg
    }, step=epoch)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
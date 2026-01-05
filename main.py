import argparse
from time import gmtime, strftime
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb
import timm
import math

from arch.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from batch_manager import BatchManagerTinyImageNet
from transforms import get_train_transform, get_val_transform
import train
import val


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='swin_tiny',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                 'deit_small', 'vit_small', 'swin_tiny'])
    parser.add_argument('--lr_base', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--drop_rate', type=float, default=0.2)
    parser.add_argument('--attn_drop_rate', type=float, default=0.1)
    parser.add_argument('--drop_path_rate', type=float, default=0.1)
    args = parser.parse_args()

    wandb.init(project="basic_code_semina_01_code_03")
    wandb.config.update(vars(args))
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")

    # define model
    if args.arch.startswith('resnet'):
        if args.arch == 'resnet18':
            model = resnet18(num_classes=200)
        elif args.arch == 'resnet34':
            model = resnet34(num_classes=200)
        elif args.arch == 'resnet50':
            model = resnet50(num_classes=200)
        elif args.arch == 'resnet101':
            model = resnet101(num_classes=200)
        elif args.arch == 'resnet152':
            model = resnet152(num_classes=200)
        else:
            raise NotImplementedError(f"architecture {args.arch} is not implemented")
    elif args.arch == 'deit_small':
        model = timm.create_model(
            'vit_small_patch16_224', pretrained=True, num_classes=200,
            drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate, drop_path_rate=args.drop_path_rate
        )
    elif args.arch == 'vit_small':
        model = timm.create_model(
            'vit_small_patch16_224', pretrained=True, num_classes=200,
            drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate, drop_path_rate=args.drop_path_rate
        )
    elif args.arch == 'swin_tiny':
        model = timm.create_model(
            'swin_tiny_patch4_window7_224', pretrained=True, num_classes=200,
            drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate, drop_path_rate=args.drop_path_rate
        )
    else:
        raise NotImplementedError(f"architecture {args.arch} is not implemented")

    model = model.cuda()
    model = torch.nn.parallel.DataParallel(model)

    # define dataloaders (train은 epoch마다 새로 정의)
    dataloader_val = DataLoader(
        BatchManagerTinyImageNet(split='val', transform=get_val_transform()),
        shuffle=False, num_workers=10, batch_size=args.batch_size, pin_memory=True
    )
    dataloader_test = DataLoader(
        BatchManagerTinyImageNet(split='test', transform=get_val_transform()),
        shuffle=False, num_workers=10, batch_size=args.batch_size, pin_memory=True
    )

    # define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_base, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # save_path
    current_time = strftime('%Y-%m-%d_%H:%M', gmtime())
    save_dir = os.path.join(f'checkpoints/{current_time}')
    os.makedirs(save_dir, exist_ok=True)

    # train and val
    best_perform, best_epoch = -100, -100
    for epoch in range(1, args.epochs + 1):
        # 🔥 Cosine 스케줄로 증강 강도 증가
        progress = epoch / args.epochs
        aug_strength = 1 - math.cos((math.pi / 2) * progress)

        dataloader_train = DataLoader(
            BatchManagerTinyImageNet(split='train', transform=get_train_transform(aug_strength)),
            shuffle=True, num_workers=10, batch_size=args.batch_size, pin_memory=True
        )

        print(f"Training at epoch {epoch}. Aug strength={aug_strength:.2f}, LR {optimizer.param_groups[0]['lr']}")

        train.train(model, dataloader_train, criterion, optimizer, epoch=epoch)
        acc1, acc5, _, _, _ = val.val(model, dataloader_val, epoch=epoch, criterion=criterion)
        _, _, image_names, preds, _ = val.val(model, dataloader_test, epoch=epoch, test=True)

        scheduler.step()

        save_data = {
            'epoch': epoch,
            'acc1': acc1,
            'acc5': acc5,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(save_data, os.path.join(save_dir, f'{epoch:03d}.pth.tar'))
        if epoch > 1:
            os.remove(os.path.join(save_dir, f'{epoch - 1:03d}.pth.tar'))
        if acc1 >= best_perform:
            torch.save(save_data, os.path.join(save_dir, 'best.pth.tar'))
            best_perform = acc1
            best_epoch = epoch

            # save prediction results
            save_path = os.path.join(save_dir, 'best_test_preds.csv')
            with open(save_path, "w") as f:
                f.write('name,' + ','.join(['cls' + str(i) for i in range(200)]) + '\n')
                for name, pred in zip(image_names, preds):
                    f.write(name + ',' + ','.join([str(p) for p in pred]) + '\n')
            print(f"best test prediction saved at {save_path}")
        print(f"best performance {best_perform} at epoch {best_epoch}")

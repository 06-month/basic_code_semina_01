from utils import accuracy, AverageMeter

def train(model, dataloader, criterion, optimizer, epoch=9999):
    acc1_meter = AverageMeter(name='accuracy top 1')
    acc5_meter = AverageMeter(name='accuracy top 5')
    loss_meter = AverageMeter(name='loss')
    n_iters = len(dataloader)
    model.train()
    for iter_idx, (images, labels, _) in enumerate(dataloader):

        images = images.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        loss_meter.update(loss.item(), images.shape[0])
        acc1_meter.update(acc1[0], images.shape[0])
        acc5_meter.update(acc5[0], images.shape[0])

        print(f"[Epoch {epoch}] iter {iter_idx} / {n_iters}: \tLoss {loss_meter.val:.4f}({loss_meter.avg:.4f}) \tAcc top-1 {acc1_meter.val:.2f}({acc1_meter.avg:.2f}) \tAcc top-5 {acc5_meter.val:.2f}({acc5_meter.avg:.2f})", end='\r')
    print("")
    print(f"Epoch {epoch} training finished")

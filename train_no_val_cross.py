import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from data.card_voc import VOCDetection, detection_collate, AnnotationTransform
from data.preproc import preproc
from util.config import train_cfg, model_cfg, data_cfg
from peleenet import load_model, PeleeNet, save_step
from util.util import init_summery, AverageMeter, adjust_lr, accu, add_summery, LabelSmoothSoftmaxCEV1

parser = argparse.ArgumentParser(description='Card Classification')
parser.add_argument('--task_name', default='train_and_val', help='tag task name for training')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
args = parser.parse_args()


def train(train_loader, model, optimizer, criterion, writer, count, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    phase = 'train'
    model.train()
    iter_len = train_loader.__len__()
    for i in range(0, iter_len//train_cfg['per_batch_size']):
        ### Adjust learning rate
        # lr = adjust_lr_test(optimizer, count, train_cfg['init_lr'])
        model.train()
        batch_iterator = iter(DataLoader(train_loader, train_cfg['per_batch_size'], shuffle=True, num_workers=args.num_workers,
                                         collate_fn=detection_collate))
        input, target = next(batch_iterator)
        if train_cfg['cuda']:
            target = target.cuda()
            input = input.cuda()

        # input_var = torch.autograd.Variable(input)
        # target_var = torch.autograd.Variable(target)

        # compute output
        with torch.set_grad_enabled(phase == 'train'):
            output = model(input)
            loss = criterion(output, target.long())
            acc1, acc5 = accu(output.data, target, topk=(1, 5))

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += 1
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=count)
        writer = add_summery(writer, 'train', loss.data, acc1, acc5, count)

        if i % train_cfg['print_que'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'lr: [{3}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader)//train_cfg['per_batch_size'], optimizer.param_groups[0]['lr'], loss=losses, top1=top1, top5=top5))

    return count, losses.avg


def val(dataset, model, optimizer, criterion, writer, count, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    phase = 'test'
    iter_len = dataset.__len__()
    for i in range(0, iter_len//train_cfg['per_batch_size']):
        ### Adjust learning rate
        # lr = adjust_lr_test(optimizer, count, train_cfg['init_lr'])
        model.eval()
        batch_iterator = iter(DataLoader(dataset, train_cfg['per_batch_size'], shuffle=True, num_workers=args.num_workers,
                                         collate_fn=detection_collate))
        input, target = next(batch_iterator)
        if train_cfg['cuda']:
            target = target.cuda()
            input = input.cuda()

        # input_var = torch.autograd.Variable(input)
        # target_var = torch.autograd.Variable(target)

        # compute output
        with torch.set_grad_enabled(phase == phase):
            output = model(input)
            loss = criterion(output, target.long())
            acc1, acc5 = accu(output.data, target, topk=(1, 5))

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        count += 1
        # compute gradient and do SGD step
    writer.add_scalar('val/loss', losses.avg, global_step=count)
    writer.add_scalar('val/acc_top1', acc1, global_step=count)
    writer.add_scalar('val/acc_top5', acc5, global_step=count)

    print('Epoch: [{0}]\t'
              'lr: [{1}]\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            epoch, optimizer.param_groups[0]['lr'], loss=losses, top1=top1, top5=top5))

    return losses.avg, acc1, acc5


def train_warmup(train_loader, model, optimizer, criterion, epoch, writer, count):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    phase = 'train'
    model.train()
    for i, (input, target) in enumerate(train_loader):
        ### Adjust learning rate
        lr = adjust_lr(optimizer, epoch, count, train_cfg['init_lr'],
                  data_cfg['iterations_per_epoch'], method='warmup')
        if train_cfg['cuda']:
            target = target.cuda()
            input = input.cuda()

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        with torch.set_grad_enabled(phase == 'train'):
            output = model(input_var)
            loss = criterion(output, target_var)
            acc1, acc5 = accu(output.data, target_var, topk=(1, 5))

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += 1
        writer.add_scalar('lr', lr, global_step=count)
        writer = add_summery(writer, 'train', loss.data, acc1, acc5, count)

        if i % train_cfg['print_que'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'lr: [{3}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), lr, loss=losses, top1=top1, top5=top5))

    return count


if __name__ == '__main__':
    # load data
    train_set = VOCDetection(train_cfg['data_set_dir'], 'train', preproc(224, (104, 117, 123)), AnnotationTransform())
    val_set = VOCDetection(train_cfg['data_set_dir'], 'val', preproc(224, (104, 117, 123)), AnnotationTransform())

    # load model
    if model_cfg['if_pre_train']:
        model = load_model(model_cfg['pre_trained_path'], data_classes=model_cfg['data_classes'])
    else:
        model = PeleeNet(num_classes=model_cfg['model_classes'])
    if train_cfg['cuda']:
        model = model.cuda()

    # init the summery
    writer = init_summery('runs', args.task_name)

    # global optimizer_conv
    optimizer_conv = optim.Adam(model.parameters(), lr=train_cfg['init_lr'], weight_decay=0.001)
    schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer_conv, factor=0.1, patience=2, verbose=False,
                                                    threshold=0.1, cooldown=0, min_lr=train_cfg['min_lr'])
    criterion = nn.CrossEntropyLoss()
    
    # criterion = LabelSmoothSoftmaxCEV1(lb_smooth=0.1)
    count = 0

    for epoch in range(train_cfg['num_epochs']):
        if epoch < 0:
            count = train_warmup(train_set, model, optimizer_conv, criterion, epoch, writer, count)
        else:
            count, train_loss = train(train_set, model, optimizer_conv, criterion, writer, count, epoch)
            print("train_loss : {0}, lr : {1}".format(train_loss, optimizer_conv.param_groups[0]['lr']))
            schedule.step(train_loss)

        losses, acc1, acc5 = val(val_set, model, optimizer_conv, criterion, writer, count, epoch)


        if_best_model = False
        filepath = 'weights/epoch_' + str(epoch) + 'checkpoint.pth.tar'
        save_step({
            'epoch': epoch + 1,
            'arch': 'peleenet',
            'state_dict': model.state_dict(),
        }, if_best_model, 'test', filename=filepath)






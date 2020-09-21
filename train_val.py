import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_data_sets
from util.config import train_cfg, model_cfg, data_cfg
from peleenet import load_model, PeleeNet, save_step
from util.util import init_summery, AverageMeter, adjust_lr, accu, add_summery, LabelSmoothSoftmaxCEV1
import time


def train(train_loader, val_set, model, optimizer, criterion, writer, count, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    #val_iterval = 1000

    # switch to train mode
    #phase = 'train'
    #model.train()
    for i, (input, target) in enumerate(train_loader):
        ### Adjust learning rate
        # lr = adjust_lr_test(optimizer, count, train_cfg['init_lr'])
        phase = 'train'
        model.train()
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
        """losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))"""

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        count += 1
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=count)
        writer = add_summery(writer, 'train', loss.data, acc1, acc5, count)

        if i % train_cfg['print_que'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'lr: [{3}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), optimizer.param_groups[0]['lr'], loss=losses, top1=top1, top5=top5))
        
        """if i % val_iterval == 0:
            val_top1, val_top3, val_top5, val_loss = validate(val_set, model, criterion)
            writer = add_summery(writer, 'val', val_loss, val_top1, val_top5, count)"""

    return count, losses.avg

def train_warmup(train_loader, val_set, model, optimizer, criterion, epoch, writer, count):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    #phase = 'train'
    #model.train()
    """if epoch == 0:
        val_interval = 1000
    elif epoch == 1:
        val_interval = 1000"""
    for i, (input, target) in enumerate(train_loader):
        phase = 'train'
        model.train()
        ### Adjust learning rate
        lr = adjust_lr(optimizer, epoch, count, train_cfg['init_lr'],
                  data_cfg['iterations_per_epoch'], method='warmup')
        if train_cfg['cuda']:
            target = target.cuda()
            input = input.cuda()

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        phase = 'train'

        # compute output
        with torch.set_grad_enabled(phase == 'train'):
            output = model(input_var)
            loss = criterion(output, target_var)
            acc1, acc5 = accu(output.data, target_var, topk=(1, 5))

        # measure accuracy and record loss
        """losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))"""

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))


        count += 1
        #writer.add_scalar('lr', lr, global_step=count)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=count)
        writer = add_summery(writer, 'train', loss.data, acc1, acc5, count)

        if i % train_cfg['print_que'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'lr: [{3}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), optimizer.param_groups[0]['lr'], loss=losses, top1=top1, top5=top5))
        """if i % val_interval == 0:
            val_top1, val_top3, val_top5, val_loss = validate(val_set, model, criterion)
            writer = add_summery(writer, 'val', val_loss, val_top1, val_top5, count)"""

    return count


# verification model
def validate(val_loader, model, criterion, if_print=False):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top3 = AverageMeter()
    model.eval()
    for i, (input, target) in enumerate(val_loader):
        # target = target.cuda(async=True)
        if train_cfg['cuda']:
            input_var = input.cuda()
            target_var = target.cuda()
        #input_var = torch.autograd.Variable(input_var)
        #target_var = torch.autograd.Variable(target_var)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        acc1, acc3, acc5 = accu(output, target_var, topk=(1, 3, 5))

        losses.update(loss.data, input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        top3.update(acc3[0], input.size(0))

        if if_print == True:
            print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@3 {top3.val:.3f} ({top3.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), loss=losses,
                top1=top1, top3=top3, top5=top5))

    print(' * Acc@1 {top1.avg:.3f} Acc@3 {top3.avg:.3f} Acc@5 {top5.avg:.3f} Loss {loss.avg:.3f}'
          .format(top1=top1, top3=top3, top5=top5, loss=losses))

    return top1.avg, top3.avg, top5.avg, losses.avg


if __name__ == '__main__':
    # load data
    train_set = get_data_sets(train_cfg['data_set_dir'], 'train')
    val_set = get_data_sets(train_cfg['data_set_dir'], 'val')
    # load model
    if model_cfg['if_pre_train']:
        model = load_model(model_cfg['pre_trained_path'], data_classes=model_cfg['data_classes'])
    else:
        model = PeleeNet(num_classes=model_cfg['model_classes'])
    if train_cfg['cuda']:
        model = model.cuda()
    #print(model.state_dict())

    # init the summery
    writer = init_summery('runs', 'train_val')
    #global optimizer_conv
    
    optimizer_conv = optim.Adam(model.parameters(), lr=train_cfg['init_lr'], weight_decay=0.0001)
    #schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer_conv, factor=0.1, patience=2, verbose=False, threshold=0.1, cooldown=0, min_lr=train_cfg['min_lr'])
    schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer_conv, factor=0.9, patience=2, threshold=0.01, min_lr=train_cfg['min_lr'])
    #schedule = optim.lr_scheduler.MultiStepLR(optimizer_conv, milestones=[20,80], gamma = 0.9)
    criteria1 = LabelSmoothSoftmaxCEV1(lb_smooth=0.1)
    #criteria1 = nn.CrossEntropyLoss()
    count = 0
    best_acc = 0.0


    num_epoch = 120

    for epoch in range(num_epoch):
        if epoch < 2:
            count = train_warmup(train_set, val_set, model, optimizer_conv, criteria1, epoch, writer, count)
        else:
            count, train_loss = train(train_set, val_set, model, optimizer_conv, criteria1, writer, count, epoch)
            print("train_loss : {0}, lr : {1}".format(train_loss, optimizer_conv.param_groups[0]['lr']))
            schedule.step(train_loss)
            #schedule.step()

        val_top1, val_top3, val_top5, val_loss = validate(val_set, model, criteria1)
        writer = add_summery(writer, 'val', val_loss, val_top1, val_top5, count)
        if_best_model = (val_top1 > best_acc)
        best_acc = max(val_top1, best_acc)

        filepath = 'weights/epoch_' + str(epoch) + 'checkpoint.pth.tar'
        save_step({
            'epoch': epoch + 1,
            'arch': 'peleenet',
            'state_dict': model.state_dict(),
            'acc': val_top1,
        }, if_best_model, 'test', filename=filepath)






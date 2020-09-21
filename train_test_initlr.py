import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_data_sets
from util.config import train_cfg, model_cfg
from peleenet import load_model, PeleeNet, save_step
from util.util import init_summery, AverageMeter, adjust_lr_test, accu, add_summery


def train(train_loader, model, criterion, optimizer, epoch, writer, count):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    phase = 'train'
    model.train()
    for i, (input, target) in enumerate(train_loader):
        ### Adjust learning rate
        lr = adjust_lr_test(optimizer, count, train_cfg['test_init_lr'])
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
        if lr < 10:
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=count)
            writer = add_summery(writer, 'train', loss.data, acc1, acc5, count)

        if i % train_cfg['print_que'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'lr: [{3}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), optimizer.param_groups[0]['lr'], loss=losses, top1=top1, top5=top5))

    return count, losses.avg


if __name__ == '__main__':
    # load data
    train_set = get_data_sets(train_cfg['data_set_dir'], 'train')

    # load model
    if model_cfg['if_pre_train']:
        model = load_model(model_cfg['pre_trained_path'], model_classes=model_cfg['imagenet_classes'],
                           data_classes=model_cfg['data_classes'])
    else:
        model = PeleeNet(num_classes=model_cfg['data_classes'])
    if train_cfg['cuda']:
        model = model.cuda()
    #print(model.state_dict())

    # init the summery
    writer = init_summery('runs', 'init_lr')
    
    optimizer_conv = optim.Adam(model.parameters(), lr=train_cfg['init_lr'])
    schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer_conv, factor=0.1, patience=2, verbose=False,
                                                threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    criterion = nn.CrossEntropyLoss()
    count = 0

    for epoch in range(train_cfg['num_epochs']):
        count, train_loss = train(train_set, model, criterion, optimizer_conv, epoch, writer, count)
        #schedule.step(train_loss)

        #if_best_model = False
        #save_step({
        #    'epoch': epoch + 1,
        #    'arch': 'peleenet',
        #    'state_dict': model.state_dict(),
        # }, if_best_model, 'test')






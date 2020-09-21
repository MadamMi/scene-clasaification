import os
import shutil
import math
from tensorboardX import SummaryWriter
from util.config import train_cfg
import torch.nn as nn
import torch


def init_summery(summery_path, filename):
    if os.path.exists(os.path.join(summery_path, filename)):
        shutil.rmtree(os.path.join(summery_path, filename))

    if not os.path.exists(summery_path):
        os.mkdir(summery_path)

    writer = SummaryWriter(summery_path + '/' + filename)

    return writer


def add_summery(writer, tag, loss, acc1, acc5, global_step=None):
    writer.add_scalar(tag + '_loss', loss.data, global_step=global_step)
    writer.add_scalar(tag + '_acc_top1', acc1, global_step=global_step)
    writer.add_scalar(tag + '_acc_top5', acc5, global_step=global_step)
    return writer


# compute accuracy
def accu(output, target, topk=(1, 5)):
    maxk = max(topk)
    batch_size = target.size(0)
    # batch_size = 1

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.cuda()
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred).long())

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# adjust lr
def adjust_lr(optimizer, epoch, count, init_lr, iterations_per_epoch, method='warmup'):
    global lr
    if method == 'warmup':
        warmup = train_cfg['warmup']
    else:
        warmup = 0
    if count < warmup:
        lr = init_lr * (count + 1.0) / warmup
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    elif method == 'cos':
        T_total = train_cfg['num_epochs'] * iterations_per_epoch
        T_cur = (epoch % train_cfg['num_epochs']) * iterations_per_epoch + count
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * T_cur / T_total))
        if epoch > 120:
            lr = train_cfg['min_lr']

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def adjust_lr_test(optimizer, count, init_lr):
    lr = init_lr * (5 ** count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class LabelSmoothSoftmaxCEV1(nn.Module):
    '''
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)
        # self.log_softmax = nn.Softmax(dim=1)

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # overcome ignored label
        # use fp32 to avoid nan
        logits = logits.float()
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()

            ignore = label == self.lb_ignore
            n_valid = (ignore == 0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            label = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * label, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid.float()

        if self.reduction == 'sum':
            loss = loss.sum()

        return loss



class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num=138, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha*(torch.pow((1 - probs), self.gamma))*log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss

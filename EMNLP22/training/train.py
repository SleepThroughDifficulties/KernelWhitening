import os
import random
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.autograd import Variable
from utilis.matrix import accuracy
from utilis.meters import AverageMeter, ProgressMeter

from training.reweighting import weight_learner
from transformers import get_linear_schedule_with_warmup

def train(train_loader, model, criterion, optimizer, epoch, args, tensor_writer=None):
    '''
        the size of each feature is 512, os we need a tensor of 1024 * 512
        replace the last one every time
        and a weight with size of 1024,
        replace the last one every time


        args:
            tensor_writer: SummaryWriter(log_dir)
    '''

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    training_steps = (len(train_loader) - 1 / args.epochs + 1) * args.epochs
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps = 0.1 * training_steps,
        num_training_steps=training_steps
        )


    model.train()

    end = time.time()
    for i, (input_ids, attention_masks, segment_ids, target) in enumerate(train_loader):
        data_time.update(time.time() - end)


        input_ids = input_ids.cuda(args.gpu, non_blocking=True)
        attention_masks = attention_masks.cuda(args.gpu, non_blocking=True)
        segment_ids = segment_ids.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        output, cfeatures = model(input_ids, attention_masks, segment_ids)
        #print('cfeatures size is {}'.format(cfeatures.size()))   32*768
        pre_features = model.pre_features
        pre_weight1 = model.pre_weight1
        #print('pre_features size is {}'.format(pre_features.size()))   32*768
        

        if epoch >= args.epochp:
            weight1, pre_features, pre_weight1 = weight_learner(cfeatures, pre_features, pre_weight1, args, epoch, i)

        else:
            weight1 = Variable(torch.ones(cfeatures.size()[0], 1).cuda())
        # weight1 = Variable(torch.ones(cfeatures.size()[0], 1).cuda())

        model.pre_features.data.copy_(pre_features)
        model.pre_weight1.data.copy_(pre_weight1)


        loss = criterion(output, target).view(1, -1).mm(weight1).view(1)
        
        # if args.alignment:
        #     loss = loss + criterion(output2, target).view(1, -1).mm(1 - weight1).view(1)
        acc1, acc5 = accuracy(output, target, args=args, topk=(1, 1))
        losses.update(loss.item(), input_ids.size(0))
        top1.update(acc1[0], input_ids.size(0))
        top5.update(acc5[0], input_ids.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        method_name = args.log_path.split('/')[-2]
        if i % args.print_freq == 0:
            progress.display(i, method_name)
            progress.write_log(i, args.log_path)
            print("weight1:")
            print(weight1)
    tensor_writer.add_scalar('loss/train', losses.avg, epoch)
    tensor_writer.add_scalar('ACC@1/train', top1.avg, epoch)
    tensor_writer.add_scalar('ACC@5/train', top5.avg, epoch)

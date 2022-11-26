import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.backends.cudnn as cudnn

import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utilis.datasets import datasets
from utilis.datasets import Collate_function
from torch.utils.tensorboard import SummaryWriter

from ops.config import parser
from training.schedule import lr_setter
from training.train import train
from training.validate import validate
from utilis.meters import AverageMeter
from utilis.saving import save_checkpoint

from transformers import AutoConfig, AutoTokenizer, AdamW
from models.model_slabt import AutoModelForSlabt



def main():
    args = parser.parse_args()

    best_acc1 = 0

    args.log_path = os.path.join(args.log_base, args.dataset, "log.txt")

    if not os.path.exists(os.path.dirname(args.log_path)):
        os.makedirs(os.path.dirname(args.log_path))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True


    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.pretrained:
        config = AutoConfig.from_pretrained('bert-base-uncased')
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModelForSlabt('bert-base-uncased', config=config, args=args, cls_num=args.classes_num)
    else:
        pass


    nn.init.xavier_uniform_(model.fc2.weight, .1)
    nn.init.constant_(model.fc2.bias, 0.)
    # nn.init.xavier_uniform_(model.fc3.weight, .1)
    # nn.init.constant_(model.fc3.bias, 0.)
    # model.fc2 = model.fc1
    # model.fc3 = model.fc1

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        pass


    # define loss function (criterion) and optimizer
    if args.dataset == 'MNLI':
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    elif args.dataset == 'FEVER':
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    elif args.dataset == 'QQP':
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    
    criterion_train = nn.CrossEntropyLoss(reduce=False).cuda(args.gpu)

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)


    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    print("Whole args:\n")
    print(args)

    # optionally resume from a checkpoint

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.dataset == 'MNLI':
        # /root/slabt/dataset/ + MNLI/ +
        traindir = os.path.join(args.data, args.dataset, 'train.tsv')
        valdir = os.path.join(args.data, args.dataset, 'dev_matched.tsv')
        testdir = os.path.join(args.data, args.dataset, 'dev_mismatched.tsv')

        # /root/slabt/dataset/ + HANS/ +
        test2dir = os.path.join(args.data, args.sub_dataset, 'hansdev.tsv')
        
    elif args.dataset == 'FEVER':
        # /root/slabt/dataset/ + FEVER/ +
        traindir = os.path.join(args.data, args.dataset, 'train.tsv')
        valdir = os.path.join(args.data, args.dataset, 'dev.tsv')
        testdir = os.path.join(args.data, args.dataset, 'symmv1.tsv')
        test2dir = os.path.join(args.data, args.dataset, 'symmv2.tsv')

    elif args.dataset == 'QQP':
        # /root/slabt/dataset/ + QQP/ +
        traindir = os.path.join(args.data, args.dataset, 'train.tsv')
        valdir = os.path.join(args.data, args.dataset, 'dev.tsv')
        testdir = os.path.join(args.data, args.dataset, 'test.tsv')
        test2dir = os.path.join(args.data, args.dataset, 'paws.tsv')

    log_dir = os.path.dirname(args.log_path)
    print('tensorboard dir {}'.format(log_dir))
    tensor_writer = SummaryWriter(log_dir)

    if args.evaluate:

        val_dataset = datasets(valdir, tokenizer, args.dataset, args)
        test_dataset = datasets(testdir, tokenizer, args.dataset, args)
        test2_dataset = datasets(test2dir, tokenizer, args.sub_dataset, args)

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True,
                                                 collate_fn=Collate_function())

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.workers, pin_memory=True,
                                                  collate_fn=Collate_function())
        test2_loader = torch.utils.data.DataLoader(test2_dataset, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=args.workers, pin_memory=True,
                                                   collate_fn=Collate_function())

        if args.dataset == 'MNLI':
            validate(val_loader, model, criterion, 0, False, args, tensor_writer, datasetname='MNLI')
            validate(test_loader, model, criterion, 0, True, args, tensor_writer, datasetname='MNLI')
            validate(test2_loader, model, criterion, 0, True, args, tensor_writer, datasetname='HANS')
        elif args.dataset == 'FEVER':
            validate(val_loader, model, criterion, 0, False, args, tensor_writer, datasetname='FEVER')
            validate(test_loader, model, criterion, 0, True, args, tensor_writer, datasetname='SYMMv1')
            validate(test2_loader, model, criterion, 0, True, args, tensor_writer, datasetname='SYMMv2')
        elif args.dataset == 'QQP':
            validate(val_loader, model, criterion, 0, False, args, tensor_writer, datasetname='QQP_dev')
            validate(test_loader, model, criterion, 0, True, args, tensor_writer, datasetname='QQP_test')
            validate(test2_loader, model, criterion, 0, True, args, tensor_writer, datasetname='PAWS')

        return

    train_dataset = datasets(traindir, tokenizer, args.dataset, args)
    val_dataset = datasets(valdir, tokenizer, args.dataset, args)
    test_dataset = datasets(testdir, tokenizer, args.dataset, args)
    test2_dataset = datasets(test2dir, tokenizer, args.sub_dataset, args)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, collate_fn=Collate_function())

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True, collate_fn=Collate_function())

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True, collate_fn=Collate_function())

    test2_loader = torch.utils.data.DataLoader(test2_dataset, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.workers, pin_memory=True, collate_fn=Collate_function())

    print('\n*****train_dataset len is : {}'.format(len(train_dataset)))
    print('\n*****val_dataset len is : {}'.format(len(val_dataset)))
    print('\n*****test_dataset len is : {}'.format(len(test_dataset)))

    # begin to train
    for epoch in range(args.start_epoch, args.epochs):
        # lr_setter(optimizer, epoch, args)

        train(train_loader, model, criterion_train, optimizer, epoch, args, tensor_writer)

        if args.dataset == 'MNLI':
            val_acc1 = validate(val_loader, model, criterion, epoch, False, args, tensor_writer, datasetname='MNLI')
            acc1 = validate(test_loader, model, criterion, epoch, True, args, tensor_writer, datasetname='MNLI')
            acc2 = validate(test2_loader, model, criterion, epoch, True, args, tensor_writer, datasetname='HANS')
        elif args.dataset == 'FEVER':
            val_acc1 = validate(val_loader, model, criterion, epoch, False, args, tensor_writer, datasetname='FEVER')
            acc2 = validate(test_loader, model, criterion, epoch, True, args, tensor_writer, datasetname='SYMMv1')
            acc1 = validate(test2_loader, model, criterion, epoch, True, args, tensor_writer, datasetname='SYMMv2')
        elif args.dataset == 'QQP':
            val_acc1 = validate(val_loader, model, criterion, epoch, False, args, tensor_writer, datasetname='QQP_dev')
            acc2 = validate(test_loader, model, criterion, epoch, True, args, tensor_writer, datasetname='QQP_test')
            acc1 = validate(test2_loader, model, criterion, epoch, True, args, tensor_writer, datasetname='PAWS')


        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        print('Saving...')
        save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.log_path, epoch)

    # harddir = os.path.join(args.data, 'MNLI_hard', 'dev_mismatched.tsv')
    # hard_dataset = datasets(test2dir, tokenizer, args)

    # hard_loader = torch.utils.data.DataLoader(hard_dataset, batch_size=args.batch_size, shuffle=False,
    #                                          num_workers=args.workers, pin_memory=True,
    #                                          collate_fn=Collate_function())
    # validate(hard_loader, model, criterion, 6, True, args, tensor_writer, datasetname='MNLI_hard')

if __name__ == '__main__':
    main()

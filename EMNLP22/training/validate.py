import time
import pandas as pd
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utilis.matrix import accuracy
from utilis.meters import AverageMeter, ProgressMeter
import os


def validate(val_loader, model, criterion, epoch=0, test=True, args=None, tensor_writer=None, datasetname=None):
    if test:
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5],
            prefix='Test: ')
    else:
        batch_time = AverageMeter('val Time', ':6.3f')
        losses = AverageMeter('val Loss', ':.4e')
        top1 = AverageMeter('Val Acc@1', ':6.2f')
        top5 = AverageMeter('Val Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5],
            prefix='Val: ')

    model.eval()
    print('******************datasetname is {}******************'.format(datasetname))

    log_path = os.path.join(r'/workspace/nyslearning/MyResults', args.predout)
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    with torch.no_grad():
        end = time.time()
        a = True
        for i, (input_ids, attention_masks, segment_ids, target) in enumerate(val_loader):

            input_ids = input_ids.cuda(args.gpu, non_blocking=True)
            attention_masks = attention_masks.cuda(args.gpu, non_blocking=True)
            segment_ids = segment_ids.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            output, cfeatures = model(input_ids, attention_masks, segment_ids)
            
            if args.DLSS:
                threshold = 0.8
                if datasetname == 'HANS':
                    output[:, 0] = 1-output[:, 1]*threshold
                    output[:, 1] = output[:, 1]*threshold
                elif datasetname in ['SYMMv1', 'SYMMv2']:
                    output[:, 0] = output[:, 0]*threshold
                    output[:, 1] = 1 - output[:, 0] * threshold
                elif datasetname in ['QQP_dev', 'QQP_test', 'PAWS']:
                    pass

            pred = torch.cat((output, target.view(-1,1)), dim=1).cpu().detach()
            if a:
                predy = pred
                a = False
            else:
                predy = torch.cat((predy, pred), dim=0)

            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 1), args=args, datasetname=datasetname)
            losses.update(loss.item(), input_ids.size(0))
            top1.update(acc1[0], input_ids.size(0))
            top5.update(acc5[0], input_ids.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                method_name = args.log_path.split('/')[-2]
                progress.display(i, method_name)
                progress.write_log(i, args.log_path)

        if datasetname in ['QQP_dev', 'QQP_test', 'PAWS']: 
            sample = pd.DataFrame(predy.numpy(), columns=['0', '1', 'label'])
        else:
            sample = pd.DataFrame(predy.numpy(), columns=['0', '1', '2', 'label'])
        sample.to_csv(os.path.join(log_path, datasetname) + str(epoch) + '.csv', sep=',', index = False)


        print(' * Acc@1 {top1.avg:.3f} Acc@1 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        with open(args.log_path, 'a') as f1:
            f1.writelines(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                          .format(top1=top1, top5=top5))
        if test:
            tensor_writer.add_scalar('loss/test', loss.item(), epoch)
            tensor_writer.add_scalar('ACC@1/test', top1.avg, epoch)
            tensor_writer.add_scalar('ACC@5/test', top5.avg, epoch)
        else:
            tensor_writer.add_scalar('loss/val', loss.item(), epoch)
            tensor_writer.add_scalar('ACC@1/val', top1.avg, epoch)
            tensor_writer.add_scalar('ACC@5/val', top5.avg, epoch)

    return top1.avg

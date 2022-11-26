import torch
import torch.nn as nn

def accuracy(output, target, topk=(1,), args=None, datasetname=None):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        output = softmax(output, args)
        # if datasetname in ['HANS', 'SYMMv1', 'SYMMv2']:
        #     output[:, 0] = output[:, 0] + output[:, 2]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        
        if datasetname != None:
            if datasetname == 'HANS':
                tmp_zero = torch.zeros_like(pred).cuda(args.gpu, non_blocking=True)
                pred = torch.where(pred == 2, tmp_zero, pred).cuda(args.gpu, non_blocking=True)
        # print(pred)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def Multiaccuracy(output, target, topk=(1,), args=None, datasetname=None):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        # 319加了这里
        if datasetname == 'HANS':
            output[:, 0] = output[:, 0] + output[:, 2]
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        if datasetname != None:
            if datasetname == 'HANS':
                tmp_zero = torch.zeros_like(pred).cuda(args.gpu, non_blocking=True)
                pred = torch.where(pred == 2, tmp_zero, pred).cuda(args.gpu, non_blocking=True)
        # print(pred)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def softmax(output, args):
    softmax = nn.Softmax(dim=1)
    output = softmax(output).cuda(args.gpu)
    return output

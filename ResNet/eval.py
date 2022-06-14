import math
from tabnanny import check
import time
import torch.cuda
import utils.common as utils
import models
from utils.common import *
from utils.conv_type import *
from data import imagenet

visible_gpus_str = ','.join(str(i) for i in args.gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = visible_gpus_str
args.gpus = [i for i in range(len(args.gpus))]
now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
# Calculate the number of combinations
list1 = list(range(1, args.M + 1))
list2 = list(combinations(list1, args.N))
combinations_num = len(list2)
# The flag for judging whether the model meets the sparsity
satisfy_flg = 0

if args.label_smoothing is None:
    loss_func = nn.CrossEntropyLoss().cuda()
else:
    loss_func = LabelSmoothing(smoothing=args.label_smoothing)

# load training data
print('==> Preparing data..')
data_tmp = imagenet.Data(args)
val_loader = data_tmp.testLoader

def validate(val_loader, model, criterion, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    num_iter = len(val_loader)
    model.eval()
    with torch.no_grad():
        end = time.time()
        i = 0
        for batch_idx, (images, targets) in enumerate(val_loader):
            if args.debug:
                if i > 5:
                    break
                i += 1
            images = images.cuda()
            targets = targets.cuda()

            # compute output
            logits = model(images)
            loss = criterion(logits, targets)

            # measure accuracy and record loss
            pred1, pred5 = utils.accuracy(logits, targets, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                    .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def get_model(args):
    model = models.__dict__[args.arch]().to(device)
    model = model.to(device)
    return model

def main():
    model = get_model(args)
    checkpoint = torch.load(args.pretrain_dir)
    model.load_state_dict(checkpoint["state_dict"])
    model = nn.DataParallel(model, device_ids=args.gpus)
    valid_obj, test_acc_top1, test_acc = validate(val_loader, model, loss_func, args)


if __name__ == '__main__':
    main()

import math
import time
import torch.cuda
import utils.common as utils
import models
from utils.common import *
from utils.conv_type import *
from data import imagenet

if args.debug:
    pass

visible_gpus_str = ','.join(str(i) for i in args.gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = visible_gpus_str
args.gpus = [i for i in range(len(args.gpus))]
checkpoint = utils.checkpoint(args)
now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
logger = utils.get_logger(os.path.join(args.job_dir, 'logger' + now + '.log'))
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
train_loader = data_tmp.trainLoader
val_loader = data_tmp.testLoader


def train(epoch, train_loader, model, criterion, optimizer):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    model.train()
    end = time.time()

    num_iter = len(train_loader)

    print_freq = num_iter // 10
    i = 0

    for batch_idx, (images, targets) in enumerate(train_loader):
        remove_alpha(model)
        if args.debug:
            if i > 5:
                break
            i += 1
        images = images.cuda()
        targets = targets.cuda()
        data_time.update(time.time() - end)

        adjust_learning_rate(optimizer, epoch, batch_idx, num_iter)

        # compute output
        logits = model(images)
        loss = loss_func(logits, targets)

        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)  # accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % print_freq == 0 and batch_idx != 0:
            logger.info(
                'Epoch[{0}]({1}/{2}): '
                'Loss {loss.avg:.4f} '
                'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                    epoch, batch_idx, num_iter, loss=losses,
                    top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


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

        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                    .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def get_model(args):
    model = models.__dict__[args.arch]().to(device)
    model = model.to(device)
    return model


def check_model(model):
    global satisfy_flg
    # Judge whether the model meets the n:m sparsity
    alphas = [((torch.abs(v) > 0).sum(1).float() == 1) for n, v in model.named_parameters() if
              ("alpha" in n)]
    flg = 1
    for i in alphas:
        if i.sum() != len(i):
            flg = 0
            break
    if flg:
        logger.info("The model satisfy the N:M sparsity")
        satisfy_flg = 1
    else:
        logger.info("The model doesn't satisfy the N:M sparsity")

def remove_alpha(model):
    with torch.no_grad():
        for n, alphas in model.module.named_parameters():
            if 'alpha' in n:
                index0 = alphas == 0
                alphas -= index0 * 1000
                value, index = torch.topk(alphas, k=args.k, dim=1, largest=False)
                alphas.scatter_(1, index, 0)


def main():
    start_epoch = 0
    best_acc = 0.0
    best_acc_top1 = 0.0
    model = get_model(args)
    optimizer = get_optimizer(args, model)

    if args.resume == True:
        start_epoch, best_acc_top1 = resume(args, model, optimizer)

    model = nn.DataParallel(model, device_ids=args.gpus)

    for epoch in range(start_epoch, args.num_epochs):
        args.pr_target = 1
        if not satisfy_flg:
            check_model(model)
            if args.k < combinations_num - 1:
                card = 1 + math.ceil((combinations_num-1)*(1-(epoch-args.t_i)/(args.t_f-args.t_i))**3)
                args.k = combinations_num - card
                
            if args.k == combinations_num - 1:
                optimizer.param_groups[1]['weight_decay'] = 0
                optimizer.param_groups[1]['momentum'] = 0
                for n, v in model.module.named_parameters():
                    if "alpha" in n:
                        v.requires_grad = False

        train_obj, train_acc_top1, train_acc = train(epoch, train_loader, model, loss_func, optimizer)
        valid_obj, test_acc_top1, test_acc = validate(val_loader, model, loss_func, args)

        is_best = best_acc_top1 < test_acc_top1
        best_acc_top1 = max(best_acc_top1, test_acc_top1)
        best_acc = max(best_acc, test_acc)

        model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()

        state = {
            'state_dict': model_state_dict,
            'best_acc': best_acc,
            'best_acc_top1': best_acc_top1,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args.k':args.k,
        }
        checkpoint.save_model(state, epoch + 1, is_best)

    logger.info('Best accurary(top5): {:.3f} (top1): {:.3f}'.format(float(best_acc), float(best_acc_top1)))


def resume(args, model, optimizer):
    if os.path.exists(args.job_dir + '/checkpoint/model_last.pt'):
        print(f"=> Loading checkpoint ")

        checkpoint = torch.load(args.job_dir + '/checkpoint/model_last.pt')

        start_epoch = checkpoint["epoch"]

        best_acc_top1 = checkpoint["best_acc_top1"]

        args.k = checkpoint["args.k"]

        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"=> Loaded checkpoint (epoch) {checkpoint['epoch']})")

        return start_epoch, best_acc_top1
    else:
        print(f"=> No checkpoint found at '{args.job_dir}' '/checkpoint/")


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    # Warmup
    if args.lr_policy == 'step':
        factor = epoch // 30
        if epoch >= 80:
            factor = factor + 1
        lr = args.lr * (0.1 ** factor)
    elif args.lr_policy == 'cos':  # cos with warm-up
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * (epoch - 5) / (args.num_epochs - 5)))
    elif args.lr_policy == 'exp':
        step = 1
        decay = 0.96
        lr = args.lr * (decay ** (epoch // step))
    elif args.lr_policy == 'fixed':
        lr = args.lr
    else:
        raise NotImplementedError
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

    if step == 0:
        print('current learning rate:{0}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_optimizer(args, model):
    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        alpha_params = [v for n, v in parameters if ("alpha" in n) and v.requires_grad]
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and ("alpha" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0,
                },
                {
                    "params": alpha_params,
                    "weight_decay": args.weight_decay,
                },
                {"params": rest_params, "weight_decay": args.weight_decay},
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )
    else:
        print("please choose sgd or adam")
        
    return optimizer


if __name__ == '__main__':
    main()

import argparse

parser = argparse.ArgumentParser(description='NM Pruning')

parser.add_argument(
    "--label_smoothing",
    type=float,
    help="Label smoothing to use, default 0.0",
    default=0.0
)

parser.add_argument(
    "--warmup_length",
    default=5,
    type=int,
    help="Number of warmup iterations"
)

parser.add_argument(
    '--gpus',
    type=int,
    nargs='+',
    default=[0],
    help='Select gpu_id to use. default:[0]',
)

parser.add_argument(
    '--data_path',
    type=str,
    default='/media/DATA/ImageNet',
    help='The dictionary where the input is stored. default:',
)

parser.add_argument(
    '--job_dir',
    type=str,
    default='experiments/Imagenet',
    help='The directory where the summaries will be stored. default:./experiments'
)

parser.add_argument(
    '--pretrain_dir',
    type=str,
    default='resnet2-4.pt',
    help='The directory where the pretrain model is stored. default:resnet2-4.pt'
)

parser.add_argument(
    '--resume',
    action='store_true',
    help='Load the model from the specified checkpoint.'
)


## Training
parser.add_argument(
    '--arch',
    type=str,
    default='resnet50',
    help='Architecture of model. default:resnet32_cifar10, resnet 50, resnet18'
)

parser.add_argument(
    '--num_epochs',
    type=int,
    default=120,
    help='The num of epochs to train. default:180'
)

parser.add_argument(
    '--train_batch_size',
    type=int,
    default=256,
    help='Batch size for training. default:256'
)

parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=256,
    help='Batch size for validation. default:256'
)

parser.add_argument(
    '--momentum',
    type=float,
    default=0.9,
    help='Momentum for MomentumOptimizer. default:0.9'
)

parser.add_argument(
    '--lr',
    type=float,
    default=0.1,
    help='Learning rate for train. default:0.1'
)

parser.add_argument(
    "--optimizer",
    help="Which optimizer to use",
    default="sgd"
)

parser.add_argument(
    "--lr_policy",
    default="cos",
    help="Policy for the learning rate."
)

parser.add_argument(
    "--lr_adjust",
    default=30, type=int,
    help="Interval to drop lr"
)
parser.add_argument(
    "--lr_gamma",
    default=0.1,
    type=float,
    help="Multistep multiplier"
)

parser.add_argument(
    '--weight_decay',
    type=float,
    default=1e-4,
    help='The weight decay of loss. default:1e-4'
)

parser.add_argument(
    "--nesterov",
    default=False,
    action="store_true",
    help="Whether or not to use nesterov for SGD",
)


parser.add_argument(
    "--conv_type",
    type=str,
    default='NMConv',
    help="Conv type of conv layer. Default: NMConv"
)

parser.add_argument(
    "--bn_type",
    default='LearnedBatchNorm',
    help="BatchNorm type"
)

parser.add_argument(
    "--init",
    default="kaiming_normal",
    help="Weight initialization modifications"
)

parser.add_argument(
    "--mode",
    default="fan_in",
    help="Weight initialization mode"
)

parser.add_argument(
    "--nonlinearity",
    default="relu",
    help="Nonlinearity used by initialization"
)

parser.add_argument(
    '--debug',
    action='store_true',
    help='input to open debug state'
)


parser.add_argument(
    "--N",
    default=2,
    type=int,
    help="N:M's N"
)
parser.add_argument(
    "--M",
    default=4,
    type=int,
    help="N:M's M"
)

parser.add_argument(
    "--k",
    default=0,
    type=int,
    help="remove k smallest alpha in group"
)
parser.add_argument(
    "--t_i",
    default=0,
    type=int,
    help="begin remove epoch"
)
parser.add_argument(
    "--t_f",
    default=80,
    type=int,
    help="finish remove epoch"
)
parser.add_argument(
    "--scale-fan",
    action="store_true",
    default=False,
    help="scale fan"
)
args = parser.parse_args()

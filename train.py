import argparse
import logging
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
from ignite.engine import Events, _prepare_batch, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import EarlyStopping, ModelCheckpoint

import monai
from monai.data import decollate_batch, DataLoader
from dataset import LungCancerDataset
from monai.handlers import ROCAUC, StatsHandler, TensorBoardStatsHandler, stopping_fn_from_metric
from monai.transforms import Activations, AsDiscrete, Compose, LoadImaged, RandRotate90d, Resized, ScaleIntensityd

# from SegMamba.model_segmamba.segmamba import CLS_Mamba
from models import nnMambaSeg
from models.resnet import generate_model
from models.moblienet3D import MobileNet
from monai.networks.nets import DenseNet121
from models.shufflenet3D import ShuffleNet_g8
from models.AlexNet3D import AlexNet
from models.GoogLeNet3D import GoogLeNet
from models.VGG3D import VGG11
from models.CNN_Mamba.LowTransformer import LowTransformer
from models.MambaIDH import MambaIDH_T
from models.MedViT3D import MedViT_small


parser = argparse.ArgumentParser(description='PCR Progonsis for Lung Cancer in Pytorch')

parser.add_argument('--data_dir', default='./mRSdata', type=str, 
                    help='dataset dir')
parser.add_argument('--model_name', default='PCM', type=str, 
                    help='select model to train')
# ResNet, nnMamba, DenseNet121, AlexNet, MobileNet, ShuffleNet, PCM
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--device', default='cuda', type=str,
                    help='device to use for training / testing')

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--validation_every_n_epochs', default=1, type=int,
                    metavar='N', help='print frequency (default: 1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')


parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=20)


def params(net):
    param= sum([p.numel() for p in net.parameters()])
    print('='*18)
    print("params:", param / 1e6, "M")
    print('='*18)

def main(args):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

   
    data_path = args.data_dir
    dataset = LungCancerDataset(fold=0, data_dir=data_path)

    # create a validation data loader
    val_ds = dataset.get_val_dataset()
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=args.workers, pin_memory=torch.cuda.is_available())  # batch_size=args.batch_size,
    # create a training data loader
    train_ds = dataset.get_train_dataset()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, 
                              shuffle=True, num_workers=args.workers, pin_memory=torch.cuda.is_available())

    device = torch.device(args.device)
    # net = CLS_Mamba(num_classes=2, in_chans=1).to(device)
    if args.model_name == 'ResNet':
        net = generate_model(18).to(device)
    elif args.model_name == 'nnMamba':
        net = nnMambaSeg().to(device)
    elif args.model_name == 'DenseNet121':
        net = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
    elif args.model_name == 'MobileNet':
        net = MobileNet(num_classes=2).to(device)
    elif args.model_name == 'ShuffleNet':
        net = ShuffleNet_g8(num_classes=2).to(device)
    elif args.model_name == 'AlexNet':
        net = AlexNet(num_classes=2).to(device)
    elif args.model_name == 'GoogLeNet':
        net = GoogLeNet(num_classes=2, aux_logits=False).to(device)
    elif args.model_name == 'MedViT':
        net = MedViT_small().to(device)
    elif args.model_name == 'VGG11': 
        net = VGG11().to(device) 
    elif args.model_name == 'PCM':
        kernel_size = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        stride = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        padding = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
        size = [32, 16, 8, 4]
        # embed_dims = [32, 64, 128, 256, 320, 512, 1024]
        embed_dims = [16, 32, 64,128, 256, 320, 512]
        net = LowTransformer(kernel_size=kernel_size, stride=stride, padding=padding, embed_dims=embed_dims, size=size).to(device)
        
        
    elif args.model_name == 'MambaIDH':
        net = MambaIDH_T(2,1).to(device)
      
    params(net)
    loss = torch.nn.CrossEntropyLoss()
    lr = args.lr
    opt = torch.optim.Adam(net.parameters(), lr)

    # Ignite trainer expects batch=(img, label) and returns output=loss at every iteration,
    # user can add output_transform to return other values, like: y_pred, y, etc.
    def prepare_batch(batch, device=None, non_blocking=False):
        return _prepare_batch((batch["img"], batch["label"]), device, non_blocking)
    
    metric_name = "AUC"
    val_metrics = {metric_name: ROCAUC()}

    post_label = Compose([AsDiscrete(to_onehot=2)])
    post_pred = Compose([Activations(softmax=True)])
    

    trainer = create_supervised_trainer(
        net, 
        opt, 
        loss, 
        device, 
        False, 
        prepare_batch=prepare_batch)

    evaluator = create_supervised_evaluator(
        net,
        val_metrics,
        device,
        True,
        prepare_batch=prepare_batch,
        output_transform=lambda x, y, y_pred: (
            [post_pred(i) for i in decollate_batch(y_pred)],
            [post_label(i) for i in decollate_batch(y, detach=False)],
        ),
    )

    @trainer.on(Events.EPOCH_COMPLETED(every=args.validation_every_n_epochs))
    def run_validation(engine):
        evaluator.run(val_loader)

    # adding checkpoint handler to save models (network params and optimizer stats) during training
    checkpoint_handler = ModelCheckpoint("./results/checkpoints/", args.model_name, n_saved=10, require_empty=False)
    trainer.add_event_handler(
        event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler, to_save={"net": net, "opt": opt}
    )

    # StatsHandler prints loss at every iteration and print metrics at every epoch,
    # we don't set metrics for trainer here, so just print loss, user can also customize print functions
    # and can use output_transform to convert engine.state.output if it's not loss value
    train_stats_handler = StatsHandler(name="trainer", output_transform=lambda x: x)
    train_stats_handler.attach(trainer)

    
    #定义每次实验的Tensorboard的数据记录路径
    log_dir = "./results/tensorboard/" + args.model_name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # TensorBoardStatsHandler plots loss at every iteration and plots metrics at every epoch, same as StatsHandler
    train_tensorboard_stats_handler = TensorBoardStatsHandler(log_dir=log_dir, output_transform=lambda x: x)   # tag_name = type(net).__name__
    train_tensorboard_stats_handler.attach(trainer)

    # add stats event handler to print validation stats via evaluator
    val_stats_handler = StatsHandler(
        name="evaluator",
        output_transform=lambda x: None,  # no need to print loss value, so disable per iteration output
        global_epoch_transform=lambda x: trainer.state.epoch,
    )  # fetch global epoch number from trainer
    val_stats_handler.attach(evaluator)

    # add handler to record metrics to TensorBoard at every epoch
    val_tensorboard_stats_handler = TensorBoardStatsHandler(
        log_dir=log_dir,
        output_transform=lambda x: None,
        global_epoch_transform=lambda x: trainer.state.epoch,
    )  # fetch global epoch number from trainer
    val_tensorboard_stats_handler.attach(evaluator)

    # add early stopping handler to evaluator
    early_stopper = EarlyStopping(patience=5, score_function=stopping_fn_from_metric(metric_name), trainer=trainer)
    evaluator.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=early_stopper)
    
    state = trainer.run(train_loader, args.epochs)
    print(state)


if __name__ == "__main__":
    # 设置运行的显卡
    args = parser.parse_args()
    main(args)
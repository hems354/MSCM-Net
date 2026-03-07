# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys

import numpy as np
import torch
from ignite.engine import _prepare_batch, create_supervised_evaluator
from ignite.metrics import Accuracy, Precision, Recall, ROC_AUC, ConfusionMatrix,ClassificationReport
from ignite.metrics import Metric

from dataset import LungCancerDataset

import monai
from monai.data import DataLoader
# from SegMamba.model_segmamba.segmamba import CLS_Mamba
from monai.handlers import CheckpointLoader, ClassificationSaver, StatsHandler
from monai.transforms import Compose, LoadImaged, Resized, ScaleIntensityd
from models import nnMambaSeg
from models.resnet import generate_model
from monai.data import decollate_batch, DataLoader
from dataset import LungCancerDataset
from monai.handlers import ROCAUC, StatsHandler, TensorBoardStatsHandler, stopping_fn_from_metric
from monai.transforms import Activations, AsDiscrete, Compose, LoadImaged, RandRotate90d, Resized, ScaleIntensityd
from models import nnMambaSeg
from models.resnet import generate_model
from models.moblienet3D import MobileNet
from monai.networks.nets import DenseNet121
from models.shufflenet3D import ShuffleNet_g8
from models.AlexNet3D import AlexNet
from models.GoogLeNet3D import GoogLeNet
from models.MedViT3D import MedViT_small
from models.CNN_Mamba.LowTransformer import LowTransformer
import io
from contextlib import redirect_stdout
import re

import logging
import io


parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')

parser.add_argument('--data_dir', default='./mRSdata', type=str, 
                    help='dataset dir')
parser.add_argument('--model_name', default='PCM', type=str, 
                    help='select model to test')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--device', default='cuda', type=str,
                    help='device to use for training / testing')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 128)')


class Specificity(Metric):
    def __init__(self, output_transform=lambda x: x, device=None):
        self._output_transform = output_transform
        self._device = device or torch.device("cpu")
        self._confusion_matrix = None
        super(Specificity, self).__init__()

    def reset(self):
        self._confusion_matrix = None

    def update(self, output):
        y_pred, y = self._output_transform(output)
        cm = ConfusionMatrix(num_classes=2, output_transform=lambda x: (y_pred, y))
        self._confusion_matrix = cm.update(output).compute().cpu().numpy()

    def compute(self):
        if self._confusion_matrix is None:
            raise NotComputableError("Specificity must have at least one example before it can be computed.")
        tn, fp, fn, tp = self._confusion_matrix.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return specificity

def main(args):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    data_path = args.data_dir
    dataset = LungCancerDataset(fold=0, data_dir=data_path)

    # create a validation data loader
    val_ds = dataset.get_val_dataset()
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=args.workers, pin_memory=torch.cuda.is_available())


    device = torch.device(args.device)
    # net = CLS_Mamba(num_classes=2, in_chans=1).to(device)
    if args.model_name == 'ResNet':
        net = generate_model(18).to(device)
    elif args.model_name == 'nnMamba':
        net = nnMambaSeg().to(device)
    elif args.model_name == 'DenseNet121':
        net = DenseNet121().to(device)
    elif args.model_name == 'MobileNet':
        net = MobileNet().to(device)
    elif args.model_name == 'ShuffleNet':
        net = ShuffleNet_g8().to(device)
    elif args.model_name == 'AlexNet':
        net = AlexNet(num_classes=2).to(device)
    elif args.model_name == 'GoogLeNet':
        net = GoogLeNet(num_classes=2, aux_logits=False).to(device)
    elif args.model_name == 'PCM':
        kernel_size = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        stride = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        padding = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
        size = [32, 16, 8, 4]
        embed_dims = [32, 64, 128, 256, 320, 512, 1024]
        net = LowTransformer(kernel_size=kernel_size, stride=stride, padding=padding, embed_dims=embed_dims, size=size).to(device)
    elif args.model_name == 'MedViT':
        net = MedViT_small().to(device)
    

    def prepare_batch(batch, device=None, non_blocking=False):
        return _prepare_batch((batch["img"], batch["label"]), device, non_blocking)

    def thresholded_output_transform(output):
            y_pred, y = output
            y_pred = torch.stack([torch.argmax(tensor) for tensor in y_pred])
            y = torch.stack([torch.argmax(tensor) for tensor in y])
            return y_pred, y

    with torch.no_grad():
        # add evaluation metric to the evaluator engine
        val_metrics = {
            "Accuracy": Accuracy(output_transform=thresholded_output_transform),
            "Precision": Precision(output_transform=thresholded_output_transform),
            "Recall": Recall(output_transform=thresholded_output_transform),
            "ROC_AUC": ROC_AUC(),
            # "ConfusionMatrix": ConfusionMatrix(num_classes=2),
            # "SENS": Recall(output_transform=thresholded_output_transform)  # Sensitivity 和 Recall 是相同的
        }
        # Ignite evaluator expects batch=(img, label) and returns output=(y_pred, y) at every iteration,
        # user can add output_transform to return other values
        # evaluator = create_supervised_evaluator(net, val_metrics, device, True, prepare_batch=prepare_batch)
        post_label = Compose([AsDiscrete(to_onehot=2)])
        post_pred = Compose([Activations(softmax=True)])
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

        # add stats event handler to print validation stats via evaluator
        val_stats_handler = StatsHandler(
            name="evaluator",
            output_transform=lambda x: None,  # no need to print loss value, so disable per iteration output
        )
        val_stats_handler.attach(evaluator)

        # for the array data format, assume the 3rd item of batch data is the meta_data
        prediction_saver = ClassificationSaver(
            output_dir="tempdir",
            name="evaluator",
            batch_transform=lambda batch: batch["img"].meta,
            output_transform=lambda output: torch.stack([torch.argmax(tensor) for tensor in output[0]])  #
        )
        prediction_saver.attach(evaluator)

        # the model was trained by "densenet_training_dict" example
        CheckpointLoader(load_path="./results/checkpoints/PCM_checkpoint_1770.pt", load_dict={"net": net}).attach(evaluator)

        evaluator.run(val_loader)
        
 

        

if __name__ == "__main__":
    # 我这里的Precision是所有真正为PCR的样本中，预测结果为PCR的比例
    # 举例 测试集一共22个样本，指标依次为Metrics -- Accuracy: 0.8182 Precision: 0.8571 ROC_AUC: 0.8376 Recall: 0.6667 
    # 混淆矩阵为 [[12  1] 
    #           [ 3  6]]
    # Precision = TP/(TP+FP) = 6/(6+1) = 0.8571
    # Recall = TP/(TP+FN) = 6/(6+3) = 0.6667
    # Accuracy = (TP+TN) / (TP+TN+FP+FN) = (12+6)/(12+6+1+3) = 0.8182
    
    
    args = parser.parse_args()
    main(args)
    # cal_specificity()
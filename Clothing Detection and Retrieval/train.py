import datetime
import os
import time

import torch
import torch.utils.data
from torchvision import transforms
from torchvision.models.detection.mask_rcnn import (MaskRCNN,
                                                    resnet_fpn_backbone)

import utils as utils
from data_aug import AffineTransform, HorizontalFlip
from dataset_helpers import ClothesDataset
from engine import evaluate, train_one_epoch
from group_by_aspect_ratio import (GroupedBatchSampler,
                                   create_aspect_ratio_groups)


def get_model(num_classes, args):
    assert args.trainable_backbone_layers <= 5 and args.trainable_backbone_layers >= 0
    backbone = resnet_fpn_backbone(args.backbone_name, pretrained=True, trainable_layers=args.trainable_backbone_layers)
    return MaskRCNN(backbone, num_classes)

def main(args):
    torch.manual_seed(1)
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    train_transform = transforms.Compose([HorizontalFlip(prob = 0.5),
                                        AffineTransform(degrees = 10, translate=(.05, .05), scale=(0.9, 1.1), shear=[-5, 5, -5, 5])])

    val_transform = transforms.Compose([])

    train_dataset = ClothesDataset(os.path.join(args.data_path, 'train'), transform = train_transform, is_train = True)
    val_dataset = ClothesDataset(os.path.join(args.data_path, 'validation'), transform = val_transform, is_train = False)

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(val_dataset)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=8,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn, pin_memory=True)

    print("Creating model")
    
    model = get_model(len(train_dataset.get_labels()), args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    # 0.005 -> 0,1,2,3 trainable: 2
    # checkpoint = torch.load("models/model_3.pth", map_location='cpu')
    # model_without_ddp.load_state_dict(checkpoint['model'])

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        evaluate(model, val_loader, device=device, data_coco_pkl_path="cache/val_coco_api_v2.pkl")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, train_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        if args.output_dir:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch},
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

        # evaluate after every epoch
        evaluate(model, val_loader, device=device, data_coco_pkl_path="cache/val_coco_api_v2.pkl")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--data_path', default='data/', help='dataset')
    parser.add_argument('--backbone_name', default='resnet18', help='backbone model name')
    parser.add_argument(
        "--trainable-backbone-layers", default=1, type=int, help="number of trainable layers of backbone"
    )
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=0.02, type=float,
                        help='initial learning rate, 0.02 is the default value for training')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[2, 4, 6], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)

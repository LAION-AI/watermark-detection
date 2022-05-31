import argparse
from functools import partial
from io import BytesIO
from random import sample

import braceexpand
import deepspeed
import timm
import torch
import torch.backends.cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch_optimizer
import wandb
import webdataset as wds
from PIL import Image
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import transforms as T
from tqdm.auto import tqdm, trange

from scalable_shampoo import Shampoo

torch.backends.cudnn.benchmark = True


def load_image(jpg):
    im = Image.open(BytesIO(jpg))
    if im.mode != 'RGB':
        im = im.convert('RGB')
    return im


def load_text_watervit(txt):
    return 1 if txt.decode().lower() == 'clear' else 0


def load_text_laion(__key__):
    return 1 if 'clear' in __key__.split() else 0


def get_load_mapping(is_laion):
    d = {
        'jpg': load_image,
    }
    if is_laion:
        d['__url__'] = load_text_laion
    else:
        d['txt'] = load_text_watervit
    return d


def process_text(num):
    return torch.Tensor([num]).to(dtype=torch.int64)


process_image_train = T.Compose([
    T.Resize((384, 384)),
    T.RandomResizedCrop((256, 256)),
    T.RandomRotation((-20, +20)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


process_image_test = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def get_process_map(is_train):
    return {
        'txt': process_text,
        'jpg': process_image_train if is_train else process_image_test,
    }


def get_dataset(path, batch_size, is_laion=False, is_train=False):
    l = wds.PytorchShardList(path, split_by_node=False)
    return wds.WebDataset(l).map_dict(**get_load_mapping(is_laion)).map_dict(**get_process_map(is_train)).to_tuple('jpg', 'txt').batched(batch_size, partial=True)


def test_model(model, criterion, test_dl, device, prefix, batch_size, test_dl_len=None, fp16=False):
    losses = []
    model.eval()

    for i, row in enumerate(tqdm(test_dl, desc=f'{prefix} | Testing', leave=False, total=test_dl_len)):
        im, txt = row
        im, txt = im.to(device), txt.to(device)
        if args.fp16:
            im = im.half()
        txt.squeeze_(1)

        with torch.no_grad():
            pred = model(im)

        loss = criterion(pred, txt)
        losses.append(loss.item())

        log = {
            'test_iter': i,
            'test_loss': loss.item(),
        }

        if i % 5 == 0:
            log['test_examples'] = [
                wandb.Image(img, caption=f'Real: {num_to_sym(real)}\nEstimated: {parse_output(est)}') for img, real, est in sample(list(zip(
                    im, txt, F.softmax(
                        pred, dim=1).detach().cpu().numpy().tolist())
                ), min(batch_size, 4)
                )
            ]

        wandb.log(log)

    model.train()

    avg_loss = sum(losses) / len(losses)

    return avg_loss, len(losses)


def save_to_wandb(model_path):
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)


def save_model(model, optimizer, scheduler, dir, deepspeed, epoch):
    if deepspeed:
        model.save_checkpoint(dir, epoch)
    else:
        path = f'{dir}/model_{epoch}_with_optim.pt'
        obj = {
            'weights': model.state_dict(),
            'opt_state': optimizer.state_dict(),
        }
        if scheduler is not None:
            obj['scheduler_state'] = scheduler.state_dict()

        torch.save(obj, path)
        save_to_wandb(path)


def num_to_sym(num):
    return 'clear' if num == 1 else 'watermark'


def parse_output(output):
    water_sym, clear_sym = output
    return f"{'watermark' if water_sym > clear_sym else 'clear'}\n{water_sym:.3f}%w {clear_sym:.3f}%c"


def get_optimizer(name):
    return {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': optim.SGD,
        'shampoo': Shampoo
    }[name.lower().strip()]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--two-class-classifier',
                        action='store_true', help='Use two-class classifier')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)

    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)

    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--is-laion', action='store_true',
                        help='Use Laion dataset preprocessing')
    parser.add_argument('--train-path', type=str, default='./data/train')
    parser.add_argument('--test-path', type=str, default='./data/test')
    parser.add_argument('--save-path', type=str, default='./save')

    parser.add_argument('--optimizer', type=lambda a: get_optimizer(a),
                        default='shampoo', dest='Optimizer')
    parser.add_argument('--opt-args', type=lambda a: a.split(','), default=[])
    parser.add_argument(
        '--opt-kwargs', type=lambda a: [b.split('=') for b in a.split(',')], default={'momentum': 0.9, })

    parser = deepspeed.add_config_arguments(parser)

    return parser.parse_args()


def be(s):
    return list(braceexpand.braceexpand(s))


if __name__ == '__main__':
    args = get_args()
    args.prefetch_factor = 2 if args.workers == 0 else args.batch_size//args.workers
    if args.deepspeed:
        args.device = f'cuda:{args.local_rank}'
    else:
        args.device = f'cuda:{args.device}' if torch.cuda.is_available(
        ) else 'cpu'

    # './watermarks/water_vit_wds/water-vit-train_00{0..8}.tar'
    # '/mnt/akatta/datasets/water_vit_wds/water-vit-train_00{0..8}.tar'
    #train_path = '/mnt/akatta/datasets/water_vit_wds/water-vit-train_00{0..8}.tar'
    # be('pipe:aws s3 cp s3://laion-watermark/watermark/{00000..00077.tar} -')
    # be('pipe:aws s3 cp s3://laion-watermark/clear/{00000..00162.tar} -')
    # './watermarks/water_vit_wds/water-vit-train_009.tar'
    # '/mnt/akatta/datasets/water_vit_wds/water-vit-train_009.tar'
    #test_dl = '/mnt/akatta/datasets/water_vit_wds/water-vit-train_009.tar'

    #opt_args = []
    # opt_kwargs = {

    # 'weight_decay': 2e-5
    # }
    #gamma = 0.9

    deepspeed_config = {
        'train_batch_size': args.batch_size * args.world_size,
        'scheduler': {
            'type': 'WarmupLR',
            'params': {
                'warmup_min_lr': 0,
                'warmup_max_lr': args.lr,
                'warmup_num_steps': 5000
            }
        },
        'fp16': {
            'enabled': args.fp16,
        },
        'amp': {
            'enabled': args.amp,
            'opt_level': 'O1'
        }
    }

    deepspeed.init_distributed()

    is_root = dist.get_rank() == 0
    if is_root:
        wandb.init(
            project='laion-watermark-detection',
            name='water_vit run',
            config={
                'deepspeed': args.deepspeed,
                'two_class_classifier': args.two_class_classifier,
                'epochs': args.epochs,
                'optimizer': args.Optimizer.__class__.__name__,
                # 'scheduler': 'exponential',
                'workers': args.workers,
                'batch_size': args.batch_size,
                'prefetch_factor': args.prefetch_factor,
                'device': args.device,
                'is_laion': args.is_laion,
                'train_path': args.train_path,
                'test_path': args.test_path,
                'start_lr': args.lr,
                'opt_args': args.opt_args,
                'opt_kwargs': args.opt_kwargs,
                'args': args
                # 'gamma': gamma,
            }
        )
    #dist.barrier()

    model = timm.create_model(
        'efficientnet_b3a', pretrained=True, num_classes=2)
    if args.two_class_classifier:
        model.classifier = nn.Sequential(
            # 1536 is the orginal in_features
            nn.Linear(in_features=1536, out_features=625),
            nn.ReLU(),  # ReLu to be the activation function
            nn.Dropout(p=0.3),
            nn.Linear(in_features=625, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=2),
        )
    model.train().to(args.device)

    criterion = nn.CrossEntropyLoss()
    criterion.to(args.device)

    optimizer = args.Optimizer(model.parameters(), lr=args.lr,
                               *args.opt_args, **args.opt_kwargs)
    # scheduler = None # ExponentialLR(optimizer, 0.85)

    ds = get_dataset(args.train_path, args.batch_size,
                     args.is_laion, is_train=True)
    dl = wds.WebLoader(ds, batch_size=None, num_workers=args.workers,
                       prefetch_factor=max(args.prefetch_factor, 2), pin_memory=True)

    test_ds = get_dataset(args.test_path, args.batch_size,
                          args.is_laion, is_train=False)
    test_dl = wds.WebLoader(test_ds, batch_size=None,
                            num_workers=1, prefetch_factor=128, pin_memory=True)

    dl_len = test_dl_len = None

    if args.deepspeed:
        model, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=args,
            config_params=deepspeed_config
        )

    save = partial(save_model, model, optimizer, scheduler,
                   args.save_path, args.deepspeed)

    if is_root:
        average_loss_test, test_dl_len = test_model(
            model, criterion, test_dl, args.device, 'Initial', args.batch_size, test_dl_len, fp16=args.fp16)
        tqdm.write(f'Initial | Test Loss: {average_loss_test}')

        wandb.log({
            'avg_test_loss': average_loss_test,
        })
    #dist.barrier()

    for epoch in trange(1, args.epochs + 1):
        prefix = f'EPOCH {epoch}'
        for i, row in enumerate(tqdm(dl, desc=f'{prefix} | Training', leave=False, total=dl_len, disable=not is_root)):

            im, txt = row
            im, txt = im.to(args.device, non_blocking=True), txt.to(
                args.device, non_blocking=True)
            if args.fp16:
                im = im.half()
            txt.squeeze_(1)

            pred = model(im)
            loss = criterion(pred, txt)

            if is_root:
                log = {
                    'epoch': epoch,
                    'iter': i,
                    'loss': loss.item(),
                    'lr': optimizer.param_groups[0]['lr'],
                }
                if i % 25 == 0:
                    log['examples'] = [
                        wandb.Image(img, caption=f'Real: {num_to_sym(real)}\nEstimated: {parse_output(est)}') for img, real, est in sample(list(zip(
                            im, txt, F.softmax(
                                pred, dim=1).detach().cpu().numpy().tolist())
                        ), min(args.batch_size, 4)
                        )
                    ]
                wandb.log(log)

                if i % 50 == 0:
                    averaged = loss.detach().clone()
                    dist.all_reduce(averaged)
                    total_loss = averaged / args.world_size
                    tqdm.write(
                        f'{prefix} | {i} | Local Loss {loss.item()} | World Loss: {total_loss.item()}')

            if args.deepspeed:
                model.backward(loss)
                model.step()
            else:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            dl_len = i
            scheduler.step()

        if is_root:
            save(epoch)

            average_loss_test, test_dl_len = test_model(
                model, criterion, test_dl, args.device, prefix, args.batch_size, test_dl_len, fp16=args.fp16)

            tqdm.write(f'{prefix} | Test Loss: {average_loss_test}')
            wandb.log({
                'avg_test_loss': average_loss_test,
            })
        #dist.barrier()

    if is_root:
        save('final')

        torch.save(model.state_dict(),
                   f'{args.save_path}{args.device}/model_final.pt')

        # Final Test
        average_loss_test = test_model(
            model, criterion, test_dl, args.device, 'Final', args.batch_size, test_dl_len, fp16=args.fp16)
        tqdm.write(f'Final Test Loss: {average_loss_test}')

        wandb.log({
            'final_test_loss': average_loss_test,
        })
        wandb.finish()
    #dist.barrier()

import argparse
import time
from io import BytesIO
from random import sample

import deepspeed
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import wandb
import webdataset as wds
from PIL import Image
from torchvision import transforms as T
from tqdm.auto import tqdm, trange


def load_image(jpg):
    im = Image.open(BytesIO(jpg))
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def load_text(txt):
    return 1 if txt.decode().lower() == 'clear' else 0


load_mapping = {
    'txt': load_text,
    'jpg': load_image,
}


def process_text(num):
    return torch.Tensor([num]).to(dtype=torch.int64)


process_image_eval = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


process_mapping = {
    'txt': process_text,
    'jpg': process_image_eval
}


def filter_dataset_only_watermark(sample):
    return sample['txt'] == 0


def filter_dataset_only_clear(sample):
    return sample['txt'] == 1


def get_dataset(path, batch_size, only_watermark=False, only_clear=False):
    ds = wds.WebDataset(path).map_dict(**load_mapping)
    if only_watermark:
        ds = ds.select(filter_dataset_only_watermark)
    elif only_clear:
        ds = ds.select(filter_dataset_only_clear)
    return ds.map_dict(**process_mapping).to_tuple('jpg', 'txt').batched(batch_size, partial=True)


def get_samples(im, txt, syms, batch_size, log_all_images=False):
    a = list(zip(im, txt, syms))
    if log_all_images:
        return a
    else:
        return sample(a, min(batch_size, 4))


def test_model(model, criterion, test_dl, device, batch_size, prefix, fp16=False, log_all_images=False):
    correct_water = correct_clear = false_water = false_clear = 0
    losses = []
    if model.training:
        model.eval()

    for i, row in enumerate(tqdm(test_dl, desc=f'{prefix} | Evaluating', leave=False)):
        im, txt = row
        im, txt = im.to(device), txt.to(device)
        if fp16:
            im = im.half()
        txt.squeeze_(1)

        with torch.no_grad():
            pred = model(im)

        loss = criterion(pred, txt)
        losses.append(loss.item())

        syms = F.softmax(pred, dim=1).detach().cpu().numpy().tolist()
        for sym, actual in zip(syms, txt.cpu().numpy().tolist()):
            water_sym, clear_sym = sym
            if water_sym > clear_sym:  # is watermark
                if actual == 0:  # actual is watermark
                    correct_water += 1
                else:  # actual is clear
                    false_water += 1
            else:  # is clear
                if actual == 0:  # actual is watermark
                    false_clear += 1
                else:  # actual is clear
                    correct_clear += 1

        log = {
            'eval_iter': i,
            'eval_loss': loss.item(),
            'correct_water': correct_water,
            'correct_clear': correct_clear,
            'false_water': false_water,
            'false_clear': false_clear,
        }

        log['eval_examples'] = [
            wandb.Image(img, caption=f'Real: {num_to_sym(real)}\nEstimated: {parse_output(est)}') for img, real, est in get_samples(im, txt, syms, batch_size, log_all_images)
        ]

        wandb.log(log)

    avg_loss = sum(losses) / len(losses)

    return avg_loss, len(losses)


def num_to_sym(num):
    return 'clear' if num == 1 else 'watermark'


def parse_output(output):
    water_sym, clear_sym = output
    return f"{'watermark' if water_sym > clear_sym else 'clear'}\n{water_sym:.3f}%w {clear_sym:.3f}%c"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--two-class-classifier',
                        action='store_true', help='Use two-class classifier')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--fp16', action='store_true', help='Use FP16')

    parser.add_argument('--model-path', type=str, default='./model.pt')
    parser.add_argument('--is-inprogress-model', action='store_true')
    parser.add_argument('--deepspeed-model', action='store_true')

    parser.add_argument('--test-path', type=str, default='./test')
    parser.add_argument('--only-watermark', action='store_true')
    parser.add_argument('--only-clear', action='store_true')
    parser.add_argument('--log-all-images', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    args.prefetch_factor = args.batch_size
    args.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    # './watermarks/water_vit_wds/water-vit-train_009.tar'
    # '/mnt/akatta/datasets/water_vit_wds/water-vit-train_009.tar'
    #test_dl = '/mnt/akatta/watermark_testset/watermark_testset.tar'

    wandb.init(
        project='laion-watermark-detection',
        name='water_vit run eval',
        config={
            'two_class_classifier': args.two_class_classifier,
            'batch_size': args.batch_size,
            'prefetch_factor': args.prefetch_factor,
            'device': args.device,
            'model_path': args.model_path,
            'is_inprogress_model': args.is_inprogress_model,
            'test_path': args.test_path
        }
    )

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

    if args.deepspeed_model:
        state_dict = deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(
            args.model_path)
        torch.save(state_dict, f'{args.model_path}.pt')
    else:
        state_dict = torch.load(args.model_path)
        if args.is_inprogress_model:
            state_dict = state_dict['weights']

    model.load_state_dict(state_dict)
    model.eval().to(args.device)

    criterion = nn.CrossEntropyLoss()
    criterion.to(args.device)

    test_ds = get_dataset(args.test_path, args.batch_size, args.only_watermark, args.only_clear)
    test_dl = wds.WebLoader(test_ds, batch_size=None,
                            num_workers=1, prefetch_factor=args.prefetch_factor, pin_memory=True)

    average_loss_eval, test_dl_len = test_model(
        model, criterion, test_dl, args.device, args.batch_size, 'Test', args.fp16, args.log_all_images)
    tqdm.write(f'Test Loss: {average_loss_eval}')

    wandb.log({
        'test_loss': average_loss_eval
    })
    wandb.finish()

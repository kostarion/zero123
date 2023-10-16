import argparse
import math
import numpy as np
import os
import torch
import wandb
import warnings
from ldm.util import create_carvekit_interface, load_and_preprocess, instantiate_from_config
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader


VIEWS = [
    { 'name': 'left', 'polar': 0.0, 'azimuth': -92.0, 'r': 0.0 },
    { 'name': 'right', 'polar': 0.0, 'azimuth': 92.0, 'r': 0.0 },
    { 'name': 'above', 'polar': -89., 'azimuth': 0.0, 'r': 0.0 },
    { 'name': 'behind', 'polar': 0.0, 'azimuth': 177.0, 'r': 0.0 },
    { 'name': 'right50_up20', 'polar': -20.0, 'azimuth': 50.0, 'r': 0.0 },
]

TESTSET_ELEVATIONS = {
    'alien1': 5,
    'anya': 5,
    'beach_house': 50,
    'beetle': 5,
    'boy2': 5,
    'castle2': 20,
    'dad3': 5,
    'dancer1': 10,
    'dinoskeleton2': -5,
    'dodo1': 0,
    'dog1': 10,
    'dragon2': 10,
    'forest_temple': -10,
    'futuristic_car': 15,
    'girl2': 5,
    'globe1': 0,
    'grootplant': 0,
    'hamburger': 5,
    'horse': 5,
    'modern_house': 15,
    'mona_lisa': 5,
    'nendoroid_obama1': 5,
    'phoenix': 15,
    'retriever9': 10,
    'robot': 15,
    'sadhu1': 10,
    'turtle': 10,
    'w1': 5,
    'w5': 10,
    'woman': 5,
    'worker1': 5
}


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)

    parser.add_argument(
        "--dir_ckpt",
        nargs="?",
        type=str,
        default="",
        help="path to checkpoint or directory with checkpoints to load model states from"
    )
    parser.add_argument(
        "--run_name",
        nargs="?",
        type=str,
        default="zero123_baseline",
        help="wandb name of the run, should be different for each dir_ckpt"
    )
    parser.add_argument(
        "--model_config",
        nargs="?",
        type=str,
        default="configs/sd-objaverse-finetune-c_concat-256.yaml",
        help="path to model config yaml"
    )
    parser.add_argument(
        "--test_images_folder",
        nargs="?",
        type=str,
        default="/fsx/proj-mod3d/dmitry/repos/hard_examples_preprocessed/img",
        help="folder with images"
    )
    parser.add_argument(
        "--cfg_scales",
        nargs="*",
        type=float,
        default=[2.0, 3.0, 4.0],
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
    )
    parser.set_defaults(preprocess=False)
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="gpu index to run on",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--vae_ckpt_path",
        nargs="?",
        type=str,
        default="",
        help="path to vae checkpoint to load model state from. if set, overrides main ckpt vae weights"
    )
    parser.add_argument(
        "--clip_ckpt_path",
        nargs="?",
        type=str,
        default="",
        help="path to clip checkpoint to load model state from. if set, overrides main ckpt clip weights"
    )

def load_model_from_config(config, ckpt, device, vae_ckpt='', clip_ckpt='', verbose=False, min_step=0):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    sd = pl_sd['state_dict']
    global_step = 0
    if 'global_step' in pl_sd:
        global_step = pl_sd["global_step"]
        print(f'Global Step: {global_step}')
        if global_step < min_step:
            print("Skipping ckpt step ", global_step)
            return None, global_step
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    # if len(m) > 0 and verbose:
    #     print('missing keys:')
    #     print(m)
    # if len(u) > 0 and verbose:
    #     print('unexpected keys:')
    #     print(u)
    if vae_ckpt:
        print(f'Loading VAE from {vae_ckpt}')
        model.first_stage_model.load_state_dict(torch.load(vae_ckpt))
    if clip_ckpt:
        print(f'Loading CLIP from {clip_ckpt}')
        model.cond_stage_model.load_state_dict(torch.load(clip_ckpt))

    model.to(device)
    model.eval()
    return model, global_step

def create_text_image(text, width, height, font_size):
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', font_size)
    # font = ImageFont.load_default(font_size=font_size)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        text_width, text_height = draw.textsize(text, font=font)
    x = (width - text_width) / 2
    y = (height - text_height) / 2
    # draw the text on the image
    draw.text((x, y), text, font=font, fill=(0, 0, 0))
    return img

def create_image_grid(image_batch, image_original, col_names, row_names, imsize=256, label_size=20):
    imgs_grid = make_grid(image_batch.reshape(-1, 3, imsize, imsize), nrow=len(col_names), padding=0)

    full_grid = torch.ones(3, imsize * (len(row_names) + 1), imsize * (len(col_names) + 1))
    full_grid[:, :imsize, :imsize] = image_original
    full_grid[:, imsize:, imsize:] = imgs_grid

    for i, row_name in enumerate(row_names):
        label_img = transforms.ToTensor()(create_text_image(row_name, imsize, imsize, label_size))
        full_grid[:, imsize * (i + 1):imsize * (i + 2), :imsize] = label_img
    for i, col_name in enumerate(col_names):
        label_img = transforms.ToTensor()(create_text_image(col_name, imsize, imsize, label_size))
        full_grid[:, :imsize, imsize * (i + 1):imsize * (i + 2)] = label_img
    return full_grid


class ImageViewsDataset(Dataset):
    def __init__(self,
        img_dir,
        views,
        cfg_scales,
        img_size=256,
        elev_cond=False,
        preprocess=False,
    ) -> None:
        self.img_dir = img_dir
        self.img_paths = os.listdir(self.img_dir)
        self.img_paths = [img_path for img_path in self.img_paths if img_path.endswith('.png') or img_path.endswith('.jpg') or img_path.endswith('.jpeg')]
        self.img_names = [os.path.basename(img_path).split('.')[0] for img_path in self.img_paths]
        self.views = views
        self.cfg_scales = cfg_scales
        self.img_size = img_size
        self.elev_cond = elev_cond
        image_transforms = [transforms.Resize(img_size)]
        image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: x * 2. - 1.)])
        self.image_transforms = transforms.Compose(image_transforms)
        self.preprocess = preprocess
        if preprocess:
            self.carvekit = create_carvekit_interface()
            self.preprocess_imgs()

    def preprocess_imgs(self):
        preprocessed_imgs_dir = os.path.join(self.img_dir, 'preprocessed')
        print("Preprocessing images in ", self.img_dir, " to ", preprocessed_imgs_dir)
        if not os.path.exists(preprocessed_imgs_dir):
            os.mkdir(preprocessed_imgs_dir)
        new_img_paths = []
        for img_path in self.img_paths:
            img = Image.open(os.path.join(self.img_dir, img_path))
            img = Image.fromarray(load_and_preprocess(self.carvekit, img))
            new_img_name = img_path.replace('.jpg', '.png').replace('.jpeg', '.png')
            img.save(os.path.join(preprocessed_imgs_dir, new_img_name))
            new_img_paths.append(new_img_name)
        self.img_dir = preprocessed_imgs_dir
        self.img_paths = new_img_paths

    def __len__(self):
        return len(self.img_paths) * len(self.views)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx // len(self.views)]
        img_name = os.path.basename(img_path).split('.')[0]
        view = self.views[idx % len(self.views)]
        polar_shift_value = math.radians(
            max(view['polar'], TESTSET_ELEVATIONS[img_name] - 90) \
            if self.elev_cond and img_name in TESTSET_ELEVATIONS \
            else view['polar'])
        last_cond_value = 0 \
            if not self.elev_cond \
            else (math.radians(90 - TESTSET_ELEVATIONS[img_name]) \
                if img_name in TESTSET_ELEVATIONS \
                else math.pi / 2)
        T = torch.tensor([polar_shift_value,
                          math.sin(math.radians(view['azimuth'])),
                          math.cos(math.radians(view['azimuth'])),
                          last_cond_value])
        img = Image.open(os.path.join(self.img_dir, img_path)).convert("RGBA")
        background = Image.new('RGBA', img.size, (255, 255, 255))
        img = Image.alpha_composite(background, img).convert("RGB")
        img = self.image_transforms(img)
        return img, T, img_name, view['name']


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    device=f'cuda:{args.gpu}'

    # Sorry from the past 
    min_ckpt_step=100000
    finetuned_from_step=0
    elev_cond=False
    scale_cond=False

    run_dir=os.path.join('/fsx/proj-mod3d/dmitry/zero123_visualizations/', args.run_name)
    model_config='configs/sd-objaverse-finetune-c_concat-256.yaml'
    # model_config='configs/sd-objaverse-finetune-c_concat-512.yaml'
    config = OmegaConf.load(model_config)
    wandb_project = 'zero123_visualizations'
    wandb.init(project=wandb_project, name=args.run_name, id=args.run_name, entity='mod3d', save_code=False, resume='allow', reinit=True)
    wandb_logger = WandbLogger(project=wandb_project, name=args.run_name, id=args.run_name, save_dir=run_dir, offline=False)
    out_dir=os.path.join(run_dir, 'test_views')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # filter_checkpoints(dir_ckpt)

    for ckpt in sorted(os.listdir(args.dir_ckpt)):
        ckpt_path=os.path.join(args.dir_ckpt, ckpt)
        model, global_step = load_model_from_config(
            config, ckpt_path, f'cuda:{args.gpu}',
            vae_ckpt='vae.ckpt', clip_ckpt='clip.ckpt',
            verbose=True, min_step=min_ckpt_step)
        if model is None:
            continue

        print("\n*** STARTING VISUALIZATION RUN FOR CKPT STEP ", global_step, " ***")


        dataset = ImageViewsDataset(args.test_images_folder, VIEWS, args.cfg_scales,
                                    img_size=args.img_size, elev_cond=elev_cond, preprocess=False)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        out_dir_step = os.path.join(out_dir, f'step_{global_step}')
        if not os.path.exists(out_dir_step):
            os.mkdir(out_dir_step)
        print(f'==== Saving to {out_dir_step} ======')
        grids = {}
        for img_batch, T_batch, img_names, view_names in dataloader:
            novel_views = model.sample_novel_views(img_batch.to(device), T_batch.to(device),
                                                scales=args.cfg_scales, scale_cond=scale_cond)
            for i in range(len(img_names)):
                img_name = img_names[i]
                if img_name not in grids:
                    grids[img_name] = []
                grids[img_name].append(novel_views[i].detach().cpu())
                if len(grids[img_name]) == len(VIEWS):
                    img_views_tensor = torch.stack(grids[img_name])
                    full_image_grid = create_image_grid(
                        image_batch=img_views_tensor.moveaxis(0, 1),
                        image_original=(img_batch[i] + 1.0) / 2.0, 
                        col_names=[v['name'] for v in VIEWS], 
                        row_names=[f'cfg_scale_{s}' for s in args.cfg_scales],
                        imsize=args.img_size)
                    wandb_logger.experiment.log({img_name: wandb.Image(full_image_grid)}, step=global_step+finetuned_from_step)
                    full_image_grid_img = transforms.ToPILImage()(full_image_grid)
                    full_image_grid_img.save(os.path.join(out_dir_step, f'{img_name}.png'))
                    print (f'==== Saved {img_name}.png ====')
                    del grids[img_name]
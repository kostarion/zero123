from typing import Dict
import webdataset as wds
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
import torchvision
from einops import rearrange
from ldm.util import instantiate_from_config
from datasets import load_dataset
import pytorch_lightning as pl
import copy
import csv
import cv2
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset
import json
import os, sys
import webdataset as wds
import io
import tarfile
import math
import re
from safetensors.torch import load as load_sftr, load_file as load_sftr_file
from torch.utils.data.distributed import DistributedSampler

# Some hacky things to make experimentation easier
def make_transform_multi_folder_data(paths, caption_files=None, **kwargs):
    ds = make_multi_folder_data(paths, caption_files, **kwargs)
    return TransformDataset(ds)

def make_nfp_data(base_path):
    dirs = list(Path(base_path).glob("*/"))
    print(f"Found {len(dirs)} folders")
    print(dirs)
    tforms = [transforms.Resize(512), transforms.CenterCrop(512)]
    datasets = [NfpDataset(x, image_transforms=copy.copy(tforms), default_caption="A view from a train window") for x in dirs]
    return torch.utils.data.ConcatDataset(datasets)


class VideoDataset(Dataset):
    def __init__(self, root_dir, image_transforms, caption_file, offset=8, n=2):
        self.root_dir = Path(root_dir)
        self.caption_file = caption_file
        self.n = n
        ext = "mp4"
        self.paths = sorted(list(self.root_dir.rglob(f"*.{ext}")))
        self.offset = offset

        if isinstance(image_transforms, ListConfig):
            image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        image_transforms.extend([transforms.ToTensor(),
                                 transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        image_transforms = transforms.Compose(image_transforms)
        self.tform = image_transforms
        with open(self.caption_file) as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
        self.captions = dict(rows)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        for i in range(10):
            try:
                return self._load_sample(index)
            except Exception:
                # Not really good enough but...
                print("uh oh")

    def _load_sample(self, index):
        n = self.n
        filename = self.paths[index]
        min_frame = 2*self.offset + 2
        vid = cv2.VideoCapture(str(filename))
        max_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        curr_frame_n = random.randint(min_frame, max_frames)
        vid.set(cv2.CAP_PROP_POS_FRAMES,curr_frame_n)
        _, curr_frame = vid.read()

        prev_frames = []
        for i in range(n):
            prev_frame_n = curr_frame_n - (i+1)*self.offset
            vid.set(cv2.CAP_PROP_POS_FRAMES,prev_frame_n)
            _, prev_frame = vid.read()
            prev_frame = self.tform(Image.fromarray(prev_frame[...,::-1]))
            prev_frames.append(prev_frame)

        vid.release()
        caption = self.captions[filename.name]
        data = {
            "image": self.tform(Image.fromarray(curr_frame[...,::-1])),
            "prev": torch.cat(prev_frames, dim=-1),
            "txt": caption
        }
        return data

# end hacky things


def make_tranforms(image_transforms):
    # if isinstance(image_transforms, ListConfig):
    #     image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
    image_transforms = []
    image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
    image_transforms = transforms.Compose(image_transforms)
    return image_transforms


def make_multi_folder_data(paths, caption_files=None, **kwargs):
    """Make a concat dataset from multiple folders
    Don't suport captions yet

    If paths is a list, that's ok, if it's a Dict interpret it as:
    k=folder v=n_times to repeat that
    """
    list_of_paths = []
    if isinstance(paths, (Dict, DictConfig)):
        assert caption_files is None, \
            "Caption files not yet supported for repeats"
        for folder_path, repeats in paths.items():
            list_of_paths.extend([folder_path]*repeats)
        paths = list_of_paths

    if caption_files is not None:
        datasets = [FolderData(p, caption_file=c, **kwargs) for (p, c) in zip(paths, caption_files)]
    else:
        datasets = [FolderData(p, **kwargs) for p in paths]
    return torch.utils.data.ConcatDataset(datasets)



class NfpDataset(Dataset):
    def __init__(self,
        root_dir,
        image_transforms=[],
        ext="jpg",
        default_caption="",
        ) -> None:
        """assume sequential frames and a deterministic transform"""

        self.root_dir = Path(root_dir)
        self.default_caption = default_caption

        self.paths = sorted(list(self.root_dir.rglob(f"*.{ext}")))
        self.tform = make_tranforms(image_transforms)

    def __len__(self):
        return len(self.paths) - 1


    def __getitem__(self, index):
        prev = self.paths[index]
        curr = self.paths[index+1]
        data = {}
        data["image"] = self._load_im(curr)
        data["prev"] = self._load_im(prev)
        data["txt"] = self.default_caption
        return data

    def _load_im(self, filename):
        im = Image.open(filename).convert("RGB")
        return self.tform(im)

class ObjaverseDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, total_view, train=None, validation=None,
                 test=None, num_workers=4, data_config_file=None, **kwargs):
        super().__init__(self)
        self.root_dir = root_dir
        self.data_config_file = data_config_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.total_view = total_view

        if train is not None:
            dataset_config = train
        if validation is not None:
            dataset_config = validation

        if 'image_transforms' in dataset_config:
            image_transforms = [torchvision.transforms.Resize(dataset_config.image_transforms.size)]
        else:
            image_transforms = []
        image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.image_transforms = torchvision.transforms.Compose(image_transforms)


    def train_dataloader(self):
        dataset = ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=False, \
                                image_transforms=self.image_transforms, data_config_file=self.data_config_file)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        dataset = ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=True, \
                                image_transforms=self.image_transforms, data_config_file=self.data_config_file)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return wds.WebLoader(ObjaverseData(root_dir=self.root_dir, total_view=self.total_view, validation=self.validation,
                                           data_config_file=self.data_config_file),
                             batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class ExtendedObjaverseDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, datasets, batch_size, total_view, load_tensors=False,
                 train=None, validation=None, test=None, num_workers=4, **kwargs):
        super().__init__(self)
        for dataset_params in datasets:
            assert 'root_dir' in dataset_params and 'data_config_file' in dataset_params
        self.datasets_params = datasets
        self.batch_size = batch_size
        self.load_tensors = load_tensors
        self.total_view = total_view
        self.num_workers = num_workers

        if train is not None:
            dataset_config = train
        if validation is not None:
            dataset_config = validation

        if 'image_transforms' in dataset_config:
            image_transforms = [torchvision.transforms.Resize(dataset_config.image_transforms.size)]
        else:
            image_transforms = []
        if not self.load_tensors:
            image_transforms.extend([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.image_transforms = torchvision.transforms.Compose(image_transforms)

    def train_dataloader(self):
        datasets = [ExtendedObjaverseData(
            root_dir=dataset_params.root_dir,
            data_config_file=dataset_params.data_config_file,
            load_tensors=self.load_tensors,
            total_view=self.total_view,
            validation=False,
            image_transforms=self.image_transforms) for dataset_params in self.datasets_params]
        dataset = ConcatDataset(datasets)
        sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        datasets = [ExtendedObjaverseData(
            root_dir=dataset_params.root_dir,
            data_config_file=dataset_params.data_config_file,
            load_tensors=self.load_tensors,
            total_view=self.total_view,
            validation=True,
            image_transforms=self.image_transforms) for dataset_params in self.datasets_params]
        dataset = ConcatDataset(datasets)
        # sampler = DistributedSampler(dataset)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        datasets = [ExtendedObjaverseData(
            root_dir=dataset_params.root_dir,
            data_config_file=dataset_params.data_config_file,
            total_view=self.total_view,
            load_tensors=self.load_tensors,
            validation=self.validation) for dataset_params in self.datasets_params]
        dataset = ConcatDataset(datasets)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class ExtendedObjaverseData(Dataset):
    def __init__(self,
        root_dir='.objaverse/hf-objaverse-v1/views',
        data_config_file=None,
        load_tensors=False,
        image_transforms=[],
        postprocess=None,
        return_paths=False,
        total_view=54,
        validation=False,
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        print("********** ", root_dir)
        self.root_dir = Path(root_dir)
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        self.total_view = total_view
        self.load_tensors = load_tensors

        data_config_path = data_config_file if data_config_file else os.path.join(root_dir, 'valid_paths.json')
        with open(data_config_path) as f:
            self.paths = json.load(f)
            
        total_objects = len(self.paths)
        if validation:
            self.paths = self.paths[math.floor(total_objects / 100. * 99.):] # used last 1% as validation
        else:
            self.paths = self.paths[:math.floor(total_objects / 100. * 99.)] # used first 99% as training
        print('============= length of dataset %d =============' % len(self.paths))
        self.tform = image_transforms

    def __len__(self):
        return len(self.paths)

    def load_img(self, object_storage, index, fname_format_str='rgba/rgba_{view_index:04d}.png'):
        img_fname = f'{fname_format_str.format(view_index=index)}'
        if isinstance(object_storage, tarfile.TarFile):
            object_name = object_storage.getnames()[0]
            image = object_storage.extractfile(f'{object_name}/{img_fname}').read()
            image = Image.open(io.BytesIO(image)).convert('RGBA')
        else:
            image = Image.open(os.path.join(object_storage, img_fname)).convert('RGBA')
        return image
    
    def load_tensor(self, object_storage, index, color_index=0,
                             fname_format_str='view-{view_index:03d}-c{color_index:02d}.sftr', key='vae_latent'):
        tensor_fname = f'{fname_format_str.format(view_index=index, color_index=color_index)}'
        if isinstance(object_storage, tarfile.TarFile):
            object_name = object_storage.getnames()[0]
            tensor = object_storage.extractfile(f'{object_name}/{tensor_fname}').read()
            tensor = load_sftr(tensor)[key]
        else:
            tensor = load_sftr_file(os.path.join(object_storage, tensor_fname))[key]
        return tensor

    def load_viewpoint(self, object_storage, index, prefix='frame_'):
        metas_fname = f'{prefix}{index:04d}.json'
        if isinstance(object_storage, tarfile.TarFile):
            object_name = object_storage.getnames()[0]
            metas = object_storage.extractfile(f'{object_name}/{metas_fname}').read()
            metas = json.loads(metas)
        else:
            with open(os.path.join(object_storage, metas_fname), 'r') as f:
                metas = json.loads(f.read())
        polar, azimuth, r = metas["polar"], metas["azimuth"], metas["r"]
        return polar, azimuth, r

    def process_img(self, img, background_color=(255, 255, 255)):
        background = Image.new('RGBA', img.size, background_color)
        img = Image.alpha_composite(background, img).convert("RGB")
        return self.tform(img) if self.tform else img

    def get_T(self, object_storage, index_target, index_cond):
        target_polar, target_azimuth, target_r = self.load_viewpoint(object_storage, index_target)
        cond_polar, cond_azimuth, cond_r = self.load_viewpoint(object_storage, index_cond)
        
        d_polar = target_polar - cond_polar
        d_azimuth = (target_azimuth - cond_azimuth) % (2 * math.pi)
        d_r = target_r - cond_r
        
        d_T = torch.tensor([d_polar, math.sin(d_azimuth), math.cos(d_azimuth), d_r])
        return d_T

    def extract_data(self, object_storage, index_target, index_cond):
        data = {}
        if not self.load_tensors:
            data["image_target"] = self.process_img(self.load_img(object_storage, index_target))
            data["image_cond"] = self.process_img(self.load_img(object_storage, index_cond))
        else:
            data["latent_target"] = self.load_tensor(object_storage, index_target)
            data["latent_cond"] = self.load_tensor(object_storage, index_cond)
            data["clip_emb_cond"] = self.load_tensor(
                object_storage, index_cond,
                fname_format_str='clip-{view_index:03d}-c{color_index:02d}.sftr',
                key='clip_emb')
        data["T"] = self.get_T(object_storage, index_target, index_cond)

        return data

    def __getitem__(self, index):
        data = {}
        # TODO: set seed
        object_filepath = os.path.join(self.root_dir, self.paths[index])
        is_tar = False
        if object_filepath.endswith('.tar'):
            is_tar = True
        object_name = Path(object_filepath).stem
        if self.return_paths:
            data["path"] = str(object_filepath)

        object_storage = tarfile.open(object_filepath) if is_tar else object_filepath
        
        total_view = self.total_view
        object_files = object_storage.getnames() if is_tar else os.listdir(object_storage)
        if self.load_tensors:
            total_view = len([f for f in object_files if re.findall(r'view-(\d+)-c00.sftr', f)])
            total_view = min(total_view, len([f for f in object_files if re.findall(r'clip-(\d+)-c00.sftr', f)]))
        else:
            total_view = len([f for f in object_files if re.findall(r'rgba/rgba_(\d+).png', f)])

        # TODO: remove
        if total_view < 48:
            print(f"==== Invalid object {object_name} ====")
            return self.__getitem__((index + 1) % len(self.paths))

        try:
            index_target, index_cond = random.sample(range(total_view-1), 2) # without replacement
            # index_target, index_cond = 2, 1
            data = self.extract_data(object_storage, index_target, index_cond)
        except KeyboardInterrupt:
            raise
        except:
            print(f"************* Invalid files {object_filepath} {index_target} {index_cond} ***************")
            with open("/fsx/proj-mod3d/dmitry/repos/zero123/zero123/invalid_files.txt", "a") as f:
                f.write(f'{object_filepath}:({index_target}, {index_cond})\n')
            index_target, index_cond = 1, 2
            data = self.extract_data(object_storage, index_target, index_cond)
            return self.__getitem__((index + 1) % len(self.paths))

        # data['object_name'] = object_name

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data


class ObjaverseData(Dataset):
    def __init__(self,
        root_dir='.objaverse/hf-objaverse-v1/views',
        image_transforms=[],
        ext="png",
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        total_view=12,
        validation=False,
        data_config_file=None
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.default_trans = default_trans
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        self.total_view = total_view

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]

        data_config_path = data_config_file if data_config_file else os.path.join(root_dir, 'valid_paths.json')
        with open(data_config_path) as f:
            self.paths = json.load(f)
            
        total_objects = len(self.paths)
        if validation:
            self.paths = self.paths[math.floor(total_objects / 100. * 99.):] # used last 1% as validation
        else:
            self.paths = self.paths[:math.floor(total_objects / 100. * 99.)] # used first 99% as training
        print('============= length of dataset %d =============' % len(self.paths))
        self.tform = image_transforms

    def __len__(self):
        return len(self.paths)
        
    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        
        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond
        
        d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_T

    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        try:
            img = plt.imread(path)
        except:
            print(path)
            sys.exit()
        img[img[:, :, -1] == 0.] = color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img

    def __getitem__(self, index):

        data = {}
        total_view = 12
        index_target, index_cond = random.sample(range(total_view), 2) # without replacement
        filename = os.path.join(self.root_dir, self.paths[index])

        # print(self.paths[index])

        if self.return_paths:
            data["path"] = str(filename)
        
        color = [1., 1., 1., 1.]

        target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
        cond_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_cond), color))
        target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
        cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
        # try:
        #     target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
        #     cond_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_cond), color))
        #     target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
        #     cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
        # except:
        #     # very hacky solution, sorry about this
        #     filename = os.path.join(self.root_dir, '692db5f2d3a04bb286cb977a7dba903e_1') # this one we know is valid
        #     target_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_target), color))
        #     cond_im = self.process_im(self.load_im(os.path.join(filename, '%03d.png' % index_cond), color))
        #     target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
        #     cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))
        #     target_im = torch.zeros_like(target_im)
        #     cond_im = torch.zeros_like(cond_im)

        data["image_target"] = target_im
        data["image_cond"] = cond_im
        data["T"] = self.get_T(target_RT, cond_RT)

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)

class FolderData(Dataset):
    def __init__(self,
        root_dir,
        caption_file=None,
        image_transforms=[],
        ext="jpg",
        default_caption="",
        postprocess=None,
        return_paths=False,
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.default_caption = default_caption
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        if caption_file is not None:
            with open(caption_file, "rt") as f:
                ext = Path(caption_file).suffix.lower()
                if ext == ".json":
                    captions = json.load(f)
                elif ext == ".jsonl":
                    lines = f.readlines()
                    lines = [json.loads(x) for x in lines]
                    captions = {x["file_name"]: x["text"].strip("\n") for x in lines}
                else:
                    raise ValueError(f"Unrecognised format: {ext}")
            self.captions = captions
        else:
            self.captions = None

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]

        # Only used if there is no caption file
        self.paths = []
        for e in ext:
            self.paths.extend(sorted(list(self.root_dir.rglob(f"*.{e}"))))
        self.tform = make_tranforms(image_transforms)

    def __len__(self):
        if self.captions is not None:
            return len(self.captions.keys())
        else:
            return len(self.paths)

    def __getitem__(self, index):
        data = {}
        if self.captions is not None:
            chosen = list(self.captions.keys())[index]
            caption = self.captions.get(chosen, None)
            if caption is None:
                caption = self.default_caption
            filename = self.root_dir/chosen
        else:
            filename = self.paths[index]

        if self.return_paths:
            data["path"] = str(filename)

        im = Image.open(filename).convert("RGB")
        im = self.process_im(im)
        data["image"] = im

        if self.captions is not None:
            data["txt"] = caption
        else:
            data["txt"] = self.default_caption

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)
import random

class TransformDataset():
    def __init__(self, ds, extra_label="sksbspic"):
        self.ds = ds
        self.extra_label = extra_label
        self.transforms = {
            "align": transforms.Resize(768),
            "centerzoom": transforms.CenterCrop(768),
            "randzoom": transforms.RandomCrop(768),
        }


    def __getitem__(self, index):
        data = self.ds[index]

        im = data['image']
        im = im.permute(2,0,1)
        # In case data is smaller than expected
        im = transforms.Resize(1024)(im)

        tform_name = random.choice(list(self.transforms.keys()))
        im = self.transforms[tform_name](im)

        im = im.permute(1,2,0)

        data['image'] = im
        data['txt'] = data['txt'] + f" {self.extra_label} {tform_name}"

        return data

    def __len__(self):
        return len(self.ds)

def hf_dataset(
    name,
    image_transforms=[],
    image_column="image",
    text_column="text",
    split='train',
    image_key='image',
    caption_key='txt',
    ):
    """Make huggingface dataset with appropriate list of transforms applied
    """
    ds = load_dataset(name, split=split)
    tform = make_tranforms(image_transforms)

    assert image_column in ds.column_names, f"Didn't find column {image_column} in {ds.column_names}"
    assert text_column in ds.column_names, f"Didn't find column {text_column} in {ds.column_names}"

    def pre_process(examples):
        processed = {}
        processed[image_key] = [tform(im) for im in examples[image_column]]
        processed[caption_key] = examples[text_column]
        return processed

    ds.set_transform(pre_process)
    return ds

class TextOnly(Dataset):
    def __init__(self, captions, output_size, image_key="image", caption_key="txt", n_gpus=1):
        """Returns only captions with dummy images"""
        self.output_size = output_size
        self.image_key = image_key
        self.caption_key = caption_key
        if isinstance(captions, Path):
            self.captions = self._load_caption_file(captions)
        else:
            self.captions = captions

        if n_gpus > 1:
            # hack to make sure that all the captions appear on each gpu
            repeated = [n_gpus*[x] for x in self.captions]
            self.captions = []
            [self.captions.extend(x) for x in repeated]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        dummy_im = torch.zeros(3, self.output_size, self.output_size)
        dummy_im = rearrange(dummy_im * 2. - 1., 'c h w -> h w c')
        return {self.image_key: dummy_im, self.caption_key: self.captions[index]}

    def _load_caption_file(self, filename):
        with open(filename, 'rt') as f:
            captions = f.readlines()
        return [x.strip('\n') for x in captions]



import random
import json
class IdRetreivalDataset(FolderData):
    def __init__(self, ret_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(ret_file, "rt") as f:
            self.ret = json.load(f)

    def __getitem__(self, index):
        data = super().__getitem__(index)
        key = self.paths[index].name
        matches = self.ret[key]
        if len(matches) > 0:
            retreived = random.choice(matches)
        else:
            retreived = key
        filename = self.root_dir/retreived
        im = Image.open(filename).convert("RGB")
        im = self.process_im(im)
        # data["match"] = im
        data["match"] = torch.cat((data["image"], im), dim=-1)
        return data

import os
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from qualiteacher.utils import padding, solo_padding
from qualiteacher.data.data_util import unpaired_paths_from_folder
from qualiteacher.data.transforms import solo_random_crop
from pathlib import Path
import numpy as np

import cv2

from .randaugment import RandAugment
import torchvision.transforms as transforms

import mat73

@DATASET_REGISTRY.register()
class SingleDatasetUnderwaterLA(data.Dataset):
    """Single-image dataset with an auxiliary LA map.

    Expected directory layout:
      dataroot_lq: <folder with input images>
      dataroot_la: <folder with LA images>  (same filenames as lq)

    Returns:
      - 'lq'    : RGB tensor (C,H,W)
      - 'depth' : LA tensor  (C,H,W)  (NOT normalized)
      - 'lq_path', 'la_path', 'name'
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt["io_backend"]
        self.mean = opt.get("mean", None)
        self.std = opt.get("std", None)

        self.lq_folder = opt["dataroot_lq"]
        self.la_folder = opt["dataroot_la"]
        self.filename_tmpl = opt.get("filename_tmpl", "{}")

        # Reuse your helper that scans a folder into [{'real_path': ...}, ...]
        # It's already used in SemiDatasetUnderwaterLA for the real stream.
        self.paths = unpaired_paths_from_folder(self.lq_folder, "lq", self.filename_tmpl)

    def __getitem__(self, index):
        if not hasattr(self, "_printed_active"):
            self._printed_active = True
            print("[SingleDatasetUnderwaterLA] ACTIVE. Returning keys includes depth (LA).")

        if self.file_client is None:
            # NOTE: same pattern you use elsewhere; pop('type') is fine
            self.file_client = FileClient(self.io_backend_opt.pop("type"), **self.io_backend_opt)

        lq_path = self.paths[index]["lq_path"]  # key name from unpaired_paths_from_folder(real, key, tmpl)

        # robust fallback in case your helper uses a different key
        if not isinstance(lq_path, str):
            lq_path = self.paths[index].get("real_path")

        # Load LQ
        img_lq = imfrombytes(self.file_client.get(lq_path, "lq"), float32=True)

        # Load LA (same filename as lq)
        la_path = os.path.join(self.la_folder, os.path.basename(lq_path))
        img_la = imfrombytes(self.file_client.get(la_path, "la"), float32=True)

        # --- optional rescale for too-large images (keeps LQ and LA aligned) ---
        if self.opt.get("rescale_too_large_image", False):
            max_side = int(self.opt.get("rescale_max_side", 0))
            if max_side > 0:
                h, w = img_lq.shape[:2]
                cur_max = max(h, w)
                if cur_max > max_side:
                    s = max_side / float(cur_max)
                    new_w = int(round(w * s))
                    new_h = int(round(h * s))

                    img_lq = cv2.resize(img_lq, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    img_la = cv2.resize(img_la, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    # one-time confirmation (prints only once per process)
                    if not hasattr(self, "_printed_rescale_once"):
                        self._printed_rescale_once = True
                        print(
                            f"[SingleDatasetUnderwaterLA][RESCALE] "
                            f"{os.path.basename(lq_path)} {w}x{h} -> {new_w}x{new_h} (max_side={max_side})"
                        )
        # --- end rescale ---

        # To tensor (BGR->RGB)
        img_lq, img_la = img2tensor([img_lq, img_la], bgr2rgb=True, float32=True)

        # Normalize ONLY RGB images; DO NOT normalize LA
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)

        return {
            "lq": img_lq,
            "depth": img_la,  # LA goes here so QualiTeacher.feed_data grabs it
            "name": os.path.basename(lq_path),
            "lq_path": lq_path,
            "la_path": la_path,
        }
    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class SemiDatasetUnderwaterLA(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)

        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.real_folder = opt['dataroot_real']

        # NEW:
        self.la_folder = opt['dataroot_la']                 # labeled LA
        self.real_la_folder = opt['dataroot_real_la']       # unlabeled LA

        self.filename_tmpl = opt.get('filename_tmpl', '{}')

        self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)
        self.real_paths = unpaired_paths_from_folder(self.real_folder, 'real', self.filename_tmpl)

    def __getitem__(self, index):
        if not hasattr(self, "_printed_active"):
            self._printed_active = True
            print("[SemiDatasetUnderwaterLA] ACTIVE. Returning keys includes depth/real_depth.")

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # ---- labeled pair ----
        len_syn = len(self.paths)
        gt_path = self.paths[index % len_syn]['gt_path']
        lq_path = self.paths[index % len_syn]['lq_path']

        img_gt = imfrombytes(self.file_client.get(gt_path, 'gt'), float32=True)
        img_lq = imfrombytes(self.file_client.get(lq_path, 'lq'), float32=True)


        # load LA for labeled (same filename as lq)
        la_path = os.path.join(self.la_folder, os.path.basename(lq_path))
        img_la = imfrombytes(self.file_client.get(la_path, 'la'), float32=True)

        # ---- unlabeled real ----
        len_real = len(self.real_paths)
        real_path = self.real_paths[index % len_real]['real_path']
        img_real = imfrombytes(self.file_client.get(real_path, 'real'), float32=True)

        # load LA for real (same filename as real)
        real_la_path = os.path.join(self.real_la_folder, os.path.basename(real_path))
        img_real_la = imfrombytes(self.file_client.get(real_la_path, 'real_la'), float32=True)

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']

            # padding (keep LA aligned with LQ; solo_padding is ok if LA and LQ share size)
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)
            img_la = solo_padding(img_la, gt_size)

            img_real = solo_padding(img_real, gt_size)
            img_real_la = solo_padding(img_real_la, gt_size)

            # ---- paired crop for (gt, concat(lq, la)) ----
            img_lq_la = np.concatenate([img_lq, img_la], axis=2)   # (H,W,6)
            img_gt, img_lq_la = paired_random_crop(img_gt, img_lq_la, gt_size, scale, gt_path)
            img_lq = img_lq_la[:, :, :3]
            img_la = img_lq_la[:, :, 3:]

            # resize real + real_la consistently
            img_real = cv2.resize(img_real, (gt_size, gt_size), interpolation=cv2.INTER_CUBIC)
            img_real_la = cv2.resize(img_real_la, (gt_size, gt_size), interpolation=cv2.INTER_CUBIC)

            # augment (gt, concat(lq, la)) together
            img_lq_la = np.concatenate([img_lq, img_la], axis=2)
            img_gt, img_lq_la = augment([img_gt, img_lq_la], self.opt['use_hflip'], self.opt['use_rot'])
            img_lq = img_lq_la[:, :, :3]
            img_la = img_lq_la[:, :, 3:]

        if self.opt.get("rescale_too_large_image", False):
            max_side = int(self.opt.get("rescale_max_side", 0))
            if max_side > 0:
                h, w = img_lq.shape[:2]
                cur_max = max(h, w)
                if cur_max > max_side:
                    scale = max_side / float(cur_max)
                    new_w = int(round(w * scale))
                    new_h = int(round(h * scale))
                    img_lq = cv2.resize(img_lq, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    img_la = cv2.resize(img_la, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # to tensor
        img_gt, img_lq, img_la = img2tensor([img_gt, img_lq, img_la], bgr2rgb=True, float32=True)
        img_real = img2tensor(img_real, bgr2rgb=True, float32=True)
        img_real_la = img2tensor(img_real_la, bgr2rgb=True, float32=True)

        # strong aug (NOTE: see warning below)
        strong_aug = transforms.Compose([
            transforms.ToPILImage(),
            RandAugment(2, 10),
            transforms.ToTensor()
        ])
        img_real_strong = strong_aug(img_real)

        # normalize ONLY RGB images; DO NOT normalize LA (keep LA in [0,1])
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_real, self.mean, self.std, inplace=True)
            normalize(img_real_strong, self.mean, self.std, inplace=True)
            # do NOT normalize img_la / img_real_la

        return {
            'lq': img_lq,
            'gt': img_gt,
            'depth': img_la,                 # LA for labeled
            'real': img_real,
            'real_strong': img_real_strong,
            'real_depth': img_real_la,       # LA for real
            'real_name': os.path.basename(real_path),
            'lq_path': lq_path,              # fix your bug: was gt_path
            'gt_path': gt_path,
            'la_path': la_path,
            'real_la_path': real_la_path,
        }

    def __len__(self):
        return max(len(self.paths), len(self.real_paths))

def paired_paths_from_folder_by_prefix(
    lq_folder,
    gt_folder,
    filename_tmpl="{}",
    split_token="_",
    drop_last_n=2,
    exts=(".png", ".jpg", ".jpeg", ".bmp")
):
    """DEA-Net style pairing:
    - iterate ALL lq images
    - map each lq -> gt by base name derived from lq stem
    - returns list of dicts: [{'lq_path':..., 'gt_path':...}, ...]
    - DOES NOT require len(lq) == len(gt)
    """

    lq_folder = Path(lq_folder)
    gt_folder = Path(gt_folder)

    # build gt stem -> path map (GT is small ~9k so this is fast)
    gt_map = {}
    for p in gt_folder.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            gt_map[p.stem] = str(p)

    # list all lq files (LQ is huge ~313k)
    lq_files = [p for p in lq_folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    lq_files.sort(key=lambda p: p.name)

    def pick_base(stem: str):
        parts = stem.split(split_token)
        candidates = []
        # 1) prefix before first '_' (DEA-Net default)
        candidates.append(parts[0])
        # 2) drop last N tokens (common RESIDE: id_beta_A -> id)
        if len(parts) > drop_last_n:
            candidates.append(split_token.join(parts[:-drop_last_n]))
        # 3) full stem fallback
        candidates.append(stem)
        for c in candidates:
            if c in gt_map:
                return c
        return None

    pairs = []
    missing = 0
    for p in lq_files:
        base = pick_base(p.stem)
        if base is None:
            missing += 1
            continue
        pairs.append({"lq_path": str(p), "gt_path": gt_map[base]})

    if len(pairs) == 0:
        raise RuntimeError(
            f"[paired_paths_from_folder_by_prefix] 0 pairs found. "
            f"Check lq_folder={lq_folder} gt_folder={gt_folder}."
        )

    if missing > 0:
        print(f"[paired_paths_from_folder_by_prefix] WARNING: {missing} LQ files had no matching GT.")

    return pairs

@DATASET_REGISTRY.register()
class SemiDatasetOTSPrefix(data.Dataset):
    """SemiDataset for OTS-style (many-to-one) pairing: hazy-indexed + GT by prefix."""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)

        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.real_folder = opt['dataroot_real']
        self.filename_tmpl = opt.get('filename_tmpl', '{}')

        # DEA-style pairing (NO count equality required)
        self.paths = paired_paths_from_folder_by_prefix(
            self.lq_folder,
            self.gt_folder,
            filename_tmpl=self.filename_tmpl,
            split_token=opt.get("split_token", "_"),
            drop_last_n=opt.get("drop_last_n", 2),
        )

        # real stream stays unchanged
        self.real_paths = unpaired_paths_from_folder(self.real_folder, 'real', self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        len_syn = len(self.paths)
        pair = self.paths[index % len_syn]
        gt_path = pair['gt_path']
        lq_path = pair['lq_path']

        img_gt = imfrombytes(self.file_client.get(gt_path, 'gt'), float32=True)
        img_lq = imfrombytes(self.file_client.get(lq_path, 'lq'), float32=True)

        len_real = len(self.real_paths)
        real_path = self.real_paths[index % len_real]['real_path']
        img_real = imfrombytes(self.file_client.get(real_path, 'real'), float32=True)

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']

            img_gt, img_lq = padding(img_gt, img_lq, gt_size)
            img_real = solo_padding(img_real, gt_size)

            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            img_real = cv2.resize(img_real, (gt_size, gt_size), interpolation=cv2.INTER_CUBIC)

            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        img_real = img2tensor(img_real, bgr2rgb=True, float32=True)

        strong_aug = transforms.Compose([
            transforms.ToPILImage(),
            RandAugment(2, 10),
            transforms.ToTensor()
        ])
        img_real_strong = strong_aug(img_real)

        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_real, self.mean, self.std, inplace=True)
            normalize(img_real_strong, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'real': img_real,
            'real_strong': img_real_strong,
            'real_name': os.path.basename(real_path),
            'lq_path': lq_path,   # <-- fix: was wrong in your SemiDataset
            'gt_path': gt_path
        }

    def __len__(self):
        return max(len(self.paths), len(self.real_paths))

@DATASET_REGISTRY.register()
class PairedDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        # mean and std for normalizing the input images
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.filename_tmpl = opt['filename_tmpl'] if 'filename_tmpl' in opt else '{}'

        # file client (lmdb io backend)
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info' in self.opt and self.opt['meta_info'] is not None:
            # disk backend with meta_info
            # Each line in the meta_info describes the relative path to an image
            with open(self.opt['meta_info']) as fin:
                paths = [line.strip() for line in fin]
            self.paths = []
            for path in paths:
                gt_path, lq_path = path.split(', ')
                gt_path = os.path.join(self.gt_folder, gt_path)
                lq_path = os.path.join(self.lq_folder, lq_path)
                self.paths.append(dict([('gt_path', gt_path), ('lq_path', lq_path)]))
        else:
            # disk backend
            # it will scan the whole folder to get meta info
            # it will be time-consuming for folders with too many files. It is recommended using an extra meta txt file
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)



@DATASET_REGISTRY.register()
class SemiDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(SemiDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        # mean and std for normalizing the input images
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder, self.real_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_real']
        self.filename_tmpl = opt['filename_tmpl'] if 'filename_tmpl' in opt else '{}'

        self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)
        self.real_paths = unpaired_paths_from_folder(self.real_folder, 'real', self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        len_syn = len(self.paths)
        gt_path = self.paths[index % len_syn]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index % len_syn]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        len_real = len(self.real_paths)
        real_path = self.real_paths[index % len_real]['real_path']
        img_bytes = self.file_client.get(real_path, 'real')
        img_real = imfrombytes(img_bytes, float32=True)


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            img_real = solo_padding(img_real, gt_size)
            # img_real = cv2.resize(img_real, (gt_size, gt_size), interpolation=cv2.INTER_CUBIC)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            img_real = cv2.resize(img_real,(gt_size,gt_size), interpolation=cv2.INTER_LINEAR)
            img_real = img_real.clip(0, 1)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        img_real = img2tensor(img_real, bgr2rgb=True, float32=True)

        strong_aug = transforms.Compose([
            # TO PIL
            transforms.ToPILImage(),
            RandAugment(2, 10),
            # TO TENSOR
            transforms.ToTensor()
        ])

        img_real_strong = strong_aug(img_real)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_real, self.mean, self.std, inplace=True)
            normalize(img_real_strong, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'real': img_real,
            'real_strong': img_real_strong,
            'real_name': real_path.split('/')[-1],
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        if len(self.paths) >= len(self.real_paths):
            return len(self.paths)
        else:
            return len(self.real_paths)
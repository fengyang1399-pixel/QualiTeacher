import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from .sr_model import SRModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from torch.nn import functional as F
from collections import OrderedDict
from basicsr.utils.dist_util import master_only
import os
import os.path as osp
from basicsr.utils import get_root_logger, tensor2img, imwrite
from basicsr.metrics import calculate_metric
import pyiqa
import time
from tqdm import tqdm
import qualiteacher.archs.open_clip as open_clip
import qualiteacher.archs.memory_bank as memory_bank
import json
import scipy.linalg
#import lpips

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()

        r_index = torch.randperm(target.size(0)).to(self.device)

        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]

        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


@MODEL_REGISTRY.register()
class QualiTeacher(SRModel):
    """
    It is trained without GAN losses.
    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(QualiTeacher, self).__init__(opt)
        if self.is_train:
            self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
            if self.mixing_flag:
                mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
                use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
                self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        if self.is_train:
            self.block_size = self.opt['colabator'].get('block_size', None)
            if self.opt['colabator'].get('use_clip', False) or self.opt['train'].get('use_clip_loss', False):
                self.init_clip()
            if self.opt['colabator'].get('use_nr_iqa', False):
                self.init_nriqa()
            if self.opt['colabator'].get('use_brisque', False):
                self.init_brisque()
            self.init_mmb()
            self.musiq_score_counter = {i: 0 for i in range(10)}
            self.clip_score_counter = {i: 0 for i in range(10)}
            self.brisque_score_counter = {i: 0 for i in range(10)}
            self.combined_score_counter = {i: 0 for i in range(10)}

            self.geometric_augments = self.init_geometric_augment()
        
        self.enable_drop_stat = opt.get('enable_drop_stat', False)
        if self.enable_drop_stat:
            self.drop_stat_data = {}    

        if self.is_train:
            dpo_opt = opt.get('train', {})
            self.dpo_start_iter = dpo_opt.get('dpo_start_iter', 2000)
            self.dpo_beta = dpo_opt.get('dpo_beta', 1.0)
            self.dpo_lambda = dpo_opt.get('dpo_lambda', 1.0)
            self.dpo_delta = dpo_opt.get('dpo_delta', 0.1)       # L1 margin δ
            self.dpo_lambda2 = dpo_opt.get('dpo_lambda2', 0.5)   # L2 weight
            # self.lpips_fn = lpips.LPIPS(net='vgg').to(self.device).eval()
            # for p in self.lpips_fn.parameters():
            #     p.requires_grad = False

            self.use_local_consistency = dpo_opt.get('use_local_consistency', False)
            self.local_consistency_weight = dpo_opt.get('local_consistency_weight', 0.1)
            self.local_consistency_start_iter = dpo_opt.get('local_consistency_start_iter', 0)

            self.lambda_identity = dpo_opt.get('lambda_identity', 0.5)

            

    def init_clip(self):
        clip_model_type = self.opt['colabator'].get('clip_model_type', None)
        checkpoint = self.opt['colabator'].get('pretrained_clip_weight', None)
        tokenizer_type = self.opt['colabator'].get('tokenizer_type', None)
        self.clip_better = self.opt['colabator'].get('clip_better', None)
        self.degradation_type = self.opt['colabator'].get('degradation_type', None)
        self.weight_map_calculation = self.opt['colabator'].get('weight_map_calculation', 'addition')
        
        rank = self.opt.get('rank', 0)
        if rank == 0:
            self.clip_model, self.clip_preprocess = open_clip.create_model_from_pretrained(clip_model_type, pretrained=checkpoint)
        if self.opt['dist']:
            torch.distributed.barrier()
        if rank != 0:
            self.clip_model, self.clip_preprocess = open_clip.create_model_from_pretrained(clip_model_type, pretrained=checkpoint)

        self.clip_model = self.clip_model.to(self.device) 
        self.clip_model.eval()
        self.tokenizer = open_clip.get_tokenizer(tokenizer_type)
        degradations = ['motion-blurry', 'hazy', 'jpeg-compressed', 'low-light', 'noisy', 'raindrop', 'rainy',
                        'shadowed', 'snowy', 'uncompleted']
        text = self.tokenizer(degradations)
        text = text.to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            # if self.opt['dist']:
            #     text_features = self.clip_model.module.encode_text(text)
            # else:
            #     text_features = self.clip_model.encode_text(text)
            text_features = self.clip_model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            self.text_features = text_features

    def init_nriqa(self):
        nr_iqa_type = self.opt['colabator'].get('nr_iqa_type', None)
        self.nr_iqa_better = self.opt['colabator'].get('nr_iqa_better', None)
        self.nr_iqa_scale = self.opt['colabator'].get('nr_iqa_scale', None)

        rank = self.opt.get('rank', 0)
        if rank == 0:
            self.nr_iqa = pyiqa.create_metric(nr_iqa_type)
        if self.opt['dist']:
            torch.distributed.barrier()
        if rank != 0:
            self.nr_iqa = pyiqa.create_metric(nr_iqa_type)
        self.nr_iqa = self.nr_iqa.to(self.device).eval()
        #self.nr_iqa = pyiqa.create_metric(nr_iqa_type)
        #self.nr_iqa = self.model_to_device(self.nr_iqa).eval()
        #self.nr_iqa = self.nr_iqa.to(self.device).eval() 

    def init_brisque(self):
        rank = self.opt.get('rank', 0)
        if rank == 0:
            self.brisque = pyiqa.create_metric('brisque')
        if self.opt['dist']:
            torch.distributed.barrier()
        if rank != 0:
            self.brisque = pyiqa.create_metric('brisque')
        self.brisque = self.brisque.to(self.device).eval()                             
        #self.brisque = pyiqa.create_metric('brisque')
        #self.brisque = self.brisque.to(self.device).eval()

    def init_mmb(self):
        self.memory_bank = memory_bank.Memory_bank().to('cpu')
        memory_bank_path = self.opt['path'].get('pretrain_network_memory_bank', None)
        if memory_bank_path is not None:
            self.memory_bank.load_state_dict(torch.load(memory_bank_path, weights_only=False))

            
    def init_geometric_augment(self):             
        augments = [
        ('hflip', lambda x: torch.flip(x, [-1]), lambda x: torch.flip(x, [-1])),
        ('vflip', lambda x: torch.flip(x, [-2]), lambda x: torch.flip(x, [-2])),
        ('rot90', lambda x: torch.rot90(x, 1, [-2, -1]), lambda x: torch.rot90(x, -1, [-2, -1])),
        ]
        return augments

    def block_image(self, image, block_size):
        B, C, H, W = image.size()
        BH, BW = block_size

        # Calculate the image shape after the block
        num_H = H // BH
        num_W = W // BW

        # Reshape the image into a block shape
        blocked_image = image.view(B, C, num_H, BH, num_W, BW)

        # Exchange dimensions so that the blocks are in the right place.
        blocked_image = blocked_image.permute(2, 4, 0, 1, 3, 5).contiguous()

        # Reshaped to the original shape
        blocked_image = blocked_image.view(num_H * num_W * B, C, BH, BW)

        return blocked_image

    def unblock_image(self, blocked_image, block_size, original_shape):
        B, C, H, W = original_shape
        BH, BW = block_size

        # Calculate the image shape after the block
        num_H = H // BH
        num_W = W // BW

        # Reshape the image into a block shape
        blocked_image = blocked_image.view(num_H, num_W, B, 1, 1)

        # Exchange dimensions so that the blocks are in the right place.
        blocked_image = blocked_image.permute(2, 0, 3, 1, 4).contiguous()

        # Reshaped to the original shape
        blocked_image = blocked_image.view(B, 1, num_H, num_W)

        # Resize to original shape
        blocked_image = torch.nn.functional.interpolate(blocked_image, (H, W), mode='bilinear', align_corners=False)

        return blocked_image

    def get_clip_degrad_rate(self, img):
        image = self.clip_preprocess(img)
        sum_probs = 0
        for degradation in self.degradation_type:
            with torch.no_grad(), torch.cuda.amp.autocast():
                # if self.opt['dist']:
                #     _, degra_features = self.clip_model.module.encode_image(image, control=True)
                # else:
                #     _, degra_features = self.clip_model.encode_image(image, control=True)
                _, degra_features = self.clip_model.encode_image(image, control=True)
                # image_features /= image_features.norm(dim=-1, keepdim=True)
                degra_features /= degra_features.norm(dim=-1, keepdim=True)
                text_probs = (100.0 * degra_features @ self.text_features.T).softmax(dim=-1)
                probs = text_probs[:, degradation]
                sum_probs = sum_probs + probs
        return sum_probs
    
    def get_clip_degrad_rate_with_grad(self, img):
        """CLIP degradation rate without torch.no_grad(), for DPO gradient flow."""
        image = self.clip_preprocess(img)
        sum_probs = 0
        for degradation in self.degradation_type:
            with torch.cuda.amp.autocast():
                # if self.opt['dist']:
                #     _, degra_features = self.clip_model.module.encode_image(image, control=True)
                # else:
                #     _, degra_features = self.clip_model.encode_image(image, control=True)
                _, degra_features = self.clip_model.encode_image(image, control=True)
                degra_features = degra_features / degra_features.norm(dim=-1, keepdim=True)
                text_probs = (100.0 * degra_features @ self.text_features.T).softmax(dim=-1)
                probs = text_probs[:, degradation]
                sum_probs = sum_probs + probs
        return sum_probs

    def get_batch_avg_degrad_rate(self, imgs):
        sum_rate = self.get_clip_degrad_rate(imgs)
        sum_rate = sum_rate.mean()
        return sum_rate / imgs.shape[0]


    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'real' in data:
            self.real = data['real'].to(self.device)
        if 't' in data:
            self.transmission = data['t'].to(self.device)
        if 'real_strong' in data:
            self.real_strong = data['real_strong'].to(self.device)
        if 'depth' in data:
            self.la_sup = data['depth'].to(self.device)
        if 'real_depth' in data:
            self.la_real = data['real_depth'].to(self.device)
        if 'real_name' in data:
            self.real_name = data['real_name']
        if 'mini_gt_size' in data:
            self.mini_gt_size = data['mini_gt_size']
        if 'gt_size' in data:
            self.gt_size = data['gt_size']

        if self.is_train and self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def test(self):
        window_size = self.opt['val'].get('window_size', 0)
        pad_size = window_size if window_size else 32
        lq, mod_pad_h, mod_pad_w = self.pad_test(self.lq, pad_size)
        la, _, _ = self.pad_test(self.la_sup, pad_size)

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.outputs, self.output_transmissions, self.step_images, self.recon_images = self.net_g_ema(
                    img=lq, depth=la, score=torch.tensor(7, device=lq.device), debug=True)
                self.output = self.outputs[0].clamp(0,1)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.outputs, self.output_transmissions, self.step_images, self.recon_images = self.net_g(
                    img=lq, depth=la, score=torch.tensor(7, device=lq.device), debug=True)
                self.output = self.outputs[0].clamp(0,1)
            self.net_g.train()

        scale = self.opt.get('scale', 1)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def labal_selection(self, teacher_transmission_list, teacher_list, 
                    musiq_scores=None, clip_scores=None, brisque_scores=None):
        image_name = self.real_name[0]  

        if len(teacher_list) == 3:
            new_candidates = []

            for idx in range(len(teacher_list)):
                teacher = teacher_list[idx]
                teacher_transmission = teacher_transmission_list[idx]
            
                with torch.no_grad():
                    teacher_tar = teacher.detach()
                    original_shape = teacher_tar.size()

                    teacher_tar_blocks = self.block_image(teacher_tar, (self.block_size, self.block_size))
            
                    if self.opt['colabator'].get('use_nr_iqa', False):
                        # local
                        teacher_nr_iqa_score_sequence = self.nr_iqa(teacher_tar_blocks)

                        # global
                        if musiq_scores is not None:
                            teacher_nr_iqa_score = musiq_scores[idx]
                        else:
                            teacher_nr_iqa_score = (self.nr_iqa(teacher_tar) - self.nr_iqa_scale[0]) / (
                                    self.nr_iqa_scale[1] - self.nr_iqa_scale[0])
                
                        # unblock image 
                        if self.nr_iqa_scale != 'sigmoid':
                            teacher_nr_iqa_score_mask = (self.unblock_image(teacher_nr_iqa_score_sequence,
                                                                    (self.block_size, self.block_size),
                                                                    original_shape) - self.nr_iqa_scale[0]) / (
                                                            self.nr_iqa_scale[1] - self.nr_iqa_scale[0])
                        else:
                            teacher_nr_iqa_score_mask = torch.sigmoid(self.unblock_image(teacher_nr_iqa_score_sequence,
                                                                                 (self.block_size, self.block_size),
                                                                                 original_shape))
                        if self.nr_iqa_better == 'higher':
                            teacher_nr_iqa_score_mask = teacher_nr_iqa_score_mask
                            teacher_nr_iqa_score = teacher_nr_iqa_score
                        else:
                            teacher_nr_iqa_score_mask = 1 - teacher_nr_iqa_score_mask
                            teacher_nr_iqa_score = 1 - teacher_nr_iqa_score
                    else:
                        teacher_nr_iqa_score_mask = 0
                        teacher_nr_iqa_score = 0

                    if self.opt['colabator'].get('use_clip', False):
                        # local 
                        teacher_score_sequence = self.get_clip_degrad_rate(teacher_tar_blocks)
                
                        # global 
                        if clip_scores is not None:
                            teacher_score = clip_scores[idx]
                        else:
                            teacher_score = self.get_clip_degrad_rate(teacher_tar)
                            if self.clip_better == 'higher':
                                teacher_score = teacher_score
                            else:
                                teacher_score = len(self.degradation_type) - teacher_score

                        # unblock image
                        teacher_score_mask = len(self.degradation_type) - self.unblock_image(teacher_score_sequence,
                                                                                     (self.block_size, self.block_size),
                                                                                     original_shape)
                        if self.clip_better == 'higher':
                            teacher_score_mask = teacher_score_mask
                        else:
                            teacher_score_mask = len(self.degradation_type) - teacher_score_mask
                    else:
                        teacher_score_mask = 0
                        teacher_score = 0

                    if self.opt['colabator'].get('use_brisque', False):
                        if brisque_scores is not None:
                            teacher_brisque_score = brisque_scores[idx]
                        else:
                            teacher_brisque_score = self.brisque(teacher_tar)
                            teacher_brisque_score = torch.clamp(teacher_brisque_score, min=0)
                            teacher_brisque_score = (100.0 - teacher_brisque_score) / 100.0  
                    else:
                        teacher_brisque_score = 0
                
                    if self.opt['colabator'].get('use_brisque', False):
                        teacher_brisque_sequence = []
                        for i in range(teacher_tar_blocks.shape[0]):
                            try:
                                brisque_raw = self.brisque(teacher_tar_blocks[i:i+1])
                                brisque_raw = torch.clamp(brisque_raw, min=0)
                                brisque_norm = (100 - brisque_raw) / 100.0
                            except (AssertionError, RuntimeError):
                                brisque_norm = torch.tensor([0.5], device=teacher_tar_blocks.device)
                            teacher_brisque_sequence.append(brisque_norm)
                        teacher_brisque_sequence = torch.stack(teacher_brisque_sequence)
                        teacher_brisque_mask = self.unblock_image(teacher_brisque_sequence, (self.block_size, self.block_size), original_shape)
                    else:
                        teacher_brisque_mask = 0

                    # final mask
                    if self.weight_map_calculation == 'multiplication':
                        teacher_mask = teacher_nr_iqa_score_mask * (teacher_score_mask / len(self.degradation_type)) * teacher_brisque_mask
                    else:
                        teacher_mask = 0.4 * teacher_nr_iqa_score_mask + 0.2 * teacher_score_mask + 0.4 * teacher_brisque_mask 


                    new_candidates.append({
                        'image': teacher,
                        'transmission': teacher_transmission,
                        'musiq': teacher_nr_iqa_score,
                        'clip': teacher_score,
                        'brisque': teacher_brisque_score,
                        'mask': teacher_mask
                    })
        
            top3 = self.memory_bank.get_or_update_top3(image_name, new_candidates)
    
        else:
            top3 = self.memory_bank.get_or_update_top3(image_name, None)
    
        result_transmissions = []
        result_teachers = []
        result_masks = []
        result_musiq_scores = []
        result_clip_scores = []
        result_brisque_scores = []

        for item in top3:
            result_teachers.append(item['image'].to(self.device))
            result_transmissions.append(item['transmission'].to(self.device))
            result_masks.append(item['mask'].to(self.device) if item['mask'] is not None else torch.ones_like(item['image']).to(self.device))
            result_musiq_scores.append(item['musiq'].to(self.device) if torch.is_tensor(item['musiq']) else item['musiq'])
            result_clip_scores.append(item['clip'].to(self.device) if torch.is_tensor(item['clip']) else item['clip'])
            result_brisque_scores.append(item['brisque'].to(self.device) if torch.is_tensor(item['brisque']) else item['brisque'])
        while len(result_teachers) < 3:
            result_teachers.append(None)
            result_transmissions.append(None)
            result_masks.append(None)
            result_musiq_scores.append(None)
            result_clip_scores.append(None)
            result_brisque_scores.append(None)
    
        return result_transmissions, result_teachers, result_masks, result_musiq_scores, result_clip_scores, result_brisque_scores
    
    def random_crop(self, img, crop_ratio=0.5):
        B, C, H, W = img.shape
        crop_h = int(H * crop_ratio)
        crop_w = int(W * crop_ratio)
        top = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)
        cropped = img[:, :, top:top+crop_h, left:left+crop_w]
        return cropped, (top, left, crop_h, crop_w)

    def compute_combined_score(self, img):
        with torch.no_grad():
            musiq_score = self.nr_iqa(img)
            musiq_score = (musiq_score - self.nr_iqa_scale[0]) / (self.nr_iqa_scale[1] - self.nr_iqa_scale[0])
            clip_raw = self.get_clip_degrad_rate(img)
            if self.clip_better == 'higher':
                clip_score = clip_raw
            else:
                clip_score = len(self.degradation_type) - clip_raw
            brisque_raw = self.brisque(img)
            brisque_raw = torch.clamp(brisque_raw, min=0)
            brisque_score = (100 - brisque_raw) / 100.0
            combined_score = 0.4 * musiq_score + 0.2 * clip_score + 0.4 * brisque_score
        return combined_score

    def student_forward(self, img, score=None, la=None):
        with torch.enable_grad():
            outputs, transmissions = self.net_g(img, depth=la, score=score, finetune=True)
        return outputs[0].clamp(0, 1)
    
    def compute_local_consistency_reward(self, img):
        if hasattr(self.nr_iqa, 'module'):
            musiq_score = self.nr_iqa.module.net(img)
        else:
            musiq_score = self.nr_iqa.net(img)
        musiq_score = (musiq_score - self.nr_iqa_scale[0]) / (self.nr_iqa_scale[1] - self.nr_iqa_scale[0])
        clip_raw = self.get_clip_degrad_rate_with_grad(img)
        if self.clip_better == 'higher':
            clip_score = clip_raw
        else:
            clip_score = len(self.degradation_type) - clip_raw
        with torch.no_grad():
            brisque_raw = self.brisque(img)
            brisque_raw = torch.clamp(brisque_raw, min=0)
            brisque_score = (100.0 - brisque_raw) / 100.0
        reward = 0.4 * musiq_score + 0.2 * clip_score + 0.4 * brisque_score
        return reward


    def compute_local_consistency_loss(self, real_img, la=None):
        score_high = torch.tensor(7, device=real_img.device)
        full_restored = self.student_forward(real_img, score=score_high, la=la)
        crop1, _ = self.random_crop(full_restored, crop_ratio=0.5)
        s1 = self.compute_local_consistency_reward(crop1)
        crop2_input, (top, left, crop_h, crop_w) = self.random_crop(real_img, crop_ratio=0.5)
        crop2_la = la[:, :, top:top+crop_h, left:left+crop_w] if la is not None else None
        crop2_restored = self.student_forward(crop2_input, score=score_high, la=crop2_la)
        s2 = self.compute_local_consistency_reward(crop2_restored)
        loss = F.l1_loss(s1, s2)
        return loss, full_restored

    def compute_dpo_reward(self, img):
        """Compute combined IQA reward for DPO."""
        # MUSIQ: with grad
        if hasattr(self.nr_iqa, 'module'):
            musiq_score = self.nr_iqa.module.net(img)
        else:
            musiq_score = self.nr_iqa.net(img)
        musiq_score = (musiq_score - self.nr_iqa_scale[0]) / (self.nr_iqa_scale[1] - self.nr_iqa_scale[0])
        
        # CLIP: with grad
        clip_raw = self.get_clip_degrad_rate_with_grad(img)
        if self.clip_better == 'higher':
            clip_score = clip_raw
        else:
            clip_score = len(self.degradation_type) - clip_raw
        
        # BRISQUE: no grad (non-differentiable)
        with torch.no_grad():
            brisque_raw = self.brisque(img)
            brisque_raw = torch.clamp(brisque_raw, min=0)
            brisque_score = (100.0 - brisque_raw) / 100.0
        
        reward = 0.4 * musiq_score + 0.2 * clip_score + 0.4 * brisque_score
        return reward

    def optimize_parameters(self, current_iter, log_vars=None):
        if current_iter % self.opt['logger']['save_checkpoint_freq'] == 0:
            self.save_memory_bank(current_iter)
            self.save_score_distribution(current_iter)
            if self.enable_drop_stat: 
                self.save_drop_statistics(self.opt['path']['visualization'])
        
        self.optimizer_g.zero_grad()
        outputs, transmissions = self.net_g(self.lq, depth=self.la_sup, score=None, finetune=True)
        output = outputs[0].clamp(0, 1)
        transmission = transmissions[0]
        recon_lq = output * transmission + (1 - transmission)

        BATCH_SIZE = self.real.shape[0]

        all_pseudo_labels = []
        all_pseudo_transmissions = []
        all_pseudo_masks = []
        all_pseudo_musiq_scores = []
        all_pseudo_clip_scores = []
        all_pseudo_brisque_scores = []
        all_combined_discrete = []

        with torch.no_grad():
            for batch_idx in range(BATCH_SIZE):
                aug_candidates_single = []
                all_pass_drop1 = True
                temp_candidates = []
        
                for aug_name, aug_fn, inv_fn in self.geometric_augments:
                    real_single = self.real[batch_idx:batch_idx+1]
                    real_aug = aug_fn(real_single)
                    la_aug = aug_fn(self.la_real[batch_idx:batch_idx+1])
                    pseudo_aug, trans_aug = self.net_g_ema(real_aug, depth=la_aug, finetune=True)   
                    
                    pseudo_original = inv_fn(pseudo_aug[0].clamp(0, 1))                
                    trans_original = inv_fn(trans_aug[0])

                    musiq_score = self.nr_iqa(pseudo_original)                        
                    musiq_score = (musiq_score - self.nr_iqa_scale[0]) / (self.nr_iqa_scale[1] - self.nr_iqa_scale[0])

                    clip_raw = self.get_clip_degrad_rate(pseudo_original)
                    if self.clip_better == 'higher':
                        clip_score = clip_raw
                    else:
                        clip_score = len(self.degradation_type) - clip_raw
                    brisque_raw = self.brisque(pseudo_original)
                    brisque_raw = torch.clamp(brisque_raw, min=0)
                    brisque_score = (100 - brisque_raw) / 100.0

                    iqa_scores = torch.tensor([
                        musiq_score.item(),
                        brisque_score.item()
                    ])
                    iqa_std = torch.std(iqa_scores).item()
                    if self.enable_drop_stat:
                        img_name = self.real_name[batch_idx]
                        if img_name not in self.drop_stat_data:
                            self.drop_stat_data[img_name] = {'iqa_stds': [], 'musiq_scores': []}
                        self.drop_stat_data[img_name]['iqa_stds'].append(iqa_std)
                        self.drop_stat_data[img_name]['musiq_scores'].append(musiq_score.item())
       
                    drop_threshold_multi_iqa = self.opt['colabator'].get('drop_threshold_multi_iqa', 1.0)
            
                    if iqa_std <= drop_threshold_multi_iqa:
                        temp_candidates.append({
                            'pseudo_label': pseudo_original,
                            'pseudo_transmission': trans_original,
                            'musiq_score': musiq_score,
                            'clip_score': clip_score,
                            'brisque_score': brisque_score,
                            'aug_name': aug_name
                        })
                    else:
                        all_pass_drop1 = False
        
                if all_pass_drop1 and len(temp_candidates) == len(self.geometric_augments):
                    musiq_scores_list = [c['musiq_score'].item() for c in temp_candidates]
                    musiq_aug_std = torch.std(torch.tensor(musiq_scores_list)).item()
            
                    drop_threshold_aug_robust = self.opt['colabator'].get('drop_threshold_aug_robust', 1.0)
            
                    if musiq_aug_std <= drop_threshold_aug_robust:
                        aug_candidates_single = temp_candidates
                    else:
                        aug_candidates_single = []
                else:
                    aug_candidates_single = []
        
                if len(aug_candidates_single) > 0:
                    teacher_list = [c['pseudo_label'] for c in aug_candidates_single]
                    teacher_transmission_list = [c['pseudo_transmission'] for c in aug_candidates_single]
                    musiq_scores = [c['musiq_score'] for c in aug_candidates_single]
                    clip_scores = [c['clip_score'] for c in aug_candidates_single]
                    brisque_scores = [c['brisque_score'] for c in aug_candidates_single]
                else:
                    teacher_list = []
                    teacher_transmission_list = []
                    musiq_scores = None
                    clip_scores = None
                    brisque_scores = None

                pseudo_trans, pseudo_labs, pseudo_msks, musiq_s, clip_s, brisque_s = \
                    self.labal_selection(teacher_transmission_list, teacher_list,
                                    musiq_scores, clip_scores, brisque_scores)

                for i in range(len(pseudo_labs)):  
                    all_pseudo_labels.append(pseudo_labs[i])
                    all_pseudo_transmissions.append(pseudo_trans[i])
                    all_pseudo_masks.append(pseudo_msks[i])
                    all_pseudo_musiq_scores.append(musiq_s[i])
                    all_pseudo_clip_scores.append(clip_s[i])
                    all_pseudo_brisque_scores.append(brisque_s[i])

                    if pseudo_labs[i] is not None:
                        combined_score = 0.4 * musiq_s[i] + 0.2 * clip_s[i] + 0.4 * brisque_s[i] 
                        combined_discrete = torch.where(combined_score >= 0.7,
                                                torch.tensor(7, device=combined_score.device),
                                                (combined_score * 10).floor().long()).clamp(0, 7)
                        all_combined_discrete.append(combined_discrete)

                        musiq_discrete = (musiq_s[i] * 10).floor().long().clamp(0, 9)
                        clip_discrete = (clip_s[i] * 10).floor().long().clamp(0, 9)
                        brisque_discrete = (brisque_s[i] * 10).floor().long().clamp(0, 9)
                
                        self.musiq_score_counter[musiq_discrete.item()] += 1
                        self.clip_score_counter[clip_discrete.item()] += 1
                        self.brisque_score_counter[brisque_discrete.item()] += 1
                        self.combined_score_counter[combined_discrete.item()] += 1
                    else:
                            all_combined_discrete.append(None)
        valid_count = sum(1 for x in all_pseudo_labels if x is not None)
        total_count = len(all_pseudo_labels)

        l_total = 0
        loss_dict = OrderedDict()

        if self.cri_pix:
            l_pix = self.cri_pix(output, self.gt).mean()
            loss_dict['l_pix_gt'] = l_pix
            l_pix = l_pix * 5
            l_total += l_pix
        
        if self.opt['train'].get('use_clip_loss', False):
            clip_loss = self.get_batch_avg_degrad_rate(output)
            loss_dict['clip_loss_gt'] = clip_loss
            l_total += clip_loss
        
        if self.cri_contrastperceptual:
            l_percep, l_style = self.cri_contrastperceptual.standard_perceptual_loss(output, self.gt)
            if l_percep is not None:
                l_total += (l_percep * 0.2)
                loss_dict['l_percep'] = l_percep
            if l_style is not None: 
                l_total += (l_style * 0.2)
                loss_dict['l_style'] = l_style

        for i in range(len(all_pseudo_labels)):
            img_idx = i // 3
    
            if all_pseudo_labels[i] is not None:
                real_outputs, real_transmissions = self.net_g(
                    self.real_strong[img_idx:img_idx+1],
                    depth=self.la_real[img_idx:img_idx+1],
                    score=all_combined_discrete[i],
                    finetune=True
                )
                real_output = real_outputs[0]
                real_transmission = real_transmissions[0]
                recon_real_lq = real_output * real_transmission + (1 - real_transmission)
        
                if self.cri_pix:
                    l_pix_real = (self.cri_pix(real_output, all_pseudo_labels[i]) * all_pseudo_masks[i] * 2).mean()
                    l_total += l_pix_real
            
                    l_asm = (self.cri_pix(recon_real_lq, self.real_strong[img_idx:img_idx+1]) * all_pseudo_masks[i]).mean() * 0.01
                    l_total += l_asm
            
                    loss_dict[f'l_pix_real_{i}'] = l_pix_real
                    loss_dict[f'l_asm_{i}'] = l_asm
        
                if self.opt['train'].get('use_clip_loss', False):
                    clip_loss_real = self.get_batch_avg_degrad_rate(real_output)
                    l_total += clip_loss_real    
                    loss_dict[f'clip_loss_real_{i}'] = clip_loss_real
        
                if self.cri_contrastperceptual:
                    l_contrast_percep, l_contrast_style = self.cri_contrastperceptual(
                        real_output, all_pseudo_labels[i], self.real_strong[img_idx:img_idx+1]
                    )
                    if l_contrast_percep is not None:
                        l_total += l_contrast_percep
                        loss_dict[f'l_contrast_percep_{i}'] = l_contrast_percep
                    if l_contrast_style is not None:
                        l_total += l_contrast_style
                        loss_dict[f'l_contrast_style_{i}'] = l_contrast_style
            else:
                real_outputs, real_transmissions = self.net_g(
                    self.real_strong[img_idx:img_idx+1],
                    depth=self.la_real[img_idx:img_idx+1],
                    score=torch.tensor(0, device=self.device),
                    finetune=True
                )
                real_output = real_outputs[0]
        
                dummy_loss = (real_output * 0).sum()
                l_total += dummy_loss
        
                loss_dict[f'l_pix_real_{i}'] = torch.tensor(0.0, device=self.device)
                loss_dict[f'l_asm_{i}'] = torch.tensor(0.0, device=self.device)
                if self.opt['train'].get('use_clip_loss', False):
                    loss_dict[f'clip_loss_real_{i}'] = torch.tensor(0.0, device=self.device)
                if self.cri_contrastperceptual:
                    loss_dict[f'l_contrast_percep_{i}'] = torch.tensor(0.0, device=self.device)
                    loss_dict[f'l_contrast_style_{i}'] = torch.tensor(0.0, device=self.device)  
        
        l_total.backward()

        # ==================== Local Consistency Loss ====================
        if self.use_local_consistency and current_iter >= self.local_consistency_start_iter:
            l_local_total = 0
            for img_idx in range(BATCH_SIZE):
                l_local, _ = self.compute_local_consistency_loss(self.real_strong[img_idx:img_idx+1], la=self.la_real[img_idx:img_idx+1])
                l_local_total = l_local_total + l_local
            l_local_avg = self.local_consistency_weight * l_local_total / BATCH_SIZE
            l_local_avg.backward()
            loss_dict['l_local_consistency'] = l_local_avg.item()

        # ==================== DPO Loss ====================
        if current_iter >= self.dpo_start_iter:
            net_g = self.net_g.module if hasattr(self.net_g, 'module') else self.net_g

            if not hasattr(self, '_dirac_convs'):
                self._dirac_convs = []
                try:
                    from qualiteacher.archs.corun_arch import Upsample as CORunUpsample
                    for name, module in net_g.named_modules():
                        if isinstance(module, CORunUpsample):
                            self._dirac_convs.append(module.body[2])
                except ImportError:
                    pass
                if len(self._dirac_convs) == 0 and hasattr(net_g, 'ups'):
                    for up in net_g.ups:
                        if len(up) > 2:
                            self._dirac_convs.append(up[2])
                if len(self._dirac_convs) == 0 and hasattr(net_g, 'smooth_convs'):
                    for conv in net_g.smooth_convs:
                        self._dirac_convs.append(conv)
                for i, conv in enumerate(self._dirac_convs):
                    setattr(self, f'dirac_target_{i}', conv.weight.data.clone())

            # Identity Regularization
            l_identity_reg = 0.0
            for i, conv in enumerate(self._dirac_convs):
                dirac_target = getattr(self, f'dirac_target_{i}')
                l_identity_reg = l_identity_reg + torch.sum((conv.weight - dirac_target) ** 2)
            if isinstance(l_identity_reg, torch.Tensor):
                l_identity_reg = self.lambda_identity * l_identity_reg
                l_identity_reg.backward()
                loss_dict['l_identity_reg'] = l_identity_reg.item()
            else:
                loss_dict['l_identity_reg'] = 0.0

            # DPO
            l_dpo_l1_total = 0
            l_dpo_l2_total = 0
            dpo_pair_count = 0

            for img_idx in range(BATCH_SIZE):
                real_input = self.real_strong[img_idx:img_idx+1]

                # === L1: student high vs student low ===
                score_high = torch.tensor(7, device=self.device)
                score_low = torch.tensor(random.choice([4, 5, 6]), device=self.device)

                y_w_out, _ = self.net_g(real_input, depth=self.la_real[img_idx:img_idx+1], score=score_high, finetune=True)
                y_l_out, _ = self.net_g(real_input, depth=self.la_real[img_idx:img_idx+1], score=score_low, finetune=True)
                y_w = y_w_out[0].clamp(0, 1)
                y_l = y_l_out[0].clamp(0, 1)

                r_w = self.compute_dpo_reward(y_w)
                r_l = self.compute_dpo_reward(y_l)

                # L1 = -log σ(β * (r_s(y_h) - r_s(y_l) - δ))
                logit_l1 = self.dpo_beta * (r_w - r_l - self.dpo_delta)
                l_dpo_l1 = -F.logsigmoid(logit_l1)
                l_dpo_l1_total = l_dpo_l1_total + l_dpo_l1

                # === L2: student high vs teacher best ===
                # Find teacher's best reward from the 3 pseudo labels
                teacher_rewards = []
                for k in range(3):
                    pl_idx = img_idx * 3 + k
                    if pl_idx < len(all_pseudo_labels) and all_pseudo_labels[pl_idx] is not None:
                        combined = 0.4 * all_pseudo_musiq_scores[pl_idx] + \
                                   0.2 * all_pseudo_clip_scores[pl_idx] + \
                                   0.4 * all_pseudo_brisque_scores[pl_idx]
                        teacher_rewards.append(combined)

                if len(teacher_rewards) > 0:
                    # teacher best reward (detached, no grad)
                    r_t_best = torch.stack([r.detach() if torch.is_tensor(r) else torch.tensor(r, device=self.device) for r in teacher_rewards]).max()

                    # L2 = -log σ(β * (r_s(y_h) - r_t(y_best)))
                    logit_l2 = self.dpo_beta * (r_w - r_t_best)
                    l_dpo_l2 = -F.logsigmoid(logit_l2)
                    l_dpo_l2_total = l_dpo_l2_total + l_dpo_l2
                # # === L2: LPIPS anchor to teacher best ===
                # best_pl = None
                # best_reward = -1
                # for k in range(3):
                #     pl_idx = img_idx * 3 + k
                #     if pl_idx < len(all_pseudo_labels) and all_pseudo_labels[pl_idx] is not None:
                #         combined = 0.4 * all_pseudo_musiq_scores[pl_idx] + \
                #                    0.2 * all_pseudo_clip_scores[pl_idx] + \
                #                    0.4 * all_pseudo_brisque_scores[pl_idx]
                #         r_val = combined.item() if torch.is_tensor(combined) else combined
                #         if r_val > best_reward:
                #             best_reward = r_val
                #             best_pl = all_pseudo_labels[pl_idx]

                # if best_pl is not None:
                #     l_dpo_l2 = self.lpips_fn(y_w * 2 - 1, best_pl.detach() * 2 - 1).mean()
                #     l_dpo_l2_total = l_dpo_l2_total + l_dpo_l2

                dpo_pair_count += 1
                

            if dpo_pair_count > 0:
                l_dpo = self.dpo_lambda * (l_dpo_l1_total + self.dpo_lambda2 * l_dpo_l2_total) / dpo_pair_count
                l_dpo.backward()
                loss_dict['l_dpo'] = l_dpo.item() if isinstance(l_dpo, torch.Tensor) else l_dpo
                loss_dict['l_dpo_l1'] = (self.dpo_lambda * l_dpo_l1_total / dpo_pair_count).item() if isinstance(l_dpo_l1_total, torch.Tensor) else 0.0
                loss_dict['l_dpo_l2'] = (self.dpo_lambda * self.dpo_lambda2 * l_dpo_l2_total / dpo_pair_count).item() if isinstance(l_dpo_l2_total, torch.Tensor) else 0.0


        self.optimizer_g.step()
        
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        
        self.log_dict = loss_dict

    def save_score_distribution(self, current_iter):
        import os
        
        save_path = os.path.join(self.opt['path']['visualization'], 'score_distribution')
        os.makedirs(save_path, exist_ok=True)
        
        with open(os.path.join(save_path, f'distribution_iter_{current_iter}.txt'), 'w') as f:
            f.write(f"Iteration: {current_iter}\n")
            f.write(f"Total MUSIQ samples: {sum(self.musiq_score_counter.values())}\n")
            f.write(f"Total CLIP samples: {sum(self.clip_score_counter.values())}\n")
            f.write(f"Total Combined samples: {sum(self.combined_score_counter.values())}\n") 
            f.write(f"Total BRISQUE samples: {sum(self.brisque_score_counter.values())}\n") 
            f.write(f"\nMUSIQ Score Distribution:\n")
            for score in range(10):
                count = self.musiq_score_counter[score]
                percentage = count / sum(self.musiq_score_counter.values()) * 100 if sum(self.musiq_score_counter.values()) > 0 else 0
                f.write(f"Score {score}: {count:6d} ({percentage:5.2f}%)\n")
            f.write(f"\nCLIP Score Distribution:\n")
            for score in range(10):
                count = self.clip_score_counter[score]
                percentage = count / sum(self.clip_score_counter.values()) * 100 if sum(self.clip_score_counter.values()) > 0 else 0
                f.write(f"Score {score}: {count:6d} ({percentage:5.2f}%)\n")
            f.write(f"\nCombined Score Distribution:\n")
            for score in range(10):
                count = self.combined_score_counter[score]
                percentage = count / sum(self.combined_score_counter.values()) * 100 if sum(self.combined_score_counter.values()) > 0 else 0
                f.write(f"Score {score}: {count:6d} ({percentage:5.2f}%)\n")
            f.write(f"\nBRISQUE Score Distribution:\n")
            for score in range(10):
                count = self.brisque_score_counter[score]
                percentage = count / sum(self.brisque_score_counter.values()) * 100 if sum(self.brisque_score_counter.values()) > 0 else 0
                f.write(f"Score {score}: {count:6d} ({percentage:5.2f}%)\n")
    
        with open(os.path.join(save_path, 'distribution_latest.txt'), 'w') as f:
            f.write(f"Latest Iteration: {current_iter}\n")
            f.write(f"Total MUSIQ samples: {sum(self.musiq_score_counter.values())}\n")
            f.write(f"Total CLIP samples: {sum(self.clip_score_counter.values())}\n")
            f.write(f"Total BRISQUE samples: {sum(self.brisque_score_counter.values())}\n")
            f.write(f"Total Combined samples: {sum(self.combined_score_counter.values())}\n") 
            f.write(f"\nMUSIQ Score Distribution:\n")
            for score in range(10):
                count = self.musiq_score_counter[score]
                percentage = count / sum(self.musiq_score_counter.values()) * 100 if sum(self.musiq_score_counter.values()) > 0 else 0
                f.write(f"Score {score}: {count:6d} ({percentage:5.2f}%)\n")
            f.write(f"\nCLIP Score Distribution:\n")
            for score in range(10):
                count = self.clip_score_counter[score]
                percentage = count / sum(self.clip_score_counter.values()) * 100 if sum(self.clip_score_counter.values()) > 0 else 0
                f.write(f"Score {score}: {count:6d} ({percentage:5.2f}%)\n")
            f.write(f"\nCombined Score Distribution:\n")
            for score in range(10):
                count = self.combined_score_counter[score]
                percentage = count / sum(self.combined_score_counter.values()) * 100 if sum(self.combined_score_counter.values()) > 0 else 0
                f.write(f"Score {score}: {count:6d} ({percentage:5.2f}%)\n")
            f.write(f"\nBRISQUE Score Distribution:\n")
            for score in range(10):
                count = self.brisque_score_counter[score]
                percentage = count / sum(self.brisque_score_counter.values()) * 100 if sum(self.brisque_score_counter.values()) > 0 else 0
                f.write(f"Score {score}: {count:6d} ({percentage:5.2f}%)\n")

    def save_drop_statistics(self, save_path):
        if not self.enable_drop_stat:
            return
        all_iqa_stds = []
        all_aug_stds = []
    
        for img_name, data in self.drop_stat_data.items():
            iqa_stds = data['iqa_stds'] 
            musiq_scores = data['musiq_scores'] 

            all_iqa_stds.extend(iqa_stds)

            if len(musiq_scores) == 3:
                aug_std = np.std(musiq_scores)
                all_aug_stds.append(aug_std)

        stats = {
            'total_images': len(self.drop_stat_data),
            'total_candidates': len(all_iqa_stds),
            'iqa_std_percentiles': {
                '1%': float(np.percentile(all_iqa_stds, 1)),
                '5%': float(np.percentile(all_iqa_stds, 5)),
                '10%': float(np.percentile(all_iqa_stds, 10)),
                '20%': float(np.percentile(all_iqa_stds, 20)),
                '50%': float(np.percentile(all_iqa_stds, 50)),
                '60%': float(np.percentile(all_iqa_stds, 60)),  
                '70%': float(np.percentile(all_iqa_stds, 70)),  
                '80%': float(np.percentile(all_iqa_stds, 80)),  
                '90%': float(np.percentile(all_iqa_stds, 90)),  
                '95%': float(np.percentile(all_iqa_stds, 95)),  
                'mean': float(np.mean(all_iqa_stds)),
            },
            'aug_std_percentiles': {
                '1%': float(np.percentile(all_aug_stds, 1)),
                '5%': float(np.percentile(all_aug_stds, 5)),
                '10%': float(np.percentile(all_aug_stds, 10)),
                '20%': float(np.percentile(all_aug_stds, 20)),
                '50%': float(np.percentile(all_aug_stds, 50)),
                '60%': float(np.percentile(all_aug_stds, 60)),  
                '70%': float(np.percentile(all_aug_stds, 70)),  
                '80%': float(np.percentile(all_aug_stds, 80)),  
                '90%': float(np.percentile(all_aug_stds, 90)),  
                '95%': float(np.percentile(all_aug_stds, 95)),  
                'mean': float(np.mean(all_aug_stds)),
            }
        }

        stats_file = osp.join(save_path, 'drop_statistics.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        np.savez(osp.join(save_path, 'drop_statistics.npz'),
                iqa_stds=all_iqa_stds,
                aug_stds=all_aug_stds)

        print(f"\n{'='*60}")
        print(f"DROP STATISTICS SAVED")
        print(f"{'='*60}")
        print(f"Total images: {stats['total_images']}")
        print(f"Total candidates: {stats['total_candidates']}")
        print(f"\nIQA Std Percentiles:")
        for k, v in stats['iqa_std_percentiles'].items():
            print(f"  {k:>6}: {v:.6f}")
        print(f"\nAug Std Percentiles:")
        for k, v in stats['aug_std_percentiles'].items():
            print(f"  {k:>6}: {v:.6f}")
        print(f"\n💡 Recommended thresholds to drop 10%:")
        print(f"  drop_threshold_multi_iqa: {stats['iqa_std_percentiles']['10%']:.6f}")
        print(f"  drop_threshold_aug_robust: {stats['aug_std_percentiles']['10%']:.6f}")
        print(f"{'='*60}\n")
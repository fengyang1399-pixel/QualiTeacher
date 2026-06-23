import statistics

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.autograd import Variable

from torchvision import transforms

import PIL.Image as Image
import pickle
import os


class Memory_bank(nn.Module):
    def __init__(self):
        super(Memory_bank, self).__init__()

        self.dict={
        }

    def get(self, image_name):
        if image_name in self.dict:
            return self.dict[image_name]
        else:
            return None

    def update(self, image_name, image, transmission, nriqa, clip_score, brisque_score, mask=None):
        pseudo_candidate = [image, transmission, nriqa, clip_score, brisque_score, mask if mask is not None else None]
    
        if image_name not in self.dict:
            self.dict[image_name] = [pseudo_candidate]
        else:
            top3_list = self.dict[image_name]
        
            if len(top3_list) < 3:
                top3_list.append(pseudo_candidate)
            else:
                worst_idx = 0
                worst_combined = float(top3_list[0][2]) + float(top3_list[0][3]) + float(top3_list[0][4])
            
                for idx in range(1, 3):
                    combined = float(top3_list[idx][2]) + float(top3_list[idx][3]) + float(top3_list[idx][4])
                    if combined < worst_combined:
                        worst_combined = combined
                        worst_idx = idx
            
                new_combined = float(nriqa) + float(clip_score) + float(brisque_score)
                if new_combined > worst_combined:
                    top3_list[worst_idx] = pseudo_candidate
        
            top3_list.sort(key=lambda x: float(x[2]) + float(x[3]) + float(x[4]), reverse=True)
            self.dict[image_name] = top3_list

    def save(self, path, current_iter):
        # mkdir path/current_iter
        path = os.path.join(path, "memory_bank", str(current_iter) + '/')
        if not os.path.exists(path):
            os.makedirs(path)

        # save dict to file
        with open(path + 'memory_bank.pkl', 'wb') as f:
            pickle.dump(self.dict, f)

        # save all images in memory bank
        # for key in self.dict.keys():
        #     image = self.dict[key][0]
        #     transforms.ToPILImage()(image).save(path + key)

    def load(self, path):
        # load dict from file
        with open(path, 'rb') as f:
            self.dict = pickle.load(f)
    
    def get_or_update_top3(self, image_name, new_candidates=None):
        cached = self.get(image_name)
    
        if new_candidates is None or len(new_candidates) == 0:
            if cached is None:
                return []  
            else:
                result = []
                for item in cached:
                    result.append({
                        'image': item[0],
                        'transmission': item[1],
                        'musiq': item[2],
                        'clip': item[3],
                        'brisque': item[4],
                        'mask': item[5] if len(item) > 5 else None
                    })
                return result

        all_candidates = []

        if cached is not None:
            for item in cached:
                all_candidates.append({
                    'image': item[0],
                    'transmission': item[1],
                    'musiq': item[2],
                    'clip': item[3],
                    'brisque': item[4],
                    'mask': item[5] if len(item) > 5 else None
                })

        all_candidates.extend(new_candidates)

        all_candidates.sort(
            key=lambda x: float(x['musiq']) + float(x['clip']) + float(x['brisque']), 
            reverse=True
        )

        top3 = all_candidates[:3]

        top3_for_storage = []
        for item in top3:
            top3_for_storage.append([
                item['image'].detach().cpu().clone(),
                item['transmission'].detach().cpu().clone(),
                item['musiq'].detach().cpu().clone() if torch.is_tensor(item['musiq']) else item['musiq'],
                item['clip'].detach().cpu().clone() if torch.is_tensor(item['clip']) else item['clip'],
                item['brisque'].detach().cpu().clone() if torch.is_tensor(item['brisque']) else item['brisque'],
                item['mask'].detach().cpu().clone() if item['mask'] is not None else None
            ])

        self.dict[image_name] = top3_for_storage

        return top3

    @torch.no_grad()
    def forward(self, image_name_list, image_list, transmission_list, nriqa_list, clip_score_list, brisque_score_list, device, mask=None):
        pseudo_label = []
        pseudo_transmission = []
        teacher_nriqa = []
        teacher_clip_score = []
        teacher_brisque_score = []
        mask_stack = []
        for i in range(len(image_name_list)):
            temp_image_data = self.get(image_name_list[i])
            # init
            if temp_image_data is None:
                pseudo_label.append(image_list[i])
                pseudo_transmission.append(transmission_list[i])
                teacher_nriqa.append(nriqa_list[i])
                teacher_clip_score.append(clip_score_list[i])
                teacher_brisque_score.append(brisque_score_list[i])
                if mask is not None:
                    mask_stack.append(mask[i])
                    self.update(image_name_list[i], image_list[i].detach().cpu().clone(), transmission_list[i].detach().cpu().clone(), nriqa_list[i].detach().cpu().clone(), clip_score_list[i].detach().cpu().clone(), brisque_score_list[i].detach().cpu().clone(), mask[i].detach().cpu().clone())
                else:
                    self.update(image_name_list[i], image_list[i].detach().cpu().clone(), transmission_list[i].detach().cpu().clone(), nriqa_list[i].detach().cpu().clone(), clip_score_list[i].detach().cpu().clone(), brisque_score_list[i].detach().cpu().clone())
            # update
            else:
                temp_nriqa = nriqa_list[i].detach().cpu()
                temp_clip_score = clip_score_list[i].detach().cpu()
                old_combined = float(temp_image_data[0][2]) + float(temp_image_data[0][3]) + float(temp_image_data[0][4])
                new_combined = float(temp_nriqa) + float(temp_clip_score) + float(brisque_score_list[i])
                if new_combined > old_combined:
                    pseudo_label.append(image_list[i].to(device))
                    pseudo_transmission.append(transmission_list[i].to(device))
                    teacher_nriqa.append(nriqa_list[i].to(device))
                    teacher_clip_score.append(clip_score_list[i].to(device))
                    teacher_brisque_score.append(brisque_score_list[i].to(device))
                    if mask is not None:
                        mask_stack.append(mask[i].to(device))
                        self.update(image_name_list[i], image_list[i].detach().cpu().clone(), transmission_list[i].detach().cpu().clone(), temp_nriqa.clone(), temp_clip_score.clone(), brisque_score_list[i].detach().cpu().clone(), mask[i].detach().cpu().clone())
                    else:
                        self.update(image_name_list[i], image_list[i].detach().cpu().clone(), transmission_list[i].detach().cpu().clone(), temp_nriqa.clone(), temp_clip_score.clone(), brisque_score_list[i].detach().cpu().clone())
                else:
                    pseudo_label.append(temp_image_data[0][0].to(device))
                    pseudo_transmission.append(temp_image_data[0][1].to(device))
                    teacher_nriqa.append(temp_image_data[0][2].to(device))
                    teacher_clip_score.append(temp_image_data[0][3].to(device))
                    teacher_brisque_score.append(temp_image_data[0][4].to(device))
                    if mask is not None:
                        mask_stack.append(temp_image_data[0][5].to(device))

        if mask is not None:
            return torch.stack(pseudo_label).detach(), torch.stack(pseudo_transmission).detach(), torch.stack(teacher_nriqa).detach(), torch.stack(teacher_clip_score).detach(), torch.stack(teacher_brisque_score).detach(), torch.stack(mask_stack).detach()
        else:
            return torch.stack(pseudo_label).detach(), torch.stack(pseudo_transmission).detach(), torch.stack(teacher_nriqa).detach(), torch.stack(teacher_clip_score).detach(), torch.stack(teacher_brisque_score).detach()


class Memory_bank_woT(nn.Module):
    def __init__(self):
        super(Memory_bank_woT, self).__init__()

        self.dict={
        }

    def get(self, image_name):
        if image_name in self.dict:
            return self.dict[image_name]
        else:
            return None

    def update(self, image_name, image, nriqa, clip_score, mask=None):
        if mask is not None:
            self.dict.update({image_name: [image, nriqa, clip_score, mask]})
        else:
            self.dict.update({image_name: [image, nriqa, clip_score, None]})

    def save(self, path, current_iter):
        # mkdir path/current_iter
        path = os.path.join(path, "memory_bank", str(current_iter) + '/')
        if not os.path.exists(path):
            os.makedirs(path)

        # save dict to file
        with open(path + 'memory_bank.pkl', 'wb') as f:
            pickle.dump(self.dict, f)

        # save all images in memory bank
        for key in self.dict.keys():
            image = self.dict[key][0]
            transforms.ToPILImage()(image).save(path + key)

    def load(self, path):
        # load dict from file
        with open(path, 'rb') as f:
            self.dict = pickle.load(f)

    @torch.no_grad()
    def forward(self, image_name_list, image_list, nriqa_list, clip_score_list, device, mask=None):
        pseudo_label = []
        teacher_nriqa = []
        teacher_clip_score = []
        mask_stack = []
        for i in range(len(image_name_list)):
            temp_image_data = self.get(image_name_list[i])
            if temp_image_data is None:
                pseudo_label.append(image_list[i])
                teacher_nriqa.append(nriqa_list[i])
                teacher_clip_score.append(clip_score_list[i])
                if mask is not None:
                    mask_stack.append(mask[i])
                    self.update(image_name_list[i], image_list[i].detach().cpu().clone(),  nriqa_list[i].detach().cpu().clone(), clip_score_list[i].detach().cpu().clone(), mask[i].detach().cpu().clone())
                else:
                    self.update(image_name_list[i], image_list[i].detach().cpu().clone(),  nriqa_list[i].detach().cpu().clone(), clip_score_list[i].detach().cpu().clone())
            else:
                temp_nriqa = nriqa_list[i].detach().cpu()
                temp_clip_score = clip_score_list[i].detach().cpu()
                if temp_image_data[1] <= temp_nriqa and temp_image_data[2] <= temp_clip_score: # better than previous
                    pseudo_label.append(image_list[i].to(device))
                    teacher_nriqa.append(nriqa_list[i].to(device))
                    teacher_clip_score.append(clip_score_list[i].to(device))
                    if mask is not None:
                        mask_stack.append(mask[i].to(device))
                        self.update(image_name_list[i], image_list[i].detach().cpu().clone(), temp_nriqa.clone(), temp_clip_score.clone(), mask[i].detach().cpu().clone())
                    else:
                        self.update(image_name_list[i], image_list[i].detach().cpu().clone(), temp_nriqa.clone(), temp_clip_score.clone())
                else:
                    pseudo_label.append(temp_image_data[0].to(device))
                    teacher_nriqa.append(temp_image_data[1].to(device))
                    teacher_clip_score.append(temp_image_data[2].to(device))
                    if mask is not None:
                        mask_stack.append(temp_image_data[3].to(device))

        if mask is not None:
            return torch.stack(pseudo_label).detach(), torch.stack(teacher_nriqa).detach(), torch.stack(teacher_clip_score).detach(), torch.stack(mask_stack).detach()
        else:
            return torch.stack(pseudo_label).detach(), torch.stack(teacher_nriqa).detach(), torch.stack(teacher_clip_score).detach()


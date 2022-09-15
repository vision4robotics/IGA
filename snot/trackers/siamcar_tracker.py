from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import cv2

import os
from snot.core.config_car import cfg
from snot.trackers.base_tracker import SiameseTracker
from snot.utils.misc import bbox_clip
from attack_utils_car import adv_attack_search
from data_utils import tensor2img

class SiamCARTracker(SiameseTracker):
    def __init__(self, model, cfg):
        super(SiamCARTracker, self).__init__()
        hanning = np.hanning(cfg.SCORE_SIZE)
        self.window = np.outer(hanning, hanning)
        self.model = model
        self.model.eval()

    def _convert_cls(self, cls):
        cls = F.softmax(cls[:,:,:,:], dim=1).data[:,1,:,:].cpu().numpy()
        return cls

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.model.template(z_crop)

    def change(self,r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        return np.sqrt((w + pad) * (h + pad))

    def cal_penalty(self, lrtbs, penalty_lk):
        bboxes_w = lrtbs[0, :, :] + lrtbs[2, :, :]
        bboxes_h = lrtbs[1, :, :] + lrtbs[3, :, :]
        s_c = self.change(self.sz(bboxes_w, bboxes_h) / self.sz(self.size[0]*self.scale_z, self.size[1]*self.scale_z))
        r_c = self.change((self.size[0] / self.size[1]) / (bboxes_w / bboxes_h))
        penalty = np.exp(-(r_c * s_c - 1) * penalty_lk)
        return penalty

    def accurate_location(self, max_r_up, max_c_up):
        dist = int((cfg.TRACK.INSTANCE_SIZE - (cfg.TRACK.SCORE_SIZE - 1) * 8) / 2)
        max_r_up += dist
        max_c_up += dist
        p_cool_s = np.array([max_r_up, max_c_up])
        disp = p_cool_s - (np.array([cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE]) - 1.) / 2.
        return disp

    def coarse_location(self, hp_score_up, p_score_up, scale_score, lrtbs):
        upsize = (cfg.TRACK.SCORE_SIZE - 1) * cfg.TRACK.STRIDE + 1
        max_r_up_hp, max_c_up_hp = np.unravel_index(hp_score_up.argmax(), hp_score_up.shape)
        max_r = int(round(max_r_up_hp / scale_score))
        max_c = int(round(max_c_up_hp / scale_score))
        max_r = bbox_clip(max_r, 0, cfg.TRACK.SCORE_SIZE)
        max_c = bbox_clip(max_c, 0, cfg.TRACK.SCORE_SIZE)
        bbox_region = lrtbs[max_r, max_c, :]
        min_bbox = int(cfg.TRACK.REGION_S * cfg.TRACK.EXEMPLAR_SIZE)
        max_bbox = int(cfg.TRACK.REGION_L * cfg.TRACK.EXEMPLAR_SIZE)
        l_region = int(min(max_c_up_hp, bbox_clip(bbox_region[0], min_bbox, max_bbox)) / 2.0)
        t_region = int(min(max_r_up_hp, bbox_clip(bbox_region[1], min_bbox, max_bbox)) / 2.0)

        r_region = int(min(upsize - max_c_up_hp, bbox_clip(bbox_region[2], min_bbox, max_bbox)) / 2.0)
        b_region = int(min(upsize - max_r_up_hp, bbox_clip(bbox_region[3], min_bbox, max_bbox)) / 2.0)
        mask = np.zeros_like(p_score_up)
        mask[max_r_up_hp - t_region:max_r_up_hp + b_region + 1, max_c_up_hp - l_region:max_c_up_hp + r_region + 1] = 1
        p_score_up = p_score_up * mask
        return p_score_up

    def getCenter(self,hp_score_up, p_score_up, scale_score,lrtbs):
        # corse location
        score_up = self.coarse_location(hp_score_up, p_score_up, scale_score, lrtbs)
        # accurate location
        max_r_up, max_c_up = np.unravel_index(score_up.argmax(), score_up.shape)
        disp = self.accurate_location(max_r_up,max_c_up)
        disp_ori = disp / self.scale_z
        new_cx = disp_ori[1] + self.center_pos[0]
        new_cy = disp_ori[0] + self.center_pos[1]
        return max_r_up, max_c_up, new_cx, new_cy
    def save_img(self, tensor_clean, tensor_adv, save_path, frame_id):
        ## clean x_crop
        img_clean = tensor2img(tensor_clean)
        cv2.imwrite(os.path.join(save_path, '%04d_clean.jpg' % frame_id), img_clean)
        ## adv x_crop
        img_adv = tensor2img(tensor_adv)
        cv2.imwrite(os.path.join(save_path, '%04d_adv.jpg' % frame_id), img_adv)
        ## diff
        tensor_diff = (tensor_adv - tensor_clean) * 10
        # print(torch.mean(torch.abs(tensor_diff)))
        tensor_diff += 127.0
        img_diff = tensor2img(tensor_diff)
        cv2.imwrite(os.path.join(save_path, '%04d_diff.jpg' % frame_id), img_diff)

    '''supplementary material'''
    def track_supp(self, img, GAN, save_path, frame_id):
        x_crop, scale_z = self.get_x_crop(img)
        '''save clean region and heatmap'''
        x_crop_img = tensor2img(x_crop)
        cv2.imwrite(os.path.join(save_path, 'ori_search_%d.jpg' % frame_id), x_crop_img)
        '''original heatmap'''
        outputs_clean = self.model.track(x_crop)
        score = self._convert_score(outputs_clean['cls']) #(25x25x5,)
        heatmap_clean = 255.0*np.max(score.reshape(5, 25, 25), axis=0)#[0,1]
        heatmap_clean = cv2.resize(heatmap_clean,(255,255),interpolation=cv2.INTER_CUBIC)
        heatmap_clean = cv2.applyColorMap(heatmap_clean.clip(0,255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(save_path, 'heatmap_clean_%d.jpg' % frame_id), heatmap_clean)
        '''Adversarial Attack'''
        x_crop_adv = adv_attack_search(x_crop, GAN)
        output_dict = self.x_crop_2_res(img, x_crop_adv, scale_z)
        '''save adv region and heatmap'''
        x_crop_img_adv = tensor2img(x_crop_adv)
        cv2.imwrite(os.path.join(save_path, 'adv_search_%d.jpg' % frame_id), x_crop_img_adv)
        score_adv = self.score
        heatmap_adv = 255.0 * np.max(score_adv.reshape(5, 25, 25), axis=0)  # [0,1]
        heatmap_adv = cv2.resize(heatmap_adv, (255, 255), interpolation=cv2.INTER_CUBIC)
        heatmap_adv = cv2.applyColorMap(heatmap_adv.clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(save_path, 'heatmap_adv_%d.jpg' % frame_id), heatmap_adv)
        return output_dict

    def track(self, img, hp, GAN, save_path=None, frame_id=None):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        self.scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        x_crop_adv = adv_attack_search(x_crop, GAN)

        outputs = self.model.track(x_crop_adv)

        if save_path!=None and frame_id!=None:
            '''save'''
            self.save_img(x_crop,x_crop_adv,save_path,frame_id)
        cls = self._convert_cls(outputs['cls']).squeeze()
        cen = outputs['cen'].data.cpu().numpy()
        cen = (cen - cen.min()) / cen.ptp()
        cen = cen.squeeze()
        lrtbs = outputs['loc'].data.cpu().numpy().squeeze()

        upsize = (cfg.TRACK.SCORE_SIZE-1) * cfg.TRACK.STRIDE + 1
        penalty = self.cal_penalty(lrtbs, hp['penalty_k'])
        p_score = penalty * cls * cen
        if cfg.TRACK.hanming:
            hp_score = p_score*(1 - hp['window_lr']) + self.window * hp['window_lr']
        else:
            hp_score = p_score

        hp_score_up = cv2.resize(hp_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        p_score_up = cv2.resize(p_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        cls_up = cv2.resize(cls, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        lrtbs = np.transpose(lrtbs,(1,2,0))
        lrtbs_up = cv2.resize(lrtbs, (upsize, upsize), interpolation=cv2.INTER_CUBIC)

        scale_score = upsize / cfg.TRACK.SCORE_SIZE
        # get center
        max_r_up, max_c_up, new_cx, new_cy = self.getCenter(hp_score_up, p_score_up, scale_score, lrtbs)
        # get w h
        ave_w = (lrtbs_up[max_r_up,max_c_up,0] + lrtbs_up[max_r_up,max_c_up,2]) / self.scale_z
        ave_h = (lrtbs_up[max_r_up,max_c_up,1] + lrtbs_up[max_r_up,max_c_up,3]) / self.scale_z

        s_c = self.change(self.sz(ave_w, ave_h) / self.sz(self.size[0]*self.scale_z, self.size[1]*self.scale_z))
        r_c = self.change((self.size[0] / self.size[1]) / (ave_w / ave_h))
        penalty = np.exp(-(r_c * s_c - 1) * hp['penalty_k'])
        lr = penalty * cls_up[max_r_up, max_c_up] * hp['lr']
        new_width = lr*ave_w + (1-lr)*self.size[0]
        new_height = lr*ave_h + (1-lr)*self.size[1]

        # clip boundary
        cx = bbox_clip(new_cx,0,img.shape[1])
        cy = bbox_clip(new_cy,0,img.shape[0])
        width = bbox_clip(new_width,0,img.shape[1])
        height = bbox_clip(new_height,0,img.shape[0])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        return {
                'bbox': bbox,
               }

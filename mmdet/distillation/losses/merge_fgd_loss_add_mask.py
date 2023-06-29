import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import constant_init, kaiming_init
from ..builder import DISTILL_LOSSES
from torch.nn.functional import adaptive_avg_pool2d,normalize,mse_loss
from mmdet.models.roi_heads.standard_roi_head import build_roi_extractor
import numpy as np

@DISTILL_LOSSES.register_module()
class MergeEDLoss(nn.Module):

    """PyTorch version of `Focal and Global Knowledge Distillation for Detectors`
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        temp (float, optional): Temperature coefficient. Defaults to 0.5.
        name (str): the loss name of the layer
        alpha_fgd (float, optional): Weight of fg_loss. Defaults to 0.001
        beta_fgd (float, optional): Weight of bg_loss. Defaults to 0.0005
        gamma_fgd (float, optional): Weight of mask_loss. Defaults to 0.001
        lambda_fgd (float, optional): Weight of relation_loss. Defaults to 0.000005
    """
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 name,
                 temp=0.5,
                 alpha_fgd=0.001,
                 beta_fgd=0.0005,
                 gamma_fgd=0.001,
                 lambda_fgd=0.000005,
                 teacher1_similarity = 0.5,
                 teacher1_center_txt = 'my_output/collected_center/teacher1_centers.txt',
                 teacher2_center_txt = 'my_output/collected_center/teacher1_centers.txt',
                 roi_extractor=dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                     out_channels=256,
                     featmap_strides=[4, 8, 16, 32])
                 ):
        super(MergeEDLoss, self).__init__()
        self.temp = temp
        self.object_fgd = 0.000002
        self.alpha_fgd = alpha_fgd
        self.beta_fgd = beta_fgd
        self.gamma_fgd = gamma_fgd
        self.lambda_fgd = lambda_fgd
        self.teacher1_similarity = teacher1_similarity
        ########################################################################  新加的代码
        def get_centers_from_txt(file_path = 'my_output/collected_center/teacher1_centers.txt'):
            centers_dict = {}
            file_hander = open(file_path,mode='r')
            for line in file_hander.readlines():
                center,class_index = line.strip().split('|')
                centers_dict[class_index.strip()] = torch.tensor(eval(center))
            return centers_dict
        self.teacher1_centers = get_centers_from_txt(teacher1_center_txt)
        self.teacher2_centers = get_centers_from_txt(teacher2_center_txt)
        self.easy_roi_extractor = build_roi_extractor(roi_extractor)
        #######################################################################

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None
        
        self.conv_mask_s = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.conv_mask_t = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.channel_add_conv_s = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels//2, kernel_size=1),
            nn.LayerNorm([teacher_channels//2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels//2, teacher_channels, kernel_size=1))
        self.channel_add_conv_t = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels//2, kernel_size=1),
            nn.LayerNorm([teacher_channels//2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels//2, teacher_channels, kernel_size=1))

        self.reset_parameters()


    def forward(self,
                preds_S1,
                preds_S2,
                preds_T1,
                preds_T2,
                gt_bboxes,
                gt_labels,
                teacher1_fpn,
                teacher2_fpn,
                teacher1_result,
                teacher2_result,
                img_metas):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T1(Tensor): Bs*C*H*W, teacher1's feature map
            preds_T2(Tensor): Bs*C*H*W, teacher2's feature map
            gt_bboxes(tuple): Bs*[nt*4], pixel decimal: (tl_x, tl_y, br_x, br_y)
            img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        """
        # print('mmdet/distillation/losses/merge_fgd_loss_add_mask.py---------------------------------------------------------------forward')
        # # print(teacher1_result)
        # # print('teacher1_result-----------------------------------------------------------------------------------------------------------')
        # # print(teacher1_result[0])
        # print('teacher1_result-----------------------------------------------------------------------------------------------------------')
        # print(teacher1_result[0][0])
        # print('teacher1_result-----------------------------------------------------------------------------------------------------------')
        # # print(teacher2_result)
        # # print('teacher2_result-----------------------------------------------------------------------------------------------------------')
        # # print(teacher2_result[0])
        # print('teacher2_result-----------------------------------------------------------------------------------------------------------')
        # print(teacher2_result[0][0][0])
        # print('mmdet/distillation/losses/merge_fgd_loss_add_mask.py---------------------------------------------------------------forward')
        assert preds_S1.shape[-2:] == preds_T1.shape[-2:],          'the output dim of teacher and student differ'
        if self.align is not None:
            preds_S1 = self.align(preds_S1)
            preds_S2 = self.align(preds_S2)

        N,C,H,W = preds_S1.shape

        S_attention_t1, C_attention_t1 = self.get_attention(preds_T1, self.temp)
        S_attention_t2, C_attention_t2 = self.get_attention(preds_T2, self.temp)
        S_attention_s1, C_attention_s1 = self.get_attention(preds_S1, self.temp)
        S_attention_s2, C_attention_s2 = self.get_attention(preds_S2, self.temp)

        Mask_fg1 = torch.zeros_like(S_attention_t1)
        Mask_fg2 = torch.zeros_like(S_attention_t2)
        Mask_bg1 = torch.ones_like(S_attention_t1)
        Mask_bg2 = torch.ones_like(S_attention_t2)
        wmin,wmax,hmin,hmax = [],[],[],[]
        for i in range(N):
            # print('**' * 50)
            # print(gt_bboxes[i])
            # print('..'*50)
            # print(gt_labels[i])
            # print('--' * 50)
            # print(img_metas[i]['img_shape'])
            new_boxxes = torch.ones_like(gt_bboxes[i])
            new_boxxes[:, 0] = gt_bboxes[i][:, 0]/img_metas[i]['img_shape'][1]*W
            new_boxxes[:, 2] = gt_bboxes[i][:, 2]/img_metas[i]['img_shape'][1]*W
            new_boxxes[:, 1] = gt_bboxes[i][:, 1]/img_metas[i]['img_shape'][0]*H
            new_boxxes[:, 3] = gt_bboxes[i][:, 3]/img_metas[i]['img_shape'][0]*H

            wmin.append(torch.floor(new_boxxes[:, 0]).int())
            wmax.append(torch.ceil(new_boxxes[:, 2]).int())
            hmin.append(torch.floor(new_boxxes[:, 1]).int())
            hmax.append(torch.ceil(new_boxxes[:, 3]).int())

            area = 1.0/(hmax[i].view(1,-1)+1-hmin[i].view(1,-1))/(wmax[i].view(1,-1)+1-wmin[i].view(1,-1))

            for j in range(len(gt_bboxes[i])):
                Mask_fg1[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1] = \
                        torch.maximum(Mask_fg1[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1], area[0][j])

            Mask_bg1[i] = torch.where(Mask_fg1[i]>0, 0, 1)
            if torch.sum(Mask_bg1[i]):
                Mask_bg1[i] /= torch.sum(Mask_bg1[i])
        Mask_fg2 = Mask_fg1
        Mask_bg2 = Mask_bg1
        ################################################################################################################ 新计算的两个mask，来自teacher1和teacher2
        ###################################################################################
        '''计算两个教师对应的边界区域mask，中心点距离mask'''
        assert teacher1_fpn[0].shape[0] == len(teacher1_result)
        new_scaled_gt_bboxes = []
        gt_bboxes_padding = []
        N, C, H, W = preds_S1.shape
        teacher1_edge_mask = torch.ones([N, H, W], dtype=teacher1_fpn[0].dtype) * 4
        teacher2_edge_mask = torch.ones_like(teacher1_edge_mask) * 4
        teacher1_distance_mask = torch.ones_like(teacher1_edge_mask)
        teacher2_distance_mask = torch.ones_like(teacher1_edge_mask)
        ###########       4个重要的mask，分别是教师1和教师2的proposal叠加mask和离中心点距离的mask
        for i in range(N):
            new_boxxes = torch.ones_like(gt_bboxes[i])
            new_boxxes[:, 0] = torch.floor(gt_bboxes[i][:, 0] / img_metas[i]['img_shape'][1] * W)
            new_boxxes[:, 2] = torch.ceil(gt_bboxes[i][:, 2] / img_metas[i]['img_shape'][1] * W)
            new_boxxes[:, 1] = torch.floor(gt_bboxes[i][:, 1] / img_metas[i]['img_shape'][0] * H)
            new_boxxes[:, 3] = torch.ceil(gt_bboxes[i][:, 3] / img_metas[i]['img_shape'][0] * H)
            new_scaled_gt_bboxes.append(new_boxxes)
            constandpadding = nn.ConstantPad2d(padding=(1, 0, 0, 0), value=i)  # 加标签，其中value表示这个截出来的roi是属于哪张图片的
            new_boxxes_padding = constandpadding(new_boxxes)
            gt_bboxes_padding = new_boxxes_padding if i == 0 else torch.cat((gt_bboxes_padding, new_boxxes_padding),
                                                                            dim=0)  # 判断，如果是第一张图就直接赋值，第二张图就开始拼接
            gt_labels_add = gt_labels[i] if i == 0 else torch.cat((gt_labels_add, gt_labels[i]), dim=0)
            for j in range(len(teacher1_result[i])):
                ####teacher1
                if teacher1_result[i][j].shape[0] != 0:
                    teacher1_new_class_propsal_bboxes = teacher1_result[i][j]
                    teacher1_new_class_propsal_bboxes[:, 0] = np.floor(
                        teacher1_new_class_propsal_bboxes[:, 0] / img_metas[i]['img_shape'][1] * W)
                    teacher1_new_class_propsal_bboxes[:, 2] = np.ceil(
                        teacher1_new_class_propsal_bboxes[:, 2] / img_metas[i]['img_shape'][1] * W)
                    teacher1_new_class_propsal_bboxes[:, 1] = np.floor(
                        teacher1_new_class_propsal_bboxes[:, 1] / img_metas[i]['img_shape'][0] * H)
                    teacher1_new_class_propsal_bboxes[:, 3] = np.ceil(
                        teacher1_new_class_propsal_bboxes[:, 3] / img_metas[i]['img_shape'][0] * H)
                    for k in range(teacher1_result[i][j].shape[0]):
                        if teacher1_new_class_propsal_bboxes[k, 4] > 0.2:
                            teacher1_edge_mask[i,
                            teacher1_new_class_propsal_bboxes[k, 1].astype(int):teacher1_new_class_propsal_bboxes[k, 3].astype(int),
                            teacher1_new_class_propsal_bboxes[k, 0].astype(int):teacher1_new_class_propsal_bboxes[k, 2].astype(int)] += teacher1_new_class_propsal_bboxes[k, 4] * 2  # 权重赋值，赋值方式为叠加，越是重合的地方权重越大
                        else:
                            break
                        pass
                    pass
                else:
                    pass
                ####teacher2
                # print('teacher2_result------------------------------------------')
                # print(len(teacher2_result),type(teacher2_result))
                # print(len(teacher2_result[0]),len(teacher2_result[0]))
                # print(teacher2_result[0][0])
                # print(teacher2_result)
                if teacher2_result[i][0][j].shape[0] != 0:
                    teacher2_new_class_propsal_bboxes = teacher2_result[i][0][j]
                    teacher2_new_class_propsal_bboxes[:, 0] = np.floor(
                        teacher2_new_class_propsal_bboxes[:, 0] / img_metas[i]['img_shape'][1] * W)
                    teacher2_new_class_propsal_bboxes[:, 2] = np.ceil(
                        teacher2_new_class_propsal_bboxes[:, 2] / img_metas[i]['img_shape'][1] * W)
                    teacher2_new_class_propsal_bboxes[:, 1] = np.floor(
                        teacher2_new_class_propsal_bboxes[:, 1] / img_metas[i]['img_shape'][0] * H)
                    teacher2_new_class_propsal_bboxes[:, 3] = np.ceil(
                        teacher2_new_class_propsal_bboxes[:, 3] / img_metas[i]['img_shape'][0] * H)
                    for k in range(teacher2_result[i][0][j].shape[0]):
                        if teacher2_new_class_propsal_bboxes[k, 4] > 0.2:
                            teacher2_edge_mask[i,
                            teacher2_new_class_propsal_bboxes[k, 1].astype(int):teacher2_new_class_propsal_bboxes[k, 3].astype(int),
                            teacher2_new_class_propsal_bboxes[k, 0].astype(int):teacher2_new_class_propsal_bboxes[k, 2].astype(int)] += \
                            teacher2_new_class_propsal_bboxes[k, 4] * 2
                        else:
                            break
                        pass
                    pass
                else:
                    pass

                pass
            pass
        teacher1_bbox_feats = self.easy_roi_extractor(teacher1_fpn[:4], gt_bboxes_padding)
        teacher2_bbox_feats = self.easy_roi_extractor(teacher2_fpn[:4], gt_bboxes_padding)
        teacher1_feature_C = adaptive_avg_pool2d(teacher1_bbox_feats, (1, 1))  # shape(Batch,256,1,1)
        teacher2_feature_C = adaptive_avg_pool2d(teacher2_bbox_feats, (1, 1))  # shape(Batch,256,1,1)
        _, teacher1_feature_C_top20_index = torch.topk(teacher1_feature_C, 20, dim=1, sorted=False)
        _, teacher2_feature_C_top20_index = torch.topk(teacher2_feature_C, 20, dim=1, sorted=False)
        N_roi, C_roi, H_roi, W_roi = teacher1_bbox_feats.shape
        teacher1_bbox_feats_top20 = torch.zeros([N_roi, 20, H_roi, W_roi], dtype=teacher1_bbox_feats.dtype)
        teacher2_bbox_feats_top20 = torch.zeros_like(teacher1_bbox_feats_top20)
        for i in range(N):
            teacher1_bbox_feats_top20[i, :, :, :] = teacher1_bbox_feats[i,
                                                    teacher1_feature_C_top20_index[i, :].squeeze(), :, :]
            teacher2_bbox_feats_top20[i, :, :, :] = teacher2_bbox_feats[i,
                                                    teacher2_feature_C_top20_index[i, :].squeeze(), :, :]
            pass
        # print(teacher1_bbox_feats.shape,'----------------------teacher1_bbox_feats.shape')
        # print(teacher1_bbox_feats_top20.shape,'teacher1_bbox_feats_top20.shape,,,,,,,shape(batch,20,7,7)')
        # print(teacher1_feature_C_top20_index.squeeze().shape,'teacher1_feature_C_top20_index.squeeze().shape--------------------',teacher1_feature_C_top20_index.squeeze())
        teacher1_bbox_feats_top20_line = torch.flatten(torch.mean(teacher1_bbox_feats_top20, dim=1), 1,
                                                       -1)  # shape(batch,20,7,7)  -> (batch,49)
        teacher2_bbox_feats_top20_line = torch.flatten(torch.mean(teacher2_bbox_feats_top20, dim=1), 1,
                                                       -1)  # shape(batch,20,7,7)  -> (batch,49)
        # print(teacher1_bbox_feats_top20_line.shape,'teacher1_bbox_feats_top20_line------------#shape(batch,20,7,7)  -> (batch,49)')
        teacher1_bbox_feats_top20_line_normal = normalize(teacher1_bbox_feats_top20_line, p=2, dim=-1)
        teacher2_bbox_feats_top20_line_normal = normalize(teacher2_bbox_feats_top20_line, p=2, dim=-1)
        index_count = 0
        for i in range(N):
            for j in range(new_scaled_gt_bboxes[i].shape[0]):
                object_class = gt_labels[i][j].tolist()
                # print('---------------------------------------------------------------self.teacher1_centers[str(object_class)]')
                # print(object_class)
                # print(teacher1_bbox_feats_top20_line_normal[index_count, :].shape,teacher1_bbox_feats_top20_line_normal[index_count, :])
                # print(self.teacher1_centers[str(object_class)].shape,self.teacher1_centers[str(object_class)])
                # print(self.teacher1_centers)
                # print('---------------------------------------------------------------self.teacher1_centers[str(object_class)]')
                teacher1_distance = mse_loss(teacher1_bbox_feats_top20_line_normal[index_count, :],self.teacher1_centers[str(object_class)])
                teacher2_distance = mse_loss(teacher2_bbox_feats_top20_line_normal[index_count, :],self.teacher1_centers[str(object_class)])
                teacher1_distance_mask[i, new_scaled_gt_bboxes[i][j, 1].int():new_scaled_gt_bboxes[i][j, 3].int()] = torch.where(
                    teacher1_distance_mask[i,new_scaled_gt_bboxes[i][j, 1].int():new_scaled_gt_bboxes[i][j, 3].int()] < 1 / teacher1_distance,
                    teacher1_distance_mask[i, new_scaled_gt_bboxes[i][j, 1].int():new_scaled_gt_bboxes[i][j, 3].int()],
                    1 / teacher1_distance)
                teacher2_distance_mask[i, new_scaled_gt_bboxes[i][j, 1].int():new_scaled_gt_bboxes[i][j, 3].int()] = torch.where(
                    teacher2_distance_mask[i,new_scaled_gt_bboxes[i][j, 1].int():new_scaled_gt_bboxes[i][j, 3].int()] < 1 / teacher1_distance,
                    teacher2_distance_mask[i, new_scaled_gt_bboxes[i][j, 1].int():new_scaled_gt_bboxes[i][j, 3].int()],
                    1 / teacher2_distance)
                index_count += 1
                pass
            pass
        ################################################################################################################
        fg_loss1, bg_loss1 = self.get_fea_loss(preds_S1, preds_T1, Mask_fg1, Mask_bg1,
                           C_attention_s1, C_attention_t1, S_attention_s1, S_attention_t1)

        mask_loss1 = self.get_mask_loss(C_attention_s1, C_attention_t1, S_attention_s1, S_attention_t1)
        rela_loss1 = self.get_rela_loss(preds_S1, preds_T1)

        fg_loss2, bg_loss2 = self.get_fea_loss(preds_S2, preds_T2, Mask_fg2, Mask_bg2,
                                               C_attention_s2, C_attention_t2, S_attention_s2, S_attention_t2)
        mask_loss2 = self.get_mask_loss(C_attention_s2, C_attention_t2, S_attention_s2, S_attention_t2)
        rela_loss2 = self.get_rela_loss(preds_S2, preds_T2)

        fg_loss = fg_loss1 * self.teacher1_similarity + fg_loss2 * (1 - self.teacher1_similarity)
        bg_loss = bg_loss1 * self.teacher1_similarity + bg_loss2 * (1 - self.teacher1_similarity)
        mask_loss = mask_loss1 * self.teacher1_similarity + mask_loss2 * (1 - self.teacher1_similarity)
        rela_loss = rela_loss1 * self.teacher1_similarity + rela_loss2 * (1 - self.teacher1_similarity)
########################################################################################################
        object_loss1 = self.get_object_mask_loss(preds_S1, preds_T1, teacher1_edge_mask, teacher1_distance_mask,C_attention_s1, C_attention_t1, S_attention_s1, S_attention_t1)
        object_loss2 = self.get_object_mask_loss(preds_S1, preds_T1, teacher2_edge_mask, teacher2_distance_mask,C_attention_s1, C_attention_t1, S_attention_s1, S_attention_t1)
        object_loss = object_loss1 * self.teacher1_similarity + object_loss2 * (1-self.teacher1_similarity)
########################################################################################################
        loss = self.alpha_fgd * fg_loss + self.beta_fgd * bg_loss \
               + self.gamma_fgd * mask_loss + self.lambda_fgd * rela_loss
        object_loss =  self.object_fgd * object_loss
        return loss , object_loss


    def get_attention(self, preds, temp):
        """ preds: Bs*C*W*H """
        N, C, H, W= preds.shape

        value = torch.abs(preds)
        # Bs*W*H
        fea_map = value.mean(axis=1, keepdim=True)
        S_attention = (H * W * F.softmax((fea_map/temp).view(N,-1), dim=1)).view(N, H, W)

        # Bs*C
        channel_map = value.mean(axis=2,keepdim=False).mean(axis=2,keepdim=False)
        C_attention = C * F.softmax(channel_map/temp, dim=1)

        return S_attention, C_attention


    def get_fea_loss(self, preds_S, preds_T1, Mask_fg, Mask_bg, C_s, C_t, S_s, S_t):
        loss_mse = nn.MSELoss(reduction='sum')
        
        Mask_fg = Mask_fg.unsqueeze(dim=1)
        Mask_bg = Mask_bg.unsqueeze(dim=1)

        C_t = C_t.unsqueeze(dim=-1)
        C_t = C_t.unsqueeze(dim=-1)

        S_t = S_t.unsqueeze(dim=1)

        fea_t= torch.mul(preds_T1, torch.sqrt(S_t))
        fea_t = torch.mul(fea_t, torch.sqrt(C_t))
        fg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_fg))
        bg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_bg))

        fea_s = torch.mul(preds_S, torch.sqrt(S_t))
        fea_s = torch.mul(fea_s, torch.sqrt(C_t))
        fg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_fg))
        bg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_bg))

        fg_loss = loss_mse(fg_fea_s, fg_fea_t)/len(Mask_fg)
        bg_loss = loss_mse(bg_fea_s, bg_fea_t)/len(Mask_bg)

        return fg_loss, bg_loss
###################################################################################################
    def get_object_mask_loss(self, preds_S, preds_T, Mask_edge, Mask_diastance, C_s, C_t, S_s, S_t):
        preds_T = preds_T.cuda()
        Mask_edge = Mask_edge.cuda()
        Mask_diastance = Mask_diastance.cuda()
        loss_mse = nn.MSELoss(reduction='sum')

        Mask_edge = Mask_edge.unsqueeze(dim=1)
        Mask_diastance= Mask_diastance.unsqueeze(dim=1)

        C_t = C_t.unsqueeze(dim=-1)
        C_t = C_t.unsqueeze(dim=-1)

        S_t = S_t.unsqueeze(dim=1)

        fea_t = torch.mul(preds_T, torch.sqrt(S_t))
        fea_t = torch.mul(fea_t, torch.sqrt(C_t))
        edge_fea_t = torch.mul(fea_t, torch.sqrt(Mask_edge))
        diastance_fea_t = torch.mul(fea_t, torch.sqrt(Mask_diastance))

        fea_s = torch.mul(preds_S, torch.sqrt(S_t))
        fea_s = torch.mul(fea_s, torch.sqrt(C_t))
        edge_fea_s = torch.mul(fea_s, torch.sqrt(Mask_edge))
        diastance_fea_s = torch.mul(fea_s, torch.sqrt(Mask_diastance))

        object_loss = loss_mse(diastance_fea_t, diastance_fea_s) / len(Mask_diastance)

        return object_loss
###################################################################################################

    def get_mask_loss(self, C_s, C_t, S_s, S_t):

        mask_loss = torch.sum(torch.abs((C_s-C_t)))/len(C_s) + torch.sum(torch.abs((S_s-S_t)))/len(S_s)

        return mask_loss
     
    
    def spatial_pool(self, x, in_type):
        batch, channel, width, height = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        if in_type == 0:
            context_mask = self.conv_mask_s(x)
        else:
            context_mask = self.conv_mask_t(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = F.softmax(context_mask, dim=2)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context


    def get_rela_loss(self, preds_S, preds_T1):
        loss_mse = nn.MSELoss(reduction='sum')

        context_s = self.spatial_pool(preds_S, 0)
        context_t = self.spatial_pool(preds_T1, 1)

        out_s = preds_S
        out_t = preds_T1

        channel_add_s = self.channel_add_conv_s(context_s)
        out_s = out_s + channel_add_s

        channel_add_t = self.channel_add_conv_t(context_t)
        out_t = out_t + channel_add_t

        rela_loss = loss_mse(out_s, out_t)/len(out_s)
        
        return rela_loss


    def last_zero_init(self, m):
        if isinstance(m, nn.Sequential):
            constant_init(m[-1], val=0)
        else:
            constant_init(m, val=0)

    
    def reset_parameters(self):
        kaiming_init(self.conv_mask_s, mode='fan_in')
        kaiming_init(self.conv_mask_t, mode='fan_in')
        self.conv_mask_s.inited = True
        self.conv_mask_t.inited = True

        self.last_zero_init(self.channel_add_conv_s)
        self.last_zero_init(self.channel_add_conv_t)
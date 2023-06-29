import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import constant_init, kaiming_init
from ..builder import DISTILL_LOSSES


@DISTILL_LOSSES.register_module()
class FeatureLoss_small(nn.Module):
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
                 ):
        super(FeatureLoss_small, self).__init__()
        self.temp = temp
        self.alpha_fgd = alpha_fgd
        self.beta_fgd = beta_fgd
        self.gamma_fgd = gamma_fgd
        self.lambda_fgd = lambda_fgd

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        self.conv_mask_s = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.conv_mask_t = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.channel_add_conv_s = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels // 2, kernel_size=1),
            nn.LayerNorm([teacher_channels // 2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels // 2, teacher_channels, kernel_size=1))
        self.channel_add_conv_t = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels // 2, kernel_size=1),
            nn.LayerNorm([teacher_channels // 2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels // 2, teacher_channels, kernel_size=1))

        self.reset_parameters()

    def forward(self,
                preds_S,
                preds_T,
                gt_bboxes,
                gt_labels,
                img_metas):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
            gt_bboxes(tuple): Bs*[nt*4], pixel decimal: (tl_x, tl_y, br_x, br_y)
            img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:], 'the output dim of teacher and student differ'
        if self.align is not None:
            preds_S = self.align(preds_S)

        N, C, H, W = preds_S.shape

        S_attention_t, C_attention_t = self.get_attention(preds_T, self.temp)
        S_attention_s, C_attention_s = self.get_attention(preds_S, self.temp)

        Mask_fg = torch.zeros_like(S_attention_t)
        Mask_bg = torch.ones_like(S_attention_t)
        wmin, wmax, hmin, hmax = [], [], [], []
        for i in range(N):
            # print('**' * 50)
            # print(gt_bboxes[i])
            # print('..'*50)
            # print(gt_labels[i])
            # print('--' * 50)
            # print(img_metas[i]['img_shape'])
            new_boxxes = torch.ones_like(gt_bboxes[i])
            new_boxxes[:, 0] = gt_bboxes[i][:, 0] / img_metas[i]['img_shape'][1] * W
            new_boxxes[:, 2] = gt_bboxes[i][:, 2] / img_metas[i]['img_shape'][1] * W
            new_boxxes[:, 1] = gt_bboxes[i][:, 1] / img_metas[i]['img_shape'][0] * H
            new_boxxes[:, 3] = gt_bboxes[i][:, 3] / img_metas[i]['img_shape'][0] * H

            wmin.append(torch.floor(new_boxxes[:, 0]).int())
            wmax.append(torch.ceil(new_boxxes[:, 2]).int())
            hmin.append(torch.floor(new_boxxes[:, 1]).int())
            hmax.append(torch.ceil(new_boxxes[:, 3]).int())

            area = 1.0 / (hmax[i].view(1, -1) + 1 - hmin[i].view(1, -1)) / (
                        wmax[i].view(1, -1) + 1 - wmin[i].view(1, -1))

            for j in range(len(gt_bboxes[i])):
                Mask_fg[i][hmin[i][j]:hmax[i][j] + 1, wmin[i][j]:wmax[i][j] + 1] = \
                    torch.maximum(Mask_fg[i][hmin[i][j]:hmax[i][j] + 1, wmin[i][j]:wmax[i][j] + 1], area[0][j])

            Mask_bg[i] = torch.where(Mask_fg[i] > 0, 0, 1)
            if torch.sum(Mask_bg[i]):
                Mask_bg[i] /= torch.sum(Mask_bg[i])

        fg_loss, bg_loss = self.get_fea_loss(preds_S, preds_T, Mask_fg, Mask_bg,
                                             C_attention_s, C_attention_t, S_attention_s, S_attention_t)
        # mask_loss = self.get_mask_loss(C_attention_s, C_attention_t, S_attention_s, S_attention_t)
        # rela_loss = self.get_rela_loss(preds_S, preds_T)

        loss = self.alpha_fgd * fg_loss + self.beta_fgd * bg_loss

        return loss

    def get_attention(self, preds, temp):
        """ preds: Bs*C*W*H """
        N, C, H, W = preds.shape

        value = torch.abs(preds)
        # Bs*W*H
        fea_map = value.mean(axis=1, keepdim=True)
        S_attention = (H * W * F.softmax((fea_map / temp).view(N, -1), dim=1)).view(N, H, W)

        # Bs*C
        channel_map = value.mean(axis=2, keepdim=False).mean(axis=2, keepdim=False)
        C_attention = C * F.softmax(channel_map / temp, dim=1)

        return S_attention, C_attention

    def get_fea_loss(self, preds_S, preds_T, Mask_fg, Mask_bg, C_s, C_t, S_s, S_t):
        loss_mse = nn.MSELoss(reduction='sum')

        Mask_fg = Mask_fg.unsqueeze(dim=1)
        Mask_bg = Mask_bg.unsqueeze(dim=1)

        C_t = C_t.unsqueeze(dim=-1)
        C_t = C_t.unsqueeze(dim=-1)

        S_t = S_t.unsqueeze(dim=1)

        # print('---------------------------------------shape of channel attention', C_t.shape, 'shape of spicial attention', S_t.shape, 'shape of preds_S',
        #       preds_S.shape, 'shape of mask_fg', Mask_fg.shape)

        fea_t = torch.mul(preds_T, torch.sqrt(S_t))
        fea_t = torch.mul(fea_t, torch.sqrt(C_t))
        fg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_fg))
        bg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_bg))

        fea_s = torch.mul(preds_S, torch.sqrt(S_t))
        fea_s = torch.mul(fea_s, torch.sqrt(C_t))
        fg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_fg))
        bg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_bg))

        fg_loss = loss_mse(fg_fea_s, fg_fea_t) / len(Mask_fg)
        bg_loss = loss_mse(bg_fea_s, bg_fea_t) / len(Mask_bg)

        return fg_loss, bg_loss

    def get_mask_loss(self, C_s, C_t, S_s, S_t):

        mask_loss = torch.sum(torch.abs((C_s - C_t))) / len(C_s) + torch.sum(torch.abs((S_s - S_t))) / len(S_s)

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

    def get_rela_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')

        context_s = self.spatial_pool(preds_S, 0)
        context_t = self.spatial_pool(preds_T, 1)

        out_s = preds_S
        out_t = preds_T

        channel_add_s = self.channel_add_conv_s(context_s)
        out_s = out_s + channel_add_s

        channel_add_t = self.channel_add_conv_t(context_t)
        out_t = out_t + channel_add_t

        rela_loss = loss_mse(out_s, out_t) / len(out_s)

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



@DISTILL_LOSSES.register_module()
class DecoderFeatureLoss(nn.Module):
    def __init__(self,ratio=0.5,name=None,temp=0.5):
        super(DecoderFeatureLoss, self).__init__()
        self.ratio = ratio
        self.name = name
        self.temp = temp

        self.L1_loss = nn.L1Loss(reduction='mean')
        self.loss_mse = nn.MSELoss(reduction='mean')


    def forward(self,pred_S,pred_T,gt_bboxes,gt_labels,img_metas):
        # feature_loss = torch.cosine_similarity(pred_S,pred_T, dim=0).mean()

        C_attention_T = self.get_attention(pred_T,self.temp)
        C_attention_S = self.get_attention(pred_S,self.temp)
        # print('----------------the shape of detr decoder is ', pred_S.shape,'the C attention_shape is  ', C_attention_T.shape)
        C_attention_T = C_attention_T.unsqueeze(-1)
        fea_s = torch.mul(pred_S, torch.sqrt(C_attention_T))
        fea_t = torch.mul(pred_T, torch.sqrt(C_attention_T))

        # loss = self.L1_loss(fea_s,fea_t)
        loss = self.loss_mse(fea_s,fea_t)
        return loss *self.ratio

    def get_attention(self, preds, temp):
        """ preds: Bs*C*W*H """
        querys,Bs,C = preds.shape

        value = torch.abs(preds)
        # query*Bs
        channel_map = value.mean(axis=2, keepdim=False)

        C_attention = querys * F.softmax(channel_map / temp, dim=1)
        # print('decoder C_attention shape',C_attention.shape,'mean shape',channel_map.shape)
        return C_attention

@DISTILL_LOSSES.register_module()
class EncoderFeatureLoss(nn.Module):
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
                 name=None,
                 ratio = 1,
                 temp=0.5,
                 alpha_fgd=0.001,
                 beta_fgd=0.0005,
                 gamma_fgd=0.001,
                 lambda_fgd=0.000005,
                 ):
        super(EncoderFeatureLoss, self).__init__()
        self.temp = temp
        self.alpha_fgd = alpha_fgd
        self.beta_fgd = beta_fgd
        self.gamma_fgd = gamma_fgd
        self.lambda_fgd = lambda_fgd
        self.student_chanenel = student_channels
        self.ratio = ratio
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        self.conv_mask_s = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.conv_mask_t = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.channel_add_conv_s = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels // 2, kernel_size=1),
            nn.LayerNorm([teacher_channels // 2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels // 2, teacher_channels, kernel_size=1))
        self.channel_add_conv_t = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels // 2, kernel_size=1),
            nn.LayerNorm([teacher_channels // 2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels // 2, teacher_channels, kernel_size=1))

        self.reset_parameters()

    def forward(self,
                preds_S,
                preds_T,
                gt_bboxes,
                img_metas,
                student_feature_size):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
            gt_bboxes(tuple): Bs*[nt*4], pixel decimal: (tl_x, tl_y, br_x, br_y)
            img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        """
        # assert preds_S.shape[-2:] == preds_T.shape[-2:], 'the output dim of teacher and student differ'

        if self.align is not None:
            preds_S = preds_S.permute(1,2,0)
            preds_S = self.align(preds_S)
            preds_S = preds_S.permute(2,0,1)

        # print('------in encoder loss,-----student_feature_size',student_feature_size)
        N,C,H,W = student_feature_size
        # print('------in encoder loss,pred_s shape',preds_S.shape,'------in encoder loss,-pred_t shape ',preds_T.shape)
        HW,bs,c = preds_S.shape
        preds_S_new = preds_S.clone().view(H,W,N,c).permute(2,3,0,1)
        # print('here the value of encoder feature:',preds_S_new)
        preds_T_new = preds_T.detach().view(H,W,N,c).permute(2,3,0,1)
        S_attention_t, C_attention_t = self.get_attention(preds_T_new, self.temp)
        S_attention_s, C_attention_s = self.get_attention(preds_S_new, self.temp)

        Mask_fg = torch.zeros_like(S_attention_t,requires_grad=False,device='cuda')
        Mask_bg = torch.ones_like(S_attention_t,requires_grad=False,device='cuda')

        wmin, wmax, hmin, hmax = [], [], [], []
        # print('------in encoder loss,--------star generate mask')
        # for i in range(N):
        #     new_boxxes = torch.ones_like(gt_bboxes[i])
        #     new_boxxes[:, 0] = gt_bboxes[i][:, 0] / img_metas[i]['img_shape'][1] * W
        #     new_boxxes[:, 2] = gt_bboxes[i][:, 2] / img_metas[i]['img_shape'][1] * W
        #     new_boxxes[:, 1] = gt_bboxes[i][:, 1] / img_metas[i]['img_shape'][0] * H
        #     new_boxxes[:, 3] = gt_bboxes[i][:, 3] / img_metas[i]['img_shape'][0] * H
        #
        #     wmin.append(torch.floor(new_boxxes[:, 0]).int())
        #     wmax.append(torch.ceil(new_boxxes[:, 2]).int())
        #     hmin.append(torch.floor(new_boxxes[:, 1]).int())
        #     hmax.append(torch.ceil(new_boxxes[:, 3]).int())
        #
        #     area = 1.0 / (hmax[i].view(1, -1) + 1 - hmin[i].view(1, -1)) / (
        #             wmax[i].view(1, -1) + 1 - wmin[i].view(1, -1))
        #     for j in range(len(gt_bboxes[i])):
        #         fg_mask[i][hmin[i][j]+2:hmax[i][j] - 3, wmin[i][j]+2:wmax[i][j] - 3] = 1
        #             # torch.maximum(fg_mask[i][hmin[i][j]:hmax[i][j] + 1, wmin[i][j]:wmax[i][j] + 1], area[0][j])
        #         edge_mask[i][((hmin[i][j] - 2)if (hmin[i][j] - 2)>= 0 else 0):(hmax[i][j] + 2) if (hmax[i][j] + 2)<H else H ,
        #         ((wmin[i][j] - 2)if (wmin[i][j] - 2)>= 0 else 0):(wmax[i][j] + 2) if (wmax[i][j] + 2)<H else W ] = 1
        # # print('------in encoder loss,-  ----------------------------------------mask over')
        # edge_circle_mask = edge_mask - fg_mask
        # bg_mask = 1-edge_mask
        #
        # fg_mask = fg_mask.flatten(2).permute(2, 0, 1)
        # edge_circle_mask = edge_circle_mask.flatten(2).permute(2, 0, 1)
        # bg_mask = bg_mask.flatten(2).permute(2, 0, 1)
        # # print('-----------------------pred_s shape',preds_S.shape,'-----------------------pred t shape',preds_T.shape)
        # # print('fg mask-------------------',fg_mask.shape,'edge mask----------------------------',edge_circle_mask.shape,'bg mask---------------------',bg_mask.shape)
        # loss = self.get_queue_feature_loss(fg_mask,edge_circle_mask,bg_mask,preds_S,preds_T)
        # return loss
##################################################################################################
        for i in range(N):
            # print('**' * 50)
            # print(gt_bboxes[i])
            # print('..'*50)
            # print(gt_labels[i])
            # print('--' * 50)
            # print(img_metas[i]['img_shape'])
            new_boxxes = torch.ones_like(gt_bboxes[i])
            new_boxxes[:, 0] = gt_bboxes[i][:, 0] / img_metas[i]['img_shape'][1] * W
            new_boxxes[:, 2] = gt_bboxes[i][:, 2] / img_metas[i]['img_shape'][1] * W
            new_boxxes[:, 1] = gt_bboxes[i][:, 1] / img_metas[i]['img_shape'][0] * H
            new_boxxes[:, 3] = gt_bboxes[i][:, 3] / img_metas[i]['img_shape'][0] * H

            wmin.append(torch.floor(new_boxxes[:, 0]).int())
            wmax.append(torch.ceil(new_boxxes[:, 2]).int())
            hmin.append(torch.floor(new_boxxes[:, 1]).int())
            hmax.append(torch.ceil(new_boxxes[:, 3]).int())

            area = 1.0 / (hmax[i].view(1, -1) + 1 - hmin[i].view(1, -1)) / (
                    wmax[i].view(1, -1) + 1 - wmin[i].view(1, -1))

            for j in range(len(gt_bboxes[i])):
                Mask_fg[i][hmin[i][j]:hmax[i][j] + 1, wmin[i][j]:wmax[i][j] + 1] = \
                    torch.maximum(Mask_fg[i][hmin[i][j]:hmax[i][j] + 1, wmin[i][j]:wmax[i][j] + 1], area[0][j])

            Mask_bg[i] = torch.where(Mask_fg[i] > 0, 0, 1)
            if torch.sum(Mask_bg[i]):
                Mask_bg[i] /= torch.sum(Mask_bg[i])


        fg_loss, bg_loss = self.get_fea_loss(preds_S, preds_T, Mask_fg, Mask_bg,
                                             C_attention_s, C_attention_t, S_attention_s, S_attention_t)
        # mask_loss = self.get_mask_loss(C_attention_s, C_attention_t, S_attention_s, S_attention_t)
        # rela_loss = self.get_rela_loss(preds_S, preds_T)

        # loss = self.alpha_fgd * fg_loss + self.beta_fgd * bg_loss
        loss  = fg_loss + bg_loss
        return loss *self.ratio

###################################################################################################


        # S_attention_t, C_attention_t = self.get_attention(preds_T, self.temp)
        # S_attention_s, C_attention_s = self.get_attention(preds_S, self.temp)
        #
        # Mask_fg = torch.zeros_like(S_attention_t)
        # Mask_bg = torch.ones_like(S_attention_t)
        # wmin, wmax, hmin, hmax = [], [], [], []
        # for i in range(N):
        #     new_boxxes = torch.ones_like(gt_bboxes[i])
        #     new_boxxes[:, 0] = gt_bboxes[i][:, 0] / img_metas[i]['img_shape'][1] * W
        #     new_boxxes[:, 2] = gt_bboxes[i][:, 2] / img_metas[i]['img_shape'][1] * W
        #     new_boxxes[:, 1] = gt_bboxes[i][:, 1] / img_metas[i]['img_shape'][0] * H
        #     new_boxxes[:, 3] = gt_bboxes[i][:, 3] / img_metas[i]['img_shape'][0] * H
        #
        #     wmin.append(torch.floor(new_boxxes[:, 0]).int())
        #     wmax.append(torch.ceil(new_boxxes[:, 2]).int())
        #     hmin.append(torch.floor(new_boxxes[:, 1]).int())
        #     hmax.append(torch.ceil(new_boxxes[:, 3]).int())
        #
        #     area = 1.0 / (hmax[i].view(1, -1) + 1 - hmin[i].view(1, -1)) / (
        #                 wmax[i].view(1, -1) + 1 - wmin[i].view(1, -1))
        #
        #     for j in range(len(gt_bboxes[i])):
        #         Mask_fg[i][hmin[i][j]:hmax[i][j] + 1, wmin[i][j]:wmax[i][j] + 1] = \
        #             torch.maximum(Mask_fg[i][hmin[i][j]:hmax[i][j] + 1, wmin[i][j]:wmax[i][j] + 1], area[0][j])
        #
        #     Mask_bg[i] = torch.where(Mask_fg[i] > 0, 0, 1)
        #     if torch.sum(Mask_bg[i]):
        #         Mask_bg[i] /= torch.sum(Mask_bg[i])
        #
        # fg_loss, bg_loss = self.get_fea_loss(preds_S, preds_T, Mask_fg, Mask_bg,
        #                                      C_attention_s, C_attention_t, S_attention_s, S_attention_t)
        # mask_loss = self.get_mask_loss(C_attention_s, C_attention_t, S_attention_s, S_attention_t)
        # rela_loss = self.get_rela_loss(preds_S, preds_T)
        #
        # loss = self.alpha_fgd * fg_loss + self.beta_fgd * bg_loss \
        #        + self.gamma_fgd * mask_loss + self.lambda_fgd * rela_loss
        #
        # return loss

    def get_queue_feature_loss(self,fg_mask,edge_circle_mask,bg_mask,queue_feature_S,queue_feature_T):

        fg_masked_feature_T = fg_mask * queue_feature_T
        fg_masked_feature_S = fg_mask * queue_feature_S
        fg_feature_loss = torch.cosine_similarity(fg_masked_feature_T,fg_masked_feature_S,dim=0).mean()

        edge_circle_masked_feature_T = edge_circle_mask * queue_feature_T
        edge_circle_masked_feature_S = edge_circle_mask * queue_feature_S
        edge_circle_loss = torch.cosine_similarity(edge_circle_masked_feature_T,edge_circle_masked_feature_S,dim=0).mean()

        bg_masked_feature_T = bg_mask * queue_feature_T
        bg_masked_feature_S = bg_mask * queue_feature_S
        bg_feature_loss = torch.cosine_similarity(bg_masked_feature_T,bg_masked_feature_S,dim=0).mean()

        loss = (1-fg_feature_loss) + (1-edge_circle_loss) + (1-bg_feature_loss)
        return loss
    def get_attention(self, preds, temp):
        """ preds: Bs*C*W*H """
        N, C, H, W = preds.shape

        value = torch.abs(preds)
        # Bs*W*H
        fea_map = value.mean(axis=1, keepdim=True)
        S_attention = (H * W * F.softmax((fea_map / temp).view(N, -1), dim=1)).view(N, H, W)

        # Bs*C
        channel_map = value.mean(axis=2, keepdim=False).mean(axis=2, keepdim=False)
        C_attention = C * F.softmax(channel_map / temp, dim=1)

        return S_attention, C_attention

    def get_fea_loss(self, preds_S, preds_T, Mask_fg, Mask_bg, C_s, C_t, S_s, S_t):
        loss_mse = nn.MSELoss(reduction='sum')
        L1_loss = nn.L1Loss(reduction='sum')

        Mask_fg = Mask_fg.unsqueeze(dim=1)
        Mask_bg = Mask_bg.unsqueeze(dim=1)

        C_t = C_t.unsqueeze(dim=-1)
        C_t = C_t.unsqueeze(dim=-1)

        S_t = S_t.unsqueeze(dim=1)


        #####
        HW,bs,c = preds_S.shape
        C_t = C_t.permute(2,3,0,1).view(-1,bs,c)
        S_t = S_t.permute(2,3,0,1).view(-1,bs,1)
        Mask_fg = Mask_fg.permute(2,3,0,1).view(-1,bs,1)
        Mask_bg = Mask_bg.permute(2, 3, 0, 1).view(-1,bs,1)
        #####
        # print('shape of channel attention',C_t.shape,'shape of spicial attention',S_t.shape,'shape of preds_S',preds_S.shape,'shape of mask_fg',Mask_fg.shape)

        fea_t = torch.mul(preds_T, torch.sqrt(S_t))
        fea_t = torch.mul(fea_t, torch.sqrt(C_t))
        fg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_fg))
        bg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_bg))

        fea_s = torch.mul(preds_S, torch.sqrt(S_t))
        fea_s = torch.mul(fea_s, torch.sqrt(C_t))
        fg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_fg))
        bg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_bg))

        fg_loss = L1_loss(fg_fea_s, fg_fea_t) / len(Mask_fg)
        bg_loss = L1_loss(bg_fea_s, bg_fea_t) / len(Mask_bg)

        return fg_loss, bg_loss

    def get_mask_loss(self, C_s, C_t, S_s, S_t):

        mask_loss = torch.sum(torch.abs((C_s - C_t))) / len(C_s) + torch.sum(torch.abs((S_s - S_t))) / len(S_s)

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

    def get_rela_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')

        context_s = self.spatial_pool(preds_S, 0)
        context_t = self.spatial_pool(preds_T, 1)

        out_s = preds_S
        out_t = preds_T

        channel_add_s = self.channel_add_conv_s(context_s)
        out_s = out_s + channel_add_s

        channel_add_t = self.channel_add_conv_t(context_t)
        out_t = out_t + channel_add_t

        rela_loss = loss_mse(out_s, out_t) / len(out_s)

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


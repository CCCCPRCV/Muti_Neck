import torch
import argparse
from collections import OrderedDict

def change_model(args):
    pre_model = torch.load(args.fgd_path)
    backbone_model = torch.load(args.backbone_model)
    all_name = []
    for name, v in pre_model["state_dict"].items():
        if 'backbone' in name:
            print(name)
            all_name.append((name, v))
        else:
            continue
    ########################################   exact  backbone

    for name, v in pre_model["state_dict"].items():
        if ('backbone' in name) or ('encoder.layers.3' in name) or ('encoder.layers.4' in name) or ('encoder.layers.5' in name) or ('decoder.layers.3' in name) or ('decoder.layers.4' in name) or ('decoder.layers.5' in name):
            continue
        else:
            print(name)
            all_name.append((name, v))

    state_dict = OrderedDict(all_name)
    pre_model['state_dict'] = state_dict
    torch.save(pre_model, args.output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer CKPT')
    parser.add_argument('--fgd_path', type=str, default='work_dirs/fgd_retina_rx101_64x4d_distill_retina_r50_fpn_2x_coco/epoch_24.pth',
                        metavar='N',help='fgd_model path')
    parser.add_argument('--backbone_model', type=str,
                        default='work_dirs/fgd_retina_rx101_64x4d_distill_retina_r50_fpn_2x_coco/epoch_24.pth',
                        metavar='N', help='fgd_model path')
    parser.add_argument('--output_path', type=str, default='retina_res50_new.pth',metavar='N',
                        help = 'pair path')
    args = parser.parse_args()
    change_model(args)
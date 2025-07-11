import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class residual_block(nn.Module):
    def __init__(self, n_chans=768):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(n_chans, n_chans, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_chans, n_chans, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        t = self.block(x)
        x = x + t
        return x

class DimReduction(nn.Module):
    def __init__(self, in_chans=768, out_chans=768, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(in_chans, out_chans, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for i in range(self.numRes):
            self.resBlocks.append(residual_block(out_chans))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x

# 生成注意力权重矩阵
class Attention_Gated(nn.Module):
    def __init__(self, L=768, D=128, K=1):
        super(Attention_Gated, self).__init__()
        self.L = L
        self.D = D
        self.K = K

        self.attention_v = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_u = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, is_norm=True):
        A_v = self.attention_v(x)
        A_u = self.attention_u(x)
        A = self.attention_weights(A_v * A_u)
        A = torch.transpose(A, 1, 0)
        if is_norm:
            A = F.softmax(A, dim=1)
        return A

class VisionMILEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.classifier = Classifier_1fc()
        self.attention1 = Attention_Gated()
        self.dimReduction = DimReduction()
        self.attention2 = Attention_Gated()
        # self.attCls = Attention_with_cls()

        self.ins_per_group = 128
        self.distill = 'AFS' # choices: 'MaxMinS', 'MaxS', 'AFS'
        # self.distill = args.distill

    def forward(self, patches):
        patches = patches.squeeze()
        # print(f'patches.shape: {patches.shape}')
        index_chunk_list = self.split_chunk_list(patches)
        # print(f'len(index_chunk_list): {len(index_chunk_list)}, index_chunk_list[0]: {len(index_chunk_list[0])}')
        slide_sub_feats = []
        slide_pseudo_feat = []
        for t_idx in index_chunk_list:
            sub_wsi = torch.index_select(patches, dim=0, index=torch.LongTensor(t_idx).to(patches.device)).to(patches.device)
            # print(f'sub_wsi.shape: {sub_wsi.shape}')
            sub_feat = self.dimReduction(sub_wsi)
            # print(f'sub_feat.shape: {sub_feat.shape}')
            sub_AA = self.attention1(sub_feat).squeeze(0) # attention weight: [N]
            # print(f'sub_AA.shape: {sub_AA.shape}')
            sub_att_feats = torch.einsum('nd,n->nd', sub_feat, sub_AA)
            # print('sub_att_feats.shape:', sub_att_feats.shape)
            sub_att_feat = torch.sum(sub_att_feats, dim=0).unsqueeze(0) # [1, D]
            # print(f'sub_att_feat.shape: {sub_att_feat.shape}')
            slide_sub_feats.append(sub_att_feat)

            # feature distillation
            if self.distill == 'AFS':
                slide_pseudo_feat.append(sub_att_feat)
            else:
                print(f'not implemented yet: {self.distill}')

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0) # [num_group, D]
        # print(f'slide_pseudo_feat.shape: {slide_pseudo_feat.shape}')
        slide_AA = self.attention2(slide_pseudo_feat) # [1, num_group]
        slide_att_feat = torch.mm(slide_AA, slide_pseudo_feat) # [1, D]
        # print(f'slide_att_feat.shape: {slide_att_feat.shape}')

        slide_feats = torch.cat([slide_att_feat] + [slide_pseudo_feat], dim=0)
        slide_feats = slide_feats.unsqueeze(0)
        # print(f'slide_feats.shape = {slide_feats.shape}')
        return slide_feats

    def split_chunk_list(self, patches):
        # 把patches分成num_group组，每组self.ins_per_group个
        num_group = max(patches.shape[0] // self.ins_per_group + 1, 3)
        feat_index = list(range(patches.shape[0]))
        random.shuffle(feat_index)
        index_chunk_list = np.array_split(np.array(feat_index), num_group)
        index_chunk_list = [chunk.tolist() for chunk in index_chunk_list]
        return index_chunk_list

if __name__ == '__main__':
    args = type('', (), {})()  # 创建动态类
    args.distill = 'AFS'
    model = VisionMILEncoder(args)
    patches = torch.randn(260, 768)
    output = model(patches)
    print(output.shape)
    print('Done')
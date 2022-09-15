from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import torch as t
import math
from tran import TF



class TFGenerater(nn.Module):

    def __init__(self,cfg):
        super(TFGenerater, self).__init__()
        channel=32

        # self.proj = nn.Conv2d(32, channel, kernel_size=3, stride=1, padding=1)
        self.query_generation = nn.Sequential(
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.ReLU(),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.ReLU(),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                )

        Groupchannel=channel//2 if cfg.TRAIN.groupchannel is  None else cfg.TRAIN.groupchannel

        self.conv = nn.Sequential(
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(Groupchannel,channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(Groupchannel,channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(Groupchannel,channel),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel, channel,  kernel_size=3, stride=4,padding=0),
                nn.ConvTranspose2d(channel, 3, kernel_size=3, stride=2, padding=0)
                )


        self.row_embed = nn.Embedding(128, channel//2)
        self.col_embed = nn.Embedding(128, channel//2)
        self.reset_parameters()

        self.transformer = TF(channel, 8, 2, 2)




    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)


    def forward(self,x):

        b, c, w, h = x.size()

        i = t.arange(w).cuda()
        j = t.arange(h).cuda()

        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = t.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)


        attn_mask=None
        x_q = self.query_generation(x)
        x2=self.transformer(x.view(b,c,-1).permute(2, 0, 1),\
                             x.view(b,c,-1).permute(2, 0, 1),\
                             x_q.view(b, c, -1).permute(2, 0, 1), \
                             pos.view(b, c, -1).permute(2, 0, 1), \
                             w,h,\
                             src_mask=attn_mask)
#        ress3=self.transformer((pos+res1).view(b,c,-1).permute(2, 0, 1),\
 #                            (pos+res2).view(b,c,-1).permute(2, 0, 1),\
  #                           (res3).view(b, c, -1).permute(2, 0, 1))

        x=x2.permute(1,2,0).view(b,c,w,h)

        m=self.conv(x)

        return m






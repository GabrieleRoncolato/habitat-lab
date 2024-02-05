import torch
import torch.nn as nn

from timm.models.vision_transformer import VisionTransformer, Mlp, PatchEmbed, \
    _cfg
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from functools import partial


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim,
                                   self.num_classes) if self.num_classes > 0 \
            else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)

        # print(x)

        x_dist = self.head_dist(x_dist)

        #print(x.shape)
        #print(x_dist.shape)

        # print(x_dist)
        if self.training:
            return x  # , x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2


def deit_tiny_distilled_patch16_224(img_size=(224, 224), patch_size=16,
                                    embed_dim=192, **kwargs):
    model = DistilledVisionTransformer(img_size=img_size,
                                       patch_size=patch_size,
                                       embed_dim=embed_dim,
                                       num_classes=0,
                                       depth=12,
                                       num_heads=3, mlp_ratio=4,
                                       qkv_bias=True,
                                       norm_layer=partial(nn.LayerNorm,
                                                          eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

import torch
from vit_pytorch import ViT
from einops import rearrange, repeat


class NoHeadViT(ViT):
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        # return self.mlp_head(x)

        return x


def simple_vit(img_size=(224, 224), patch_size=16, embed_dim=192, **kwargs):
    model = NoHeadViT(image_size=img_size,
                      patch_size=patch_size,
                      dim=embed_dim,
                      num_classes=0,
                      depth=12,
                      heads=3,
                      mlp_dim=embed_dim,
                      dropout=0.1,
                      emb_dropout=0.1,
                      **kwargs)
    return model

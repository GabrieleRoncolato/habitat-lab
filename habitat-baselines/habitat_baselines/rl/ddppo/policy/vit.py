import torch
from vit_pytorch import SimpleViT


class NoHeadSimpleViT(SimpleViT):
    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        # return self.linear_head(x)
        return x


def simple_vit(img_size=(224, 224), patch_size=16, embed_dim=192, **kwargs):
    model = NoHeadSimpleViT(image_size=img_size,
                            patch_size=patch_size,
                            dim=embed_dim,
                            num_classes=0,
                            depth=6,
                            heads=3,
                            mlp_dim=embed_dim,
                            **kwargs)
    return model

# local_sam/model.py
import torch
import torch.nn as nn
from .image_encoder import ImageEncoderViT
from .prompt_encoder import PromptEncoder
from .mask_decoder import MaskDecoder
from .two_way_transformer import TwoWayTransformer


class Sam(nn.Module):
    def __init__(self, image_encoder, prompt_encoder, mask_decoder, pixel_mean, pixel_std):
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    def forward(self, image, points=None, boxes=None, masks=None):
        # normalize
        x = (image - self.pixel_mean) / self.pixel_std
        # encode image
        image_embeddings = self.image_encoder(x)
        # encode prompts
        sparse_embeddings, dense_embeddings = self.prompt_encoder(points, boxes, masks)
        # decode into masks
        masks, iou_pred = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
        )
        return masks, iou_pred


# Builders for official configs
def build_sam_vit_h():
    return Sam(
        image_encoder=ImageEncoderViT(
            depth=32, embed_dim=1280, num_heads=16, global_attn_indexes=[7, 15, 23, 31]
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=256, image_embedding_size=(64, 64),
            input_image_size=(1024, 1024), mask_in_chans=16
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(depth=2, embed_dim=256, num_heads=8, mlp_dim=2048),
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )


def build_sam_vit_l():
    return Sam(
        image_encoder=ImageEncoderViT(
            depth=24, embed_dim=1024, num_heads=16, global_attn_indexes=[5, 11, 17, 23]
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=256, image_embedding_size=(64, 64),
            input_image_size=(1024, 1024), mask_in_chans=16
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(depth=2, embed_dim=256, num_heads=8, mlp_dim=2048),
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )


def build_sam_vit_b():
    return Sam(
        image_encoder=ImageEncoderViT(
            depth=12, embed_dim=768, num_heads=12, global_attn_indexes=[2, 5, 8, 11]
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=256, image_embedding_size=(64, 64),
            input_image_size=(1024, 1024), mask_in_chans=16
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(depth=2, embed_dim=256, num_heads=8, mlp_dim=2048),
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

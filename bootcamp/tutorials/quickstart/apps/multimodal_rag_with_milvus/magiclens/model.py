# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# pylint: disable=redefined-outer-name,missing-module-docstring,g-importing-member,missing-function-docstring,g-bare-generic
from typing import Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from .layers import AttenTokenPoolingLayer
from .layers import StackedTransformer
from scenic.projects.baselines.clip import layers as clip_layers
from scenic.projects.baselines.clip import model as clip_model

MagicLensConfig = {
    "base": dict(
        embed_dim=512,
        ff_hidden_size=512 * 4,
        num_layers=4,
        num_heads=8,
        num_query_token=1,
        clip_model_name="vit_b16",
    ),
    "large": dict(
        embed_dim=768,
        ff_hidden_size=768 * 4,
        num_layers=4,
        num_heads=16,
        num_query_token=1,
        clip_model_name="vit_l14",
    ),
}


def largest_square_crop(images: jnp.ndarray) -> jnp.ndarray:
    assert images.ndim >= 4
    h, w, _ = images.shape[-3:]
    size = w if h > w else h

    pos_h = (h - w) // 2 if h > w else 0
    pos_w = (w - h) // 2 if w > h else 0

    return images[..., pos_h : pos_h + size, pos_w : pos_w + size, :]


class MagicLens(nn.Module):
    """MagicLens model built upon CLIP."""

    model_size: str = "base"

    def setup(self):
        self.clip_model_name = MagicLensConfig[self.model_size]["clip_model_name"]
        self.size: int = clip_model.IMAGE_RESOLUTION[self.clip_model_name]
        self.config: dict = clip_model.CONFIGS[self.clip_model_name]

        self.clip = clip_layers.CLIP(
            vocab_size=self.config["vocab_size"],
            embed_dim=self.config["embed_dim"],
            text_features=self.config["text_features"],
            text_num_layers=self.config["text_num_layers"],
            text_num_heads=self.config["text_num_heads"],
            vision_features=self.config["vision_features"],
            vision_num_layers=self.config["vision_num_layers"],
            vision_patch_size=self.config.get("vision_patch_size", None),
            vision_return_map=False,
            use_underscore_module_name=True,
            name="clip",
        )

        self.multimodal_encoder = StackedTransformer(
            num_layers=MagicLensConfig[self.model_size]["num_layers"],
            num_heads=MagicLensConfig[self.model_size]["num_heads"],
            input_dim=MagicLensConfig[self.model_size]["embed_dim"],
            hidden_dim=MagicLensConfig[self.model_size]["ff_hidden_size"],
            use_bias=True,
            add_skip_connection=True,
            use_per_dim_scale=False,
            name="multimodal_encoder",
        )

        self.contrastive_multimodal_pooler = AttenTokenPoolingLayer(
            input_dim=MagicLensConfig[self.model_size]["embed_dim"],
            query_dim=MagicLensConfig[self.model_size]["embed_dim"],
            num_heads=MagicLensConfig[self.model_size]["num_heads"],
            num_query_tokens=MagicLensConfig[self.model_size]["num_query_token"],
            use_bias=True,
            use_per_dim_scale=True,
            name="contrastive_multimodal_pooler",
        )

    def _preprocess_images(self, images: jax.Array) -> jax.Array:
        """Center crop & resize image to be compatible with the underlied vision model."""
        assert images.ndim >= 4
        target_shape = images.shape[:-3] + (self.size, self.size, images.shape[-1])
        images = largest_square_crop(images)
        images = jax.image.resize(
            images, shape=target_shape, method="bilinear", antialias=True
        )
        # Apply CLIP-specific shifting/scaling.
        # The input to `normalize_image` is expected to be in [0, 1].
        images = clip_model.normalize_image(images)
        return images

    def clip_encode(self, input_batch: Dict) -> Tuple[jax.Array, jax.Array]:
        """Computes CLIP embeds for the given batch of images and texts.

        Args:
          input_batch: A Dict of the following fields:
            * ids: [B, T] or [B, 1, T]. Text token ids
            * paddings: [B, T] or [B, 1, T]. Text token paddings.
            * image: [B, H, W, 3]. Input image.

        Returns:
          image_embs: [B, D]
          text_embs: [B, D]
          patch_embeds: [B, N, D]
          token_embds: [B, T, D]
        """
        assert input_batch["ids"].ndim <= 3
        if input_batch["ids"].ndim == 3:
            # Only takes the first caption.
            input_batch["ids"] = input_batch["ids"][:, 0, :]
        images = self._preprocess_images(input_batch["image"])
        image_embs, text_embs = self.clip(images, input_batch["ids"], normalize=False)
        return image_embs, text_embs

    def _normalize_embed(self, embed: jax.Array) -> jax.Array:
        """Applies normalization on the input embedding.

        Args:
          embed: [B, D]. The input embedding to normalize.

        Returns:
          The normalized embedding.
        """
        # Always converts embed to float32 for all precisions.
        embed = jnp.asarray(embed, dtype=jnp.float32)
        # return py_utils.l2_normalize(embed, axis=-1)
        norm = jnp.sqrt(jnp.sum(embed * embed, axis=-1, keepdims=True) + 1e-12)
        return embed / norm

    def __call__(self, input_batch: Dict) -> Dict:
        """Computes the multimodal embeddings.

        It computes the multimodal embeddings pooling from both
        text embeddings and image *generative* embeddings.
        If text is empty, use image pooling only.

        Args:
          input_batch: A Dict of the following fields:
            * ids: [B, T] or [B, 1, T]. Text token ids
            * paddings: [B, T] or [B, 1, T]. Text token paddings.
            * image: [B, H, W, 3]. Input image.
        Returns:
          A Dict contains the following fields:
            * multimodal_embed: [B, D], multimodal embedding
            * multimodal_embed_norm: [B, D], normalized multimodal embedding
        """
        img_embed, txt_embed = self.clip_encode(input_batch)  # [B, D], [B, D]
        img_embed = img_embed.reshape([-1, 1, img_embed.shape[-1]])  # [B, 1, D]
        txt_embed = txt_embed.reshape([-1, 1, txt_embed.shape[-1]])  # [B, 1, D]

        concate_mm_embed = jnp.concatenate([img_embed, txt_embed], axis=1)

        multimodal_embed = self.multimodal_encoder(concate_mm_embed)  # [B, 2, D]

        multimodal_embed = self.contrastive_multimodal_pooler(multimodal_embed)
        multimodal_embed = multimodal_embed[:, 0]

        multimodal_embed_norm = self._normalize_embed(multimodal_embed)

        # placeholder for model matching
        # contrastive_loss = 0.0
        return {
            "multimodal_embed": multimodal_embed,
            "multimodal_embed_norm": multimodal_embed_norm,
        }

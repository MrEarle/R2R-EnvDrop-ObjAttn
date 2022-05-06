from abc import abstractmethod
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from param import args


class ObjectFeatureReducer(nn.Module):
    def __init__(
        self,
        feature_dim: int = 256,
        object_dim: int = 1024,
        output_dim: int = 256,
    ) -> None:
        super().__init__()

        self.obj_feat_reducer = nn.LazyConv2d(
            out_channels=feature_dim,
            kernel_size=1,
        )

        self.obj_encoder = nn.LazyLinear(object_dim)
        if args.include_objs_lstm:
            self.obj_summarizer = nn.LazyLinear(output_dim)

    def forward(self, obj_features: Tensor, obj_orients: Tensor) -> Tensor:
        """
        Args:
            obj_features: (batch, num_objs, 2048, 2, 2)
            obj_orients: (batch, num_objs, orient_size)
        Returns:
            reduced_feats: (batch, num_objs, object_dim)
            object_summary: (batch, output_dim)
        """
        # Flatten to use num_objs as batch. [batch * num_objs, 2048, 2, 2]
        o_shape = obj_features.shape
        obj_features = obj_features.view((-1, *o_shape[2:]))

        # Reduce object features to [batch * num_objs, obj_attn_size, 2, 2]
        reduced_feats = self.obj_feat_reducer(obj_features)

        # Restore batch and num_objs, and flatten. [batch, num_objs, obj_attn_size * 4 (1024)]
        reduced_feats = reduced_feats.view((o_shape[0], o_shape[1], -1))

        # Add orientation information to objects
        reduced_feats = torch.cat([reduced_feats, obj_orients], dim=-1)

        # Reduce dimentionality to [batch, num_objs, object_dim]
        reduced_feats = self.obj_encoder(reduced_feats)

        # Compute a representation of all objects in the scene
        # [batch, num_objs * object_dim]
        object_summary = None
        if args.include_objs_lstm:
            object_summary = reduced_feats.view((o_shape[0], -1))
            object_summary = self.obj_summarizer(object_summary)

        return reduced_feats, object_summary


class ObjectHeadingViewpointSimilarity(nn.Module):
    def __init__(self, heading_feat_size: int = 64) -> None:
        super().__init__()

        self._heading_feat_size = heading_feat_size

        self._obj_head_proj = nn.LazyLinear(self._heading_feat_size)

        self._view_head_proj = nn.LazyLinear(self._heading_feat_size)

        self.softmax = nn.Softmax(dim=2)

    def forward(
        self,
        obj_feats: Tensor,
        object_headings: Tensor,
        viewpoint_headings: Tensor,
    ) -> Tensor:
        """
        params:
            obj_feats:
                shape: [batch, num_objs, obj_feat_size]
            object_headings:
                shape:  [batch, num_objs, orient_feat_size]
            viewpoint_headings:
                shape: [batch, num_viewpoints, view_heading_size]


        return:
            attended_objs:  Objects weighted by their heading similarity with the viewpoints
                shape: [batch, num_viewpoints, num_objs, obj_feat_size]
        """
        # [batch, num_objs, self._heading_feat_size]
        projected_obj_heading = self._obj_head_proj(object_headings)

        # [batch, num_viewpoints, self._heading_feat_size]
        projected_viewpoint_heading = self._view_head_proj(viewpoint_headings)

        # [batch, num_viewpoints, num_objs]
        heading_similarity = torch.einsum(
            "bos,bvs->bvo",
            projected_obj_heading,
            projected_viewpoint_heading,
        )

        # Normalizar heading similarity para que sea un score
        heading_similarity = self.softmax(heading_similarity)

        # [batch, num_viewpoints, num_objs, obj_feat_size]
        weighted_objects = torch.einsum("bvo,bof->bvof", heading_similarity, obj_feats)

        return weighted_objects


class ConnectionwiseObjectAttention(nn.Module):
    def __init__(self, obj_attn_size=512) -> None:
        super().__init__()
        self._obj_projection = nn.LazyLinear(obj_attn_size)
        self._txt_projection = nn.LazyLinear(obj_attn_size)
        self._obj_attention = nn.MultiheadAttention(
            embed_dim=obj_attn_size,
            num_heads=1,
            batch_first=True,
        )
        self._obj_viewpoint_similarity = ObjectHeadingViewpointSimilarity()

    def forward(
        self,
        object_headings: Tensor,
        object_feats: Tensor,
        viewpoint_heading: Tensor,
        text_context: Tensor,
        obj_mask: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters:
            object_headings:
                shape: [batch, num_objs, orient_feat_size]
            object_feats:
                shape:  [batch, num_objs, object_feat_Size]
            viewpoint_heading:
                shape: [batch, num_viewpoints, orient_feat_size]
            text_context:
                shape: [batch, txt_context_shape]
            obj_mask:
                shape: [batch, num_objs]
        """

        # Get object weighed by heading similarity with viewpoints
        # Shape: [batch, num_viewpoints, num_objs, obj_feat_size]
        viewpoint_objs = self._obj_viewpoint_similarity(
            obj_feats=object_feats,
            object_headings=object_headings,
            viewpoint_headings=viewpoint_heading,
        )

        # Concatenate object encoding with heading encoding
        # and perform attention over objects, based on instruction context
        batch_size = viewpoint_objs.shape[0]
        num_viewpoints = viewpoint_objs.shape[1]

        # [batch, num_objs, direction_feats] -> [batch, num_viewpoints, num_objs, direction_feats]
        repeated_headings = object_headings.unsqueeze(1).repeat_interleave(num_viewpoints, dim=1)

        # [batch, num_objs] -> [batch, num_viewpoints, num_objs]
        repeated_obj_mask = obj_mask.unsqueeze(1).repeat_interleave(num_viewpoints, dim=1)

        # [batch, ctx_size] -> [batch, num_viewpoints, obj_attn_size]
        text_context = self._txt_projection(text_context)
        repeated_text_ctx = text_context.unsqueeze(1).repeat_interleave(num_viewpoints, dim=1)

        # [batch, num_viewpoints, num_objs, obj_attn_size + direction_feat]
        objs = torch.cat((viewpoint_objs, repeated_headings), dim=-1)

        # Join batch and num_viewpoints
        # [batch * num_viewpoints, num_objs, obj_attn_size + direction_feat]
        objs = objs.view((-1, *objs.shape[-2:]))
        # [batch * num_viewpoints, num_objs]
        repeated_obj_mask = repeated_obj_mask.view((-1, repeated_obj_mask.shape[-1]))
        # [batch * num_viewpoints, 1, ctx_size]
        repeated_text_ctx = repeated_text_ctx.view((-1, 1, repeated_text_ctx.shape[-1]))

        # Project object encoding to attention size
        # [batch * num_viewpoints, num_objs, obj_attn_size]
        objs = self._obj_projection(objs)

        # Perform attn
        # Query -> repeated_text_ctx [batch * num_viewpoints, 1, ctx_size]
        # Key & Value -> objs [batch * num_viewpoints, num_objs, obj_attn_size]
        # Mask -> repeated_obj_mask [batch * num_viewpoints, num_objs]

        # attended_objects = [batch * num_viewpoints, 1, obj_attn_size]
        # attn_weights = [batch * num_viewpoints, 1, num_objs]
        attended_objects, attn_weights = self._obj_attention(
            query=repeated_text_ctx,
            key=objs,
            value=objs,
            key_padding_mask=repeated_obj_mask,
            need_weights=True,
        )

        # Split batch and num_viewpoints
        # [batch, num_viewpoints, obj_attn_size]
        attended_objects = attended_objects.view((batch_size, num_viewpoints, -1))
        # [batch, num_viewpoints, num_objs]
        attn_weights = attn_weights.view((batch_size, num_viewpoints, -1))

        return attended_objects, attn_weights

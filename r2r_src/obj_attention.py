from typing import Tuple
import torch
from torch import Tensor, nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, attn_mask=None, reverse_attn=False):
        _, len_q, _ = q.size()
        _, len_k, _ = k.size()

        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        if reverse_attn:
            attn = -attn

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).expand_as(attn)
            attn.data.masked_fill_((attn_mask == 0).data, -float("inf"))
            # attn = attn.masked_fill((attn_mask == 0).data, -np.inf)

        attn_weight = self.softmax(attn.view(-1, len_k)).view(-1, len_q, len_k)

        attn_weight = self.dropout(attn_weight)
        output = torch.bmm(attn_weight, v)
        return output, attn_weight


# Notes:
# top_x = Top x object, according to some metric
# 36 -> Images in panorama
# obj_feat_size: Size of object features
# orient_feat_size: Size of feature describing an object's relative orientation


"""
Idea:
Quizas debiera tomar los objetos, y atenderlos usando como query la orientación
del viewpoint navegable, como value la orientación del objeto, y como value
el embedding del objeto.

Tambien se podrían combinar ambas modalidades en paralelo.

Esto creo que ocurriria en el head del agent?
"""


class ObjectAttention(nn.Module):
    def __init__(
        self,
        object_attention_size: int = 256,
        num_objects=20,  # TODO: Hyperparam
    ) -> None:
        super().__init__()

        self._obj_attn_size = object_attention_size
        self._project_obj = nn.LazyLinear(self._obj_attn_size)
        self._project_text = nn.LazyLinear(self._obj_attn_size)
        self._obj_attn = ScaledDotProductAttention(d_model=num_objects)

    def forward(self, obj_feats: Tensor, text_ctx: Tensor) -> Tensor:
        """
        params:
            obj_feats:  Collection of num_objs objects detected for every image composing
                        the panorama.
                shape: [batch, num_objs, obj_feat_size + orient_feat_size]
            text_ctx:   Language features attended by visual input (RCM)
                shape: [batch, lang_feat_size]


        return:
            attended_objs:  Attended objects for each of the 36 separate orientations
                            in panorama.
                shape: [batch, 1, self._obj_attn_size]
            attn_weights:  Respective attention weights
                shape: [batch, 1, num_objs]
        """

        # [batch, num_objs, self._obj_attn_size]
        projected_obj_feats: Tensor = self._project_obj(obj_feats)

        # [batch, 1, self._obj_attn_size]
        projected_text_ctx: Tensor = self._project_text(text_ctx)
        projected_text_ctx = projected_text_ctx.unsqueeze(dim=1)

        # attended_objs: [batch, 1, self._obj_attn_size]
        # attn_weights: [batch, 1, num_objs]
        attended_objs, attn_weights = self._obj_attn(
            q=projected_text_ctx, k=projected_obj_feats, v=projected_obj_feats
        )

        return attended_objs, attn_weights


class ObjectHeadingViewpointSimilarity(nn.Module):
    def __init__(self, heading_feat_size: int = 64) -> None:
        super().__init__()

        self._heading_feat_size = heading_feat_size

        self._obj_head_proj = nn.LazyLinear(self._heading_feat_size)

        self._view_head_proj = nn.LazyLinear(self._heading_feat_size)

        self.softmax = nn.Softmax(dim=2)

    def call(
        self,
        obj_feats: Tensor,
        object_headings: Tensor,
        viewpoint_headings: Tensor,
    ) -> Tensor:
        """
        params:
            obj_feats:
                shape: [batch, num_objs, obj_feat_size
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

        # TODO: Normalizar heading similarity para que sea un score. Quizas softmax?
        heading_similarity = self.softmax(heading_similarity)

        # [batch, num_viewpoints, num_objs, obj_feat_size]
        weighted_objects = torch.einsum("bvo,bof->bvof", heading_similarity, obj_feats)

        return weighted_objects


class ConnectionwiseObjectAttention(nn.Module):
    def __init__(self, obj_attn_size=256):
        super().__init__()

        self._obj_viewpoint_similarity = ObjectHeadingViewpointSimilarity()
        self._obj_attention = ObjectAttention(object_attention_size=obj_attn_size)

    def forward(
        self,
        object_headings: Tensor,
        encoded_obj_idxs: Tensor,
        viewpoint_heading: Tensor,
        text_context: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters:
            object_headings:
                shape: [batch, num_objs, direction_feats]
            encoded_obj_idxs:
                shape:  [batch, num_objs, max_obj_len * txt_feat_size]
            viewpoint_heading:
                shape: [batch, num_viewpoints, viewpoint_heading_feat]
            text_context:
                shape: [batch, txt_context_shape]
        """

        # Get object weighed by heading similarity with viewpoints
        # Shape: [batch, num_viewpoints, num_objs, obj_feat_size]
        viewpoint_objs = self._obj_viewpoint_similarity(
            obj_feats=encoded_obj_idxs,
            object_headings=object_headings,
            viewpoint_headings=viewpoint_heading,
        )

        # Concatenate object encoding with heading encoding
        # and perform attention over objects, based on instruction context
        attended_objects = []  # [[batch, 1, obj_attn_size] * num_viewpoints]
        attn_weights = []  # [[batch, 1, num_objs] * num_viewpoints]
        for view_id in range(viewpoint_objs.shape[1]):
            objs = torch.cat(
                [
                    viewpoint_objs[:, view_id, :, :],
                    object_headings,
                ],
                dim=-1,
            )

            # [batch, 1, self._obj_attn_size], [batch, 1, num_objs]
            attn_objs, attn_weight = self._obj_attention(objs, text_context)

            attended_objects.append(attn_objs)
            attn_weights.append(attn_weight)

        # shape: [batch, num_viewpoints, obj_attn_size]
        attended_objects = torch.stack(attended_objects, dim=1)

        # shape: [batch, num_viewpoints, num_objs]
        attn_weights = torch.stack(attn_weights, dim=1)

        return attended_objects, attn_weights

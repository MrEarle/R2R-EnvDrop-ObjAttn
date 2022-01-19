from abc import abstractmethod
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from param import args


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)

        self.neginf = torch.tensor([float("-inf")]).cuda()

    def forward(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor = None, reverse_attn=False):
        _, len_q, _ = q.size()
        _, len_k, _ = k.size()

        attn: Tensor = torch.bmm(q, k.transpose(1, 2)) / self.temper

        if reverse_attn:
            attn = -attn

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
            attn = torch.where(attn_mask.bool(), attn, self.neginf)

        attn_weight = self.softmax(attn.view(-1, len_k)).view(-1, len_q, len_k)
        attn_weight = torch.nan_to_num(attn_weight)

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
        num_objects=args.max_obj_number,
    ) -> None:
        super().__init__()

        self._obj_attn_size = object_attention_size
        self._project_obj = nn.LazyLinear(self._obj_attn_size)
        self._project_text = nn.LazyLinear(self._obj_attn_size)
        self._obj_attn = ScaledDotProductAttention(d_model=num_objects)

    def forward(self, obj_feats: Tensor, text_ctx: Tensor, obj_mask: Tensor = None) -> Tensor:
        """
        params:
            obj_feats:  Collection of num_objs objects detected for every image composing
                        the panorama. (values)
                shape: [batch, num_objs, obj_feat_size + orient_feat_size]
            text_ctx:   Language features attended by visual input (RCM) (query)
                shape: [batch, lang_feat_size]
            obj_mask:   Object mask
                shape: [batch, num_objs]


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
            q=projected_text_ctx, k=projected_obj_feats, v=projected_obj_feats, attn_mask=obj_mask
        )

        return attended_objs, attn_weights


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

        # ? Normalizar heading similarity para que sea un score. Quizas softmax?
        heading_similarity = self.softmax(heading_similarity)

        # [batch, num_viewpoints, num_objs, obj_feat_size]
        weighted_objects = torch.einsum("bvo,bof->bvof", heading_similarity, obj_feats)

        return weighted_objects


class BaseObjAttn(nn.Module):
    def __init__(self, obj_attn_size=256):
        super().__init__()

        self.obj_feat_reducer = nn.LazyConv2d(
            out_channels=obj_attn_size,
            kernel_size=1,
        )

        self._obj_viewpoint_similarity = ObjectHeadingViewpointSimilarity()
        self._obj_attention = ObjectAttention(object_attention_size=obj_attn_size)
        self.traj_info = None

        if args.obj_aux_task:
            self.obj_aux_linear = nn.Sequential(
                nn.LazyLinear(obj_attn_size),
                nn.LazyLinear(args.num_obj_classes),
            )

    @abstractmethod
    def forward(
        self,
        object_headings: Tensor,
        object_feats: Tensor,
        viewpoint_heading: Tensor,
        text_context: Tensor,
        obj_mask: Tensor = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Parameters:
            object_headings:
                shape: [batch, num_objs, direction_feats]
            object_feats:
                shape:  [batch, num_objs, 2048, 2, 2]
            viewpoint_heading:
                shape: [batch, num_viewpoints, viewpoint_heading_feat]
            text_context:
                shape: [batch, txt_context_shape]
            obj_mask:
                shape: [batch, num_objs]
        """

        # Flatten to use num_objs as batch. [batch * num_objs, 2048, 2, 2]
        o_shape = object_feats.shape
        object_feats = object_feats.view((-1, *o_shape[2:]))

        # Reduce object features to [batch * num_objs, obj_attn_size, 2, 2]
        reduced_feats = self.obj_feat_reducer(object_feats)
        # TODO: Quizas agregar una capa de embedding a la posicion del objeto? Tipo LXMERT

        # Restore batch and num_objs, and flatten. [batch, num_objs, obj_attn_size * 4 (1024)]
        reduced_feats = reduced_feats.view((o_shape[0], o_shape[1], -1))

        attended_objects, attn_weights = self.forward_objects(
            object_headings, reduced_feats, viewpoint_heading, text_context, obj_mask
        )

        obj_aux_scores = None
        if args.obj_aux_task:
            # [batch, num_objs, num_obj_classes]
            obj_aux_scores = self.obj_aux_linear(reduced_feats)

        return attended_objects, attn_weights, obj_aux_scores

    @abstractmethod
    def forward_objects(
        self,
        object_headings: Tensor,
        object_feats: Tensor,
        viewpoint_heading: Tensor,
        text_context: Tensor,
        obj_mask: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        pass


class ConnectionwiseObjectAttention(BaseObjAttn):
    def __init__(self, obj_attn_size=256):
        super().__init__(obj_attn_size)
        self._obj_viewpoint_similarity = ObjectHeadingViewpointSimilarity()

    def forward_objects(
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
                shape: [batch, num_objs, direction_feats]
            object_feats:
                shape:  [batch, num_objs, 2048, 2, 2]
            viewpoint_heading:
                shape: [batch, num_viewpoints, viewpoint_heading_feat]
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
        attended_objects = []  # [[batch, 1, obj_attn_size] * num_viewpoints]
        attn_weights = []  # [[batch, 1, num_objs] * num_viewpoints]
        for view_id in range(viewpoint_objs.shape[1]):
            objs = torch.cat(
                [viewpoint_objs[:, view_id, :, :], object_headings],
                dim=-1,
            )

            # [batch, 1, self._obj_attn_size], [batch, 1, num_objs]
            attn_objs, attn_weight = self._obj_attention(objs, text_context, obj_mask=obj_mask)

            attended_objects.append(attn_objs)
            attn_weights.append(attn_weight)

        # shape: [batch, num_viewpoints, obj_attn_size]
        attended_objects = torch.stack(attended_objects, dim=1).squeeze()

        # shape: [batch, num_viewpoints, num_objs]
        attn_weights = torch.stack(attn_weights, dim=1)

        return attended_objects, attn_weights


class NoConnectionObjectAttention(BaseObjAttn):
    def forward_objects(
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
                shape: [batch, num_objs, direction_feats]
            object_feats:
                shape:  [batch, num_objs, 2048, 2, 2]
            viewpoint_heading:
                shape: [batch, num_viewpoints, viewpoint_heading_feat]
            text_context:
                shape: [batch, txt_context_shape]
            obj_mask:
                shape: [batch, num_objs]
        """

        # [batch, num_objs, obj_feat_size + angle_feat_size]
        objs = torch.cat((object_feats, object_headings), dim=-1)

        # [batch, 1, self.obj_attn_size], [batch, 1, num_objs]
        attended_objects, attn_weights = self._obj_attention(objs, text_context, obj_mask=obj_mask)

        # [batch, num_viewpoints, self.obj_attn_size]
        attended_objects = torch.repeat_interleave(attended_objects, viewpoint_heading.shape[1], 1)
        # [batch, num_viewpoints, num_objs]
        attn_weights = torch.repeat_interleave(attn_weights, viewpoint_heading.shape[1], 1)

        return attended_objects, attn_weights


def get_object_attention_class() -> BaseObjAttn:
    if args.obj_attn_type == "no_connection":
        return NoConnectionObjectAttention
    return ConnectionwiseObjectAttention

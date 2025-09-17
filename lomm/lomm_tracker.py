import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from mask2former_video.modeling.transformer_decoder.video_mask2former_transformer_decoder import SelfAttentionLayer,\
    CrossAttentionLayer, FFNLayer, MLP, _get_activation_fn
from scipy.optimize import linear_sum_assignment
import fvcore.nn.weight_init as weight_init


class ReferringCrossAttentionLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        indentify,
        tgt,
        key,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None
    ):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(key, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = indentify + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        indentify,
        tgt,
        key,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None
    ):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(key, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = indentify + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        indentify,
        tgt,
        key,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None
    ):
        # when set "indentify = tgt", ReferringCrossAttentionLayer is same as CrossAttentionLayer
        if self.normalize_before:
            return self.forward_pre(indentify, tgt, key, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(indentify, tgt, key, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


# Only T_E, corresponding to Tab. 3-(c)
class LOMM_Tracker_E(torch.nn.Module):
    def __init__(
        self,
        hidden_channel=256,
        feedforward_channel=2048,
        num_head=8,
        decoder_layer_num=6,
        mask_dim=256,
        class_num=25,
        num_queries=100,
    ):
        super(LOMM_Tracker_E, self).__init__()

        self.hidden_channel = hidden_channel
        self.num_classes = class_num
        # init transformer layers
        self.num_heads = num_head
        self.num_layers = decoder_layer_num
        # T_E: tracking existing objects
        self.transformer_self_attention_layers_E = nn.ModuleList()
        self.transformer_cross_attention_layers_E = nn.ModuleList()
        self.transformer_ffn_layers_E = nn.ModuleList()

        for _ in range(self.num_layers):
            # T_E
            self.transformer_self_attention_layers_E.append(
                SelfAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
            self.transformer_cross_attention_layers_E.append(
                ReferringCrossAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
            self.transformer_ffn_layers_E.append(
                FFNLayer(
                    d_model=hidden_channel,
                    dim_feedforward=feedforward_channel,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_channel)

        self.ref_proj_E = MLP(hidden_channel, hidden_channel, hidden_channel, 3)
        for layer in self.ref_proj_E.layers:
            weight_init.c2_xavier_fill(layer)
            
        # init heads
        self.class_embed = nn.Linear(hidden_channel, class_num + 1)
        self.mask_embed = MLP(hidden_channel, hidden_channel, mask_dim, 3)

        # record previous frame information
        self.instance_memory = None
        self.occupancy_memory = None
        
    def _clear_memory(self):
        self.instance_memory = None
        self.occupancy_memory = None
        return
        
    def forward(self, frame_embeds, mask_features, fg_prob, is_fg, resume=False, return_indices=False,
                frame_embeds_no_norm=None, image_outputs=None, early_matching=False):
        """
        :param frame_embeds: the instance queries output by the segmenter
        :param mask_features: the mask features output by the segmenter
        :param resume: whether the first frame is the start of the video
        :param return_indices: whether return the match indices
        :return: output dict, including masks, classes, embeds.
        """
        
        mask_features_shape = mask_features.shape
        
        frame_embeds = frame_embeds.permute(2, 3, 0, 1)  # t, q, b, c
        if frame_embeds_no_norm is not None:
            frame_embeds_no_norm = frame_embeds_no_norm.permute(2, 3, 0, 1)  # t, q, b, c
        n_frame, n_q, bs, _ = frame_embeds.size()
        ret_indices = [] # matching indices by Hungarian algorithm
        outputs = [] # traced object embeddings
        E_outputs = [] # predict only existing objects (It doesn't contain newly appeared objects.)
        
        for i in range(n_frame):
            E_output = [] 
            single_frame_embeds = frame_embeds[i]
            if frame_embeds_no_norm is not None:
                single_frame_embeds_no_norm = frame_embeds_no_norm[i]
            else:
                single_frame_embeds_no_norm = single_frame_embeds
            
            frame_key = single_frame_embeds_no_norm
            
            single_fg_prob = fg_prob[i]  # q b 1
            curr_occupancy = is_fg[i]  # q
            
            # the first frame of a video
            if i == 0 and resume is False:
                self._clear_memory()
                self.instance_memory = single_frame_embeds
                # T_E
                for j in range(self.num_layers):
                    if j == 0:
                        E_output.append(single_frame_embeds)
                        ret_indices.append(self.match_embds(single_frame_embeds, single_frame_embeds))
                        output = self.transformer_cross_attention_layers_E[j](
                            0, self.ref_proj_E(frame_key),
                            frame_key, single_frame_embeds_no_norm,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        output = self.transformer_self_attention_layers_E[j](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                        )
                        # FFN
                        output = self.transformer_ffn_layers_E[j](
                            output
                        )
                        E_output.append(output)
                    else:
                        output = self.transformer_cross_attention_layers_E[j](
                            E_output[-1], self.ref_proj_E(E_output[-1]),
                            frame_key, single_frame_embeds_no_norm,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        output = self.transformer_self_attention_layers_E[j](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                        )
                        # FFN
                        output = self.transformer_ffn_layers_E[j](
                            output
                        )
                        E_output.append(output)               
            else:
                last_reference = self.ref_proj_E(self.instance_memory)
                for j in range(self.num_layers):
                    if j == 0:
                        E_output.append(single_frame_embeds)
                        output = self.transformer_cross_attention_layers_E[j](
                            0, last_reference, 
                            frame_key, single_frame_embeds_no_norm,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        output = self.transformer_self_attention_layers_E[j](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                        )
                        # FFN
                        output = self.transformer_ffn_layers_E[j](
                            output
                        )
                        E_output.append(output)
                    else:
                        output = self.transformer_cross_attention_layers_E[j](
                            E_output[-1], last_reference, 
                            frame_key, single_frame_embeds_no_norm,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        output = self.transformer_self_attention_layers_E[j](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                        )
                        # FFN
                        output = self.transformer_ffn_layers_E[j](
                            output
                        )
                        E_output.append(output)

                if self.training and early_matching:
                    indices = self.match_embds(self.instance_memory, single_frame_embeds)
                else: 
                    # Occupancy-guided Hungarian Matching (OHM, Al. 1)
                    curr_obj_ = self.project_embed(frame_key, single_frame_embeds_no_norm) # project to the same space for accurate matching
                    curr_obj = self.decoder_norm(curr_obj_)
                    ex_obj = self.decoder_norm(E_output[-1]) # tracked existing objects (Eq. (4))
                    
                    if self.occupancy_memory.sum() > 0: # There are cases where no object exists in the first frame
                        indices = np.zeros(self.occupancy_memory.shape[0])
                        occupied_obj = ex_obj[self.occupancy_memory]
                        
                        # (1) matching with occupied objects 
                        occupied_indices = self.match_embds(occupied_obj, curr_obj)
                        indices[(self.occupancy_memory).nonzero().squeeze().cpu().numpy()] = occupied_indices
                        
                        rest_indices = torch.ones_like(self.occupancy_memory).bool()
                        rest_indices[occupied_indices]=False
                        
                        # (2) matching with unoccupied objects 
                        if rest_indices.sum() > 0:
                            unoccupied_obj = ex_obj[self.occupancy_memory==False]
                            unoccupied_indices = self.match_embds(unoccupied_obj, curr_obj[rest_indices])
                            idx = rest_indices.nonzero().squeeze(1).cpu().numpy()[unoccupied_indices] 
                            indices[(self.occupancy_memory==False).nonzero().squeeze(1).cpu().numpy()] = idx
                            
                    else:
                        indices = self.match_embds(ex_obj, curr_obj)
                ret_indices.append(indices) 
            
            # update memory (Eq. (3))
            if i == 0 and resume is False:
                self.occupancy_memory = curr_occupancy
                outputs.append(single_frame_embeds_no_norm)
            else:
                self.occupancy_memory = torch.logical_or(self.occupancy_memory, curr_occupancy[indices])
                self.instance_memory =(1-single_fg_prob[indices])*self.instance_memory + single_fg_prob[indices] * single_frame_embeds[indices]
                outputs.append(single_frame_embeds_no_norm[indices])
            
            E_output = torch.stack(E_output, dim=0)  # (layers, q, b, c)
            E_outputs.append(E_output[1:])
            
        E_outputs = torch.stack(E_outputs, dim=0)  # (t, l, q, b, c)
        outputs = torch.stack(outputs, dim=0)  # (t,q,b,c)
        
        E_outputs_class, E_outputs_masks = self.prediction(E_outputs, mask_features)
        out = {
           'pred_logits_E': E_outputs_class[-1].transpose(1, 2),  # (b, t, q, c)
           'pred_masks_E': E_outputs_masks[-1],  # (b, q, t, h, w)
           'aux_outputs_E': self._set_aux_loss(
               E_outputs_class, E_outputs_masks
           ),
           'pred_embds': outputs.permute(2, 3, 0, 1),  # (b, c, t, q)
        }
        if return_indices:
            return out, ret_indices
        else:
            return out

    def match_embds(self, ref_embds, cur_embds):
        #  embeds (q, b, c)
        ref_embds, cur_embds = ref_embds.detach()[:, 0, :], cur_embds.detach()[:, 0, :]
        ref_embds = ref_embds / (ref_embds.norm(dim=1)[:, None] + 1e-6)
        cur_embds = cur_embds / (cur_embds.norm(dim=1)[:, None] + 1e-6)
        cos_sim = torch.mm(cur_embds, ref_embds.transpose(0, 1))
        C = 1 - cos_sim

        C = C.cpu()
        C = torch.where(torch.isnan(C), torch.full_like(C, 0), C)

        indices = linear_sum_assignment(C.transpose(0, 1))
        indices = indices[1]
        return indices

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a.transpose(1, 2), "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]

    def prediction(self, outputs, mask_features):
        # outputs (t, l, q, b, c)
        # mask_features (b, t, c, h, w)
        decoder_output = self.decoder_norm(outputs)
        decoder_output = decoder_output.permute(1, 3, 0, 2, 4)  # (l, b, t, q, c)
        outputs_class = self.class_embed(decoder_output).transpose(2, 3)  # (l, b, q, t, cls+1)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("lbtqc,btchw->lbqthw", mask_embed, mask_features)
        return outputs_class, outputs_mask

    def project_embed(self, frame_key, single_frame_embeds_no_norm):
        for j in range(self.num_layers):
            if j == 0:
                output = self.transformer_cross_attention_layers_E[j](
                    0, self.ref_proj_E(frame_key),
                    frame_key, single_frame_embeds_no_norm,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None, query_pos=None
                )
                output = self.transformer_self_attention_layers_E[j](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=None
                )
                # FFN
                output = self.transformer_ffn_layers_E[j](
                    output
                )
            else:
                output = self.transformer_cross_attention_layers_E[j](
                    output, self.ref_proj_E(output),
                    frame_key, single_frame_embeds_no_norm,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None, query_pos=None
                )
                output = self.transformer_self_attention_layers_E[j](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=None
                )
                # FFN
                output = self.transformer_ffn_layers_E[j](
                    output
                )
        return output


# Full LOMM trackers (T_E + T_A)
class LOMM_Tracker(torch.nn.Module):
    def __init__(
        self,
        hidden_channel=256,
        feedforward_channel=2048,
        num_head=8,
        decoder_layer_num=6,
        mask_dim=256,
        class_num=25,
    ):
        super(LOMM_Tracker, self).__init__()

        self.hidden_channel = hidden_channel
        self.num_classes = class_num
        # init transformer layers
        self.num_heads = num_head
        self.num_layers = decoder_layer_num // 2
        # T_E: tracking existing objects
        self.transformer_self_attention_layers_E = nn.ModuleList()
        self.transformer_cross_attention_layers_E = nn.ModuleList()
        self.transformer_ffn_layers_E = nn.ModuleList()
        # T_A: tracking all objects
        self.transformer_self_attention_layers_A = nn.ModuleList()
        self.transformer_cross_attention_layers_A = nn.ModuleList()
        self.transformer_ffn_layers_A = nn.ModuleList()

        for _ in range(self.num_layers):
            # T_E
            self.transformer_self_attention_layers_E.append(
                SelfAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
            self.transformer_cross_attention_layers_E.append(
                ReferringCrossAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
            self.transformer_ffn_layers_E.append(
                FFNLayer(
                    d_model=hidden_channel,
                    dim_feedforward=feedforward_channel,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
            # T_A
            self.transformer_self_attention_layers_A.append(
                SelfAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
            self.transformer_cross_attention_layers_A.append(
                ReferringCrossAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
            self.transformer_ffn_layers_A.append(
                FFNLayer(
                    d_model=hidden_channel,
                    dim_feedforward=feedforward_channel,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_channel)

        self.ref_proj = MLP(hidden_channel, hidden_channel, hidden_channel, 3)
        for layer in self.ref_proj.layers:
            weight_init.c2_xavier_fill(layer)

        # init heads
        self.class_embed = nn.Linear(hidden_channel, class_num + 1)
        self.mask_embed = MLP(hidden_channel, hidden_channel, mask_dim, 3)

        # record previous frame information
        self.instance_memory_tracker = None
        self.instance_memory_segmenter = None
        self.occupancy_memory = None

    def _clear_memory(self):
        del self.instance_memory_tracker
        self.instance_memory_tracker = None
        self.instance_memory_segmenter = None
        self.occupancy_memory = None
        return

    def forward(self, frame_embeds, mask_features, fg_prob, resume=False, return_indices=False,
                frame_embeds_no_norm=None, early_matching=False):
        """
        :param frame_embeds: the instance queries output by the segmenter
        :param mask_features: the mask features output by the segmenter
        :param resume: whether the first frame is the start of the video
        :param return_indices: whether return the match indices
        :return: output dict, including masks, classes, embeds.
        """
        
        mask_features_shape = mask_features.shape
        
        frame_embeds = frame_embeds.permute(2, 3, 0, 1)  # t, q, b, c
        if frame_embeds_no_norm is not None:
            frame_embeds_no_norm = frame_embeds_no_norm.permute(2, 3, 0, 1)  # t, q, b, c
        n_frame, n_q, bs, _ = frame_embeds.size()
        E_outputs = [] # predict only existing objects (It doesn't contain newly appeared objects.)
        A_outputs = [] # predict all objects
        ret_indices = [] # matching indices by Hungarian algorithm
        
        all_sim_scores = [] # similarity between object embeddings (See Eq. (9))

        for i in range(n_frame):
            E_output = [] 
            A_output = [] 
            single_frame_embeds = frame_embeds[i]
            if frame_embeds_no_norm is not None:
                single_frame_embeds_no_norm = frame_embeds_no_norm[i]
            else:
                single_frame_embeds_no_norm = single_frame_embeds
            
            frame_key = single_frame_embeds_no_norm
            
            single_fg_prob = fg_prob[i]  # q b 1
            
            # the first frame of a video
            if i == 0 and resume is False:
                self._clear_memory()
                self.instance_memory_segmenter = single_frame_embeds
                # T_E
                for j in range(self.num_layers):
                    if j == 0:
                        E_output.append(single_frame_embeds)
                        ret_indices.append(self.match_embds(single_frame_embeds, single_frame_embeds))
                        output = self.transformer_cross_attention_layers_E[j](
                            0, self.ref_proj(frame_key),
                            frame_key, single_frame_embeds_no_norm,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        output = self.transformer_self_attention_layers_E[j](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                        )
                        # FFN
                        output = self.transformer_ffn_layers_E[j](
                            output
                        )
                        E_output.append(output)
                    else:
                        output = self.transformer_cross_attention_layers_E[j](
                            E_output[-1], self.ref_proj(E_output[-1]),
                            frame_key, single_frame_embeds_no_norm,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        output = self.transformer_self_attention_layers_E[j](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                        )
                        # FFN
                        output = self.transformer_ffn_layers_E[j](
                            output
                        )
                        E_output.append(output)
                
                # to make similar space
                sim = F.cosine_similarity(E_output[-1], single_frame_embeds_no_norm, dim=-1).unsqueeze(-1)
                all_sim_scores.append(sim)
                
                # T_A
                for j in range(self.num_layers):
                    if j == 0:
                        A_output.append(single_frame_embeds)
                        output = self.transformer_cross_attention_layers_A[j](
                            0, self.ref_proj(frame_key), 
                            frame_key, single_frame_embeds_no_norm,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        output = self.transformer_self_attention_layers_A[j](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                        )
                        # FFN
                        output = self.transformer_ffn_layers_A[j](
                            output
                        )
                        A_output.append(output)
                    else:
                        output = self.transformer_cross_attention_layers_A[j](
                            A_output[-1], self.ref_proj(A_output[-1]), 
                            frame_key, single_frame_embeds_no_norm,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        output = self.transformer_self_attention_layers_A[j](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                        )
                        # FFN
                        output = self.transformer_ffn_layers_A[j](
                            output
                        )
                        A_output.append(output)
            else:
                last_reference = self.ref_proj(self.instance_memory_tracker)
                for j in range(self.num_layers):
                    if j == 0:
                        E_output.append(single_frame_embeds)
                        indices = self.match_embds(self.instance_memory_segmenter, single_frame_embeds)
                        self.instance_memory_segmenter = (1-single_fg_prob[indices]) * self.instance_memory_segmenter + single_fg_prob[indices] * single_frame_embeds[indices]
                        ret_indices.append(indices)
                        output = self.transformer_cross_attention_layers_E[j](
                            0, last_reference, 
                            frame_key, single_frame_embeds_no_norm,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        output = self.transformer_self_attention_layers_E[j](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                        )
                        # FFN
                        output = self.transformer_ffn_layers_E[j](
                            output
                        )
                        E_output.append(output)
                    else:
                        output = self.transformer_cross_attention_layers_E[j](
                            E_output[-1], last_reference, 
                            frame_key, single_frame_embeds_no_norm,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        output = self.transformer_self_attention_layers_E[j](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                        )
                        # FFN
                        output = self.transformer_ffn_layers_E[j](
                            output
                        )
                        E_output.append(output)
                
                # Occupancy-guided Hungarian Matching (OHM, Al. 1)
                curr_obj_ = self.project_embed(frame_key, single_frame_embeds_no_norm) # project to the same space for accurate matching
                curr_obj = self.decoder_norm(curr_obj_)
                E_obj = self.decoder_norm(E_output[-1]) # tracked existing objects (Eq. (4))
                if self.training and early_matching:
                    matched_query = curr_obj_[indices]
                else:
                    if self.occupancy_memory.sum() > 0: # There are cases where no object exists in the first frame
                        occupied_obj = E_obj[self.occupancy_memory]
                        unoccupied_obj = E_obj[self.occupancy_memory==False]
                        
                        # (1) matching with occupied objects 
                        occupied_indices = self.match_embds(occupied_obj, curr_obj)
                        rest_indices = torch.ones_like(self.occupancy_memory).bool()
                        rest_indices[occupied_indices]=False
                        
                        # (2) matching with unoccupied objects 
                        unoccupied_indices = self.match_embds(unoccupied_obj, curr_obj[rest_indices])
                        
                        matched_query = curr_obj_.clone()
                        matched_query[self.occupancy_memory] = curr_obj_[occupied_indices]
                        matched_query[self.occupancy_memory==False] = curr_obj_[rest_indices][unoccupied_indices]
                        
                    else:
                        indices = self.match_embds(E_obj, curr_obj)
                        matched_query = curr_obj_[indices]
                    
                sim = F.cosine_similarity(matched_query, self.decoder_norm(self.instance_memory_tracker), dim=-1).unsqueeze(-1)
                all_sim_scores.append(sim)
                # adaptive anchor query (Eq. (6))
                A_query = sim *self.instance_memory_tracker + (1-sim) * matched_query
                last_reference = self.ref_proj(A_query)
                
                # T_A
                for j in range(self.num_layers):
                    if j == 0:
                        A_output.append(single_frame_embeds[..., :self.hidden_channel])
                        output = self.transformer_cross_attention_layers_A[j](
                            0, last_reference, 
                            frame_key, single_frame_embeds_no_norm,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        output = self.transformer_self_attention_layers_A[j](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                        )
                        # FFN
                        output = self.transformer_ffn_layers_A[j](
                            output
                        )
                        A_output.append(output)
                    else:
                        output = self.transformer_cross_attention_layers_A[j](
                            A_output[-1], last_reference, 
                            frame_key, single_frame_embeds_no_norm,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        output = self.transformer_self_attention_layers_A[j](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                        )
                        # FFN
                        output = self.transformer_ffn_layers_A[j](
                            output
                        )
                        A_output.append(output)
            E_output = torch.stack(E_output, dim=0)  # (layers, q, b, c)
            A_output = torch.stack(A_output, dim=0)  # (layers, q, b, c)
            
            decoder_output = self.decoder_norm(A_output[-1])  # q b c
            class_output = self.class_embed(decoder_output)  # q b cls+1
            bg_prob = F.softmax(class_output, dim=-1)[..., -1:]  # q b 1
            curr_is_fg = (class_output.detach().max(dim=-1)[1] != self.num_classes).squeeze()  # q
            
            # update memory (Eq. (3))
            if i == 0 and resume is False:
                self.instance_memory_tracker = A_output[-1]
                self.occupancy_memory = curr_is_fg
            else:
                self.instance_memory_tracker = bg_prob * self.instance_memory_tracker + (1-bg_prob) * A_output[-1]
                self.occupancy_memory = torch.logical_or(self.occupancy_memory, curr_is_fg)
                
            E_outputs.append(E_output[1:])
            A_outputs.append(A_output[1:])
        E_outputs = torch.stack(E_outputs, dim=0)  # (t, l, q, b, c)
        A_outputs = torch.stack(A_outputs, dim=0)  # (t, l, q, b, c)
        
        mask_features_ = mask_features
        if not self.training:
            A_outputs = A_outputs[:, -1:]
            del mask_features
        
        E_outputs_class, E_outputs_masks = self.prediction(E_outputs, mask_features_)
        A_outputs_class, A_outputs_masks = self.prediction(A_outputs, mask_features_)
        outputs = self.decoder_norm(A_outputs)
        out = {
           'pred_logits': A_outputs_class[-1].transpose(1, 2),  # (b, t, q, c)
           'pred_masks': A_outputs_masks[-1],  # (b, q, t, h, w)
           'pred_logits_E': E_outputs_class[-1].transpose(1, 2),  # (b, t, q, c)
           'pred_masks_E': E_outputs_masks[-1],  # (b, q, t, h, w)
           'aux_outputs': self._set_aux_loss(
               A_outputs_class, A_outputs_masks
           ),
           'aux_outputs_E': self._set_aux_loss(
               E_outputs_class, E_outputs_masks
           ),
           'pred_embds': outputs[:, -1].permute(2, 3, 0, 1),  # (b, c, t, q)
           'sim_scores': all_sim_scores,
        }
        if return_indices:
            return out, ret_indices
        else:
            return out

    def match_embds(self, ref_embds, cur_embds):
        #  embeds (q, b, c)
        ref_embds, cur_embds = ref_embds.detach()[:, 0, :], cur_embds.detach()[:, 0, :]
        ref_embds = ref_embds / (ref_embds.norm(dim=1)[:, None] + 1e-6)
        cur_embds = cur_embds / (cur_embds.norm(dim=1)[:, None] + 1e-6)
        cos_sim = torch.mm(cur_embds, ref_embds.transpose(0, 1))
        C = 1 - cos_sim

        C = C.cpu()
        C = torch.where(torch.isnan(C), torch.full_like(C, 0), C)

        indices = linear_sum_assignment(C.transpose(0, 1))
        indices = indices[1]
        return indices

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a.transpose(1, 2), "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]

    def prediction(self, outputs, mask_features):
        # outputs (t, l, q, b, c)
        # mask_features (b, t, c, h, w)
        decoder_output = self.decoder_norm(outputs)
        decoder_output = decoder_output.permute(1, 3, 0, 2, 4)  # (l, b, t, q, c)
        outputs_class = self.class_embed(decoder_output).transpose(2, 3)  # (l, b, q, t, cls+1)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("lbtqc,btchw->lbqthw", mask_embed, mask_features)
        return outputs_class, outputs_mask

    def project_embed(self, frame_key, single_frame_embeds_no_norm):
        for j in range(self.num_layers):
            if j == 0:
                output = self.transformer_cross_attention_layers_A[j](
                    0, self.ref_proj(frame_key),
                    frame_key, single_frame_embeds_no_norm,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None, query_pos=None
                )
                output = self.transformer_self_attention_layers_A[j](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=None
                )
                # FFN
                output = self.transformer_ffn_layers_A[j](
                    output
                )
            else:
                output = self.transformer_cross_attention_layers_A[j](
                    output, self.ref_proj(output),
                    frame_key, single_frame_embeds_no_norm,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=None, query_pos=None
                )
                output = self.transformer_self_attention_layers_A[j](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=None
                )
                # FFN
                output = self.transformer_ffn_layers_A[j](
                    output
                )
        return output


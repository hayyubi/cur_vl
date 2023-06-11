import pdb
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from maskrcnn_benchmark.utils.logged_module import LoggedModule

def wrap_nd_batch(fn, tensor):
    s = list(tensor.shape)
    tensor = tensor.reshape([-1, s[-1]])
    result = fn(tensor)
    s2 = list(result.shape)
    assert np.prod(s[:-1]) == s2[0]
    result = result.reshape(s[:-1] + s2[1:])
    return result

def choose_one(tensor):
    return wrap_nd_batch(
        lambda x: torch.multinomial(x, num_samples=1).squeeze(-1),
        tensor
    )

def remove_diag(square_matrix, dim):
    '''
    Removes the diagonal from a given matrix.
    Input is an NxN torch.Tensor.
    Returns an Nx(N-1) or (N-1)xN tensor depending on dim (1 or 0 respectively)
    '''
    assert(len(square_matrix.shape) == 2 and
           square_matrix.shape[0] == square_matrix.shape[1] and
           dim in [0, 1])
    N = square_matrix.shape[0]
    mask = (1-torch.eye(N)).to(torch.bool).to('cuda')
    if dim == 1:
        return torch.masked_select(square_matrix, mask).reshape([N, N - 1])
    if dim == 0:
        return torch.masked_select(square_matrix.t(), mask).reshape([N, N - 1]).t()

class NotCurriculumAwareLoss(Exception):
    pass

class GroundingHead(LoggedModule):
    def __init__(self, config, v_dim, l_dim):
        super(GroundingHead, self).__init__()
        self.config = config.MODEL.MMSS_HEAD.GROUNDING
        self.v_dim = v_dim
        self.l_dim = l_dim
        self.v2l_projection = nn.Linear(self.v_dim, self.l_dim)

        # local similarity/distance metric can be either 'dot' or 'cosine' or 'euclidean'
        self.local_metric = self.config.LOCAL_METRIC

        # global distance metric can be either reconstruction_mse or aligned_local
        self.global_metric = self.config.GLOBAL_METRIC

        # word to region alignment method can be either 
        # 'softmax' or 'hardmax' or 'random_top3' or random_categorical
        self.alignment = self.config.ALIGNMENT

        # Initial iteration percentage to start using new loss
        self.alignment_curriculum_t0 = 0.3

        # Alignment curriculum is used to implement curriculum aware loss
        self.alignment_curriculum = self.config.ALIGNMENT_CURRICULUM
        # Previous model to extract knowledge from
        self.prev_knowledge_model = None
        # Current model being used as a knowledge source for future
        if len(config.CURRICULUM.ITERS) > 0:
            self.prev_knowledge_extraction_iter_ratio = config.CURRICULUM.ITERS[0] / config.SOLVER.MAX_ITER
                                                                
        # When to start using knowledge from previous source
        self.prev_knowledge_source_mode=False

        # Before this percent iteration, use original cross_entropy loss
        self.curriculum_aware_triplet_bag_loss_threshold = 0.3
        
        # temperature is used as a denominator for the exponent of softmax, to make it smoother
        self.temperature = self.config.ALIGNMENT_TEMPERATURE

        # loss type can either be 'matching' or 'cross_entropy' or 'triplet'
        self.loss_type = self.config.LOSS

        # for triplet loss, negative mining method can be 'hardest', 'easiest', or 'random'
        self.negative_mining = self.config.NEGATIVE_MINING

        # distance margin for triplet loss
        self.margin = self.config.TRIPLET_MARGIN

        # whether to align each visual region to all caption words, or vice versa, or both
        self.align_words = self.config.ALIGN_WORDS_TO_REGIONS
        self.align_regions = self.config.ALIGN_REGIONS_TO_WORDS
        assert(self.align_words or self.align_regions)

        # Epsilon value for scaling curriculum aware alignment scores, change to 
        # MODEL.MMSS_HEAD.GROUNDING.CAA_EPSILON
        self.epsilon = 0.0001


    def forward(self, input_image, input_caption, **kwargs):
        iter_percent = kwargs['iter_percent'] if 'iter_percent' in kwargs else None
        images = kwargs['images'] if 'images' in kwargs else None
        targets = kwargs['targets'] if 'targets' in kwargs else None
        prev_knowledge = kwargs['prev_knowledge'] if 'prev_knowledge' in kwargs else None

        caption_emb = input_caption['grounded_input_embeddings']
        caption_mask = input_caption['grounded_attention_mask'] * (1 - input_caption['grounded_special_tokens_mask'])
        self.log('attention_mask', input_caption['grounded_attention_mask'])
        self.log('special_tokens_mask', input_caption['grounded_special_tokens_mask'])
        self.log('caption_mask', caption_mask)
        self.log('caption_emb', caption_emb)
        caption_mask = caption_mask.to(torch.float32)
        num_words = caption_mask.sum(dim=1)
        _, max_num_words = caption_mask.shape

        region_features = input_image['region_features']
        region_mask = input_image['region_mask']
        region_mask = region_mask.to(torch.float32)
        num_regions = region_mask.sum(dim=1)
        batch_size, max_num_regions, _ = region_features.shape

        image_emb = self.v2l_projection(region_features).permute(0, 2, 1)

        if self.loss_type == 'cross_entropy' or 'triplet' in self.loss_type:
            # we should compute the image-sentence distances for all image-sentence pairs 
            # in the batch, rather than only matching ones. So we replicate them BxB times.
            image_emb = image_emb[None, :, :, :].repeat(batch_size, 1, 1, 1).reshape(
                batch_size**2, self.l_dim, max_num_regions)
            caption_emb = caption_emb[:, None, :, :].repeat(1, batch_size, 1, 1).reshape(
                batch_size**2, max_num_words, self.l_dim)
            region_mask = region_mask[None, :, :].repeat(batch_size, 1, 1).reshape(
                batch_size**2, max_num_regions)
            caption_mask = caption_mask[:, None, :].repeat(1, batch_size, 1).reshape(
                batch_size**2, max_num_words)
            num_regions = num_regions[None, :].repeat(batch_size, 1).reshape(
                batch_size**2)
            num_words = num_words[:, None].repeat(1, batch_size).reshape(
                batch_size**2)

        if self.local_metric == 'dot':
            local_similarity = torch.bmm(caption_emb, image_emb)
            local_distance = - local_similarity
        elif self.local_metric == 'cosine':
            local_similarity = torch.bmm(caption_emb, image_emb)
            i_norm = (image_emb ** 2).sum(dim=1, keepdim=True).sqrt()
            c_norm = (caption_emb ** 2).sum(dim=2, keepdim=True).sqrt()
            local_similarity = local_similarity / (i_norm * c_norm)
            local_similarity = torch.where(
                torch.isnan(local_similarity), 
                torch.zeros_like(local_similarity), 
                local_similarity)
            local_distance = 1 - local_similarity
        elif self.local_metric == 'euclidean':
            local_similarity = torch.bmm(caption_emb, image_emb)
            i_norm = (image_emb ** 2).sum(dim=1, keepdim=True)
            c_norm = (caption_emb ** 2).sum(dim=2, keepdim=True)
            local_distance = i_norm + c_norm - (2 * local_similarity)
            # This implementation takes too much memory:
            # local_distance = ((image_emb[:, None, :, :] - caption_emb[:, :, :, None]) ** 2).sum(dim=2)
            local_similarity = - local_distance
        else:
            raise NotImplementedError

        local_similarity = local_similarity / self.temperature
        local_distance = local_distance / self.temperature

        self.log('local_similarity', local_similarity)
        local_similarity = torch.where(
            (caption_mask[:, :, None] * region_mask[:, None, :]) > 0,
            local_similarity, 
            local_similarity.min().detach() - 100.0
        )

        if self.alignment == 'softmax':
            knowledge_to_send_to_future = tuple()
            if self.align_words:
                prev_knowledge_w2r = None
                attention_w2r = F.softmax(local_similarity, dim=2)
                if self.alignment_curriculum != '' and not self.prev_knowledge_source_mode \
                        and self.training and iter_percent > self.prev_knowledge_extraction_iter_ratio:
                    if self.alignment_curriculum == 'current_model':
                        prev_knowledge_w2r = attention_w2r
                    elif self.alignment_curriculum == 'previous_model':
                        assert prev_knowledge is not None
                        assert images is not None
                        assert targets is not None
                        # prev_knowledge = prev_knowledge_model(images, targets)
                        prev_knowledge_w2r = prev_knowledge[0]
                    else:
                        raise NotImplementedError
                    attention_w2r = F.softmax( local_similarity * 
                                                torch.exp(-1 * torch.max(prev_knowledge_w2r, 2, keepdim=True)[0] * iter_percent),
                                                dim=2)

                if self.prev_knowledge_source_mode:
                    knowledge_to_send_to_future += (attention_w2r,)

            if self.align_regions:
                prev_knowledge_r2w = None
                attention_r2w = F.softmax(local_similarity, dim=1)
                if self.alignment_curriculum != '' and not self.prev_knowledge_source_mode \
                    and self.training and iter_percent > self.prev_knowledge_extraction_iter_ratio:
                    if self.alignment_curriculum == 'current_model':
                        prev_knowledge_r2w = attention_r2w
                    elif self.alignment_curriculum == 'previous_model':
                        if self.align_words:
                            prev_knowledge_r2w = prev_knowledge[1]
                        else:
                            assert prev_knowledge is not None
                            assert images is not None
                            assert targets is not None
                            # prev_knowledge = prev_knowledge_model(images, targets)
                            prev_knowledge_r2w = prev_knowledge[0]
                    else:
                        raise NotImplementedError
                    attention_r2w = F.softmax( local_similarity * 
                                                torch.exp(-1 * torch.max(prev_knowledge_r2w, 1, keepdim=True)[0] * iter_percent),
                                                dim=1)

                if self.prev_knowledge_source_mode:
                    knowledge_to_send_to_future += (attention_r2w,)

        elif self.alignment == 'hardmax':
            if self.align_words:
                idx = torch.argmax(local_similarity, dim=2)
                attention_w2r = F.one_hot(idx, max_num_regions).to(torch.float32)
            if self.align_regions:
                idx = torch.argmax(local_similarity, dim=1)
                attention_r2w = F.one_hot(idx, max_num_words).to(torch.float32).permute(0, 2, 1)
        elif self.alignment == 'random_categorical':
            if self.align_words:
                attention_w2r = F.softmax(local_similarity, dim=2)
                idx = choose_one(attention_w2r)
                attention_w2r = F.one_hot(idx, max_num_regions).to(torch.float32)
            if self.align_regions:
                attention_r2w = F.softmax(local_similarity, dim=1).permute(0, 2, 1)
                idx = choose_one(attention_r2w)
                attention_r2w = F.one_hot(idx, max_num_words).to(torch.float32).permute(0, 2, 1)
        elif self.alignment == 'random_top3':
            if self.align_words:
                idx = torch.topk(local_similarity, k=3, dim=2).indices
                attention_w2r = F.one_hot(idx, max_num_regions).to(torch.float32).sum(dim=2)
                idx = choose_one(attention_w2r)
                attention_w2r = F.one_hot(idx, max_num_regions).to(torch.float32)
            if self.align_regions:
                idx = torch.topk(local_similarity, k=3, dim=1).indices
                attention_r2w = F.one_hot(idx, max_num_words).to(torch.float32).sum(dim=1)
                idx = choose_one(attention_r2w)
                attention_r2w = F.one_hot(idx, max_num_words).to(torch.float32).permute(0, 2, 1)
        elif self.alignment == 'optimal_transport':
            # TODO
            raise NotImplementedError
        else:
            raise NotImplementedError

        if self.prev_knowledge_source_mode:
            return knowledge_to_send_to_future

        if self.align_words:
            self.log('attention_w2r', attention_w2r)
        if self.align_regions:
            self.log('attention_r2w', attention_r2w)

        try:
            if self.loss_type == "curriculum_aware_triplet":
                # If curriculum aware triplet training
                if iter_percent > self.curriculum_aware_triplet_bag_loss_threshold:
                    self.loss_type = "curriculum_aware_triplet"

                # if iter_percent < self.curriculum_aware_triplet_bag_loss_threshold:
                #     self.loss_type = "cross_entropy"
                #     raise NotCurriculumAwareLoss
                losses={}
                other_info={}
                pos_neg_distance = local_distance.reshape(
                    batch_size, batch_size, max_num_words, max_num_regions)
                pos_distance = torch.diagonal(pos_neg_distance, dim1=0, dim2=1).permute(2,0,1)
                # Similarity mask to only take dist metric between valid cap and img region
                sim_mask = (caption_mask[:, :, None] * region_mask[:, None, :]).reshape(
                    batch_size, batch_size, max_num_words, max_num_regions)
                sim_mask = sim_mask.to(torch.bool)
                pos_sim_mask = torch.diagonal(sim_mask, dim1=0, dim2=1).permute(2,0,1)
                if batch_size < 2:
                    if self.align_words:
                        # TODO: Add attention here.
                        cost_w2r = torch.max(pos_distance + self.margin, other=torch.zeros_like(pos_distance))
                        cost_w2r = torch.masked_select(cost_w2r, pos_sim_mask)
                    if self.align_regions:
                        # TODO: Add attention here.
                        cost_r2w = torch.max(pos_distance + self.margin, other=torch.zeros_like(pos_distance))
                        cost_r2w = torch.masked_select(cost_r2w, pos_sim_mask) 
                else:
                    # Mask to get non diagonal distance matrix
                    diag_mask = torch.zeros_like(pos_neg_distance)
                    idx = torch.arange(batch_size).to(diag_mask.device)
                    diag_mask[idx, idx, :, :] = 1
                    diag_mask = 1 - diag_mask
                    diag_mask = diag_mask.to(torch.bool)
                    # Max num of words and regions for positive pairs
                    num_words = num_words.reshape(batch_size, batch_size)
                    pos_num_words = torch.diagonal(num_words, dim1=0, dim2=1)
                    num_regions = num_regions.reshape(batch_size, batch_size)
                    pos_num_regions = torch.diagonal(num_regions, dim1=0, dim2=1)
                    # From word perspective
                    if self.align_words:
                        if self.local_metric == 'dot' or self.local_metric == 'euclidean':
                            max_region_scores = - torch.min(pos_distance, dim=1, keepdim=True).values
                            scale_region_score = torch.max(max_region_scores)
                            region_inverse_scaling = (scale_region_score - max_region_scores + self.epsilon) / \
                                                        torch.abs(scale_region_score)
                        elif self.local_metric == 'cosine':
                            max_region_scores = 1 - torch.min(pos_distance, dim=1, keepdim=True).values * self.temperature
                            scale_region_score = torch.max(max_region_scores)
                            region_inverse_scaling = (np.pi - torch.acos(scale_region_score) + self.epsilon) / \
                                                        (np.pi - torch.acos(max_region_scores) + self.epsilon)
                        else:
                            NotImplementedError
                        region_inverse_scaling = region_inverse_scaling.repeat(1, max_num_words, 1)
                        ## Comparing each (Batch sz) positive img-cap pair to all negative
                        ## img-cap pairs (Batch sz - 1)
                        neg_distance_w2r = torch.masked_select(pos_neg_distance, diag_mask).reshape(
                                            [batch_size, batch_size - 1, max_num_words, max_num_regions])
                        neg_sim_mask_w2r = torch.masked_select(sim_mask, diag_mask).reshape(
                                            [batch_size, batch_size - 1, max_num_words, max_num_regions])
                        pos_distance_w2r = pos_distance[:, None, :, :].repeat(1, batch_size-1, 1, 1)
                        pos_sim_mask_w2r = pos_sim_mask[:, None, :, :].repeat(1, batch_size-1, 1, 1)
                        region_inverse_scaling = region_inverse_scaling[:, None, :, :].repeat(1, batch_size-1, 1, 1)
                        ## Comparing each positive {word}-{image region} pair to all negative
                        ## {same word}-{all image region} pairs in the negative image
                        neg_distance_w2r = neg_distance_w2r.repeat(1, 1, 1, max_num_regions)
                        neg_sim_mask_w2r = neg_sim_mask_w2r.repeat(1, 1, 1, max_num_regions)
                        pos_distance_w2r = pos_distance_w2r[:,:,:,:,None].repeat(1,1,1,1,max_num_regions).reshape(
                                                    batch_size, batch_size-1, max_num_words, max_num_regions**2)
                        pos_sim_mask_w2r = pos_sim_mask_w2r[:,:,:,:,None].repeat(1,1,1,1,max_num_regions).reshape(
                                                    batch_size, batch_size-1, max_num_words, max_num_regions**2)
                        region_inverse_scaling = region_inverse_scaling[:,:,:,:,None].repeat(1,1,1,1,max_num_regions).reshape(
                                                    batch_size, batch_size-1, max_num_words, max_num_regions**2)
                        ## Calculating cost
                        cost_w2r = F.relu(- neg_distance_w2r + pos_distance_w2r + self.margin)
                        # cost_w2r = region_inverse_scaling * cost_w2r
                        ## Reduction: start doing mean and going backwords
                        net_mask = (pos_sim_mask_w2r * neg_sim_mask_w2r).to(torch.float)
                        cost_w2r = cost_w2r * net_mask
                        cost_w2r = cost_w2r.reshape(batch_size, batch_size-1, max_num_words, max_num_regions, max_num_regions)
                        cost_w2r = torch.sum(cost_w2r, (1,4)) # (batch_size, max_num_words, max_num_regions)
                        net_mask = net_mask.reshape(batch_size, batch_size-1, max_num_words, max_num_regions, max_num_regions)
                        net_mask = torch.sum(net_mask, (1,4))
                        cost_w2r = cost_w2r / torch.max(net_mask, torch.ones_like(net_mask)) # (batch_size, max_num_words, max_num_regions)
                        ## Positive attention multiplied
                        attention_w2r = attention_w2r.reshape(batch_size, batch_size, max_num_words, max_num_regions)
                        pos_attention_w2r = torch.diagonal(attention_w2r, dim1=0, dim2=1).permute(2,0,1)
                        pos_attention_w2r = pos_attention_w2r * pos_sim_mask.to(torch.float)
                        cost_w2r = torch.sum(cost_w2r * pos_attention_w2r, 2) # (batch_size, max_num_words)
                        cost_w2r = cost_w2r.sum(dim=1) / torch.max(pos_num_words, other=torch.ones_like(pos_num_words)) # (batch_size,)
                        cost_w2r = torch.masked_select(cost_w2r, (pos_num_words > 0) * (pos_num_regions > 0))
                    if self.align_regions:
                        # From image region perspective
                        if self.local_metric == 'dot' or self.local_metric == 'euclidean':
                            max_word_scores = - torch.min(pos_distance, dim=2, keepdim=True).values
                            scale_word_score = torch.max(max_word_scores)
                            word_inverse_scaling = (scale_word_score - max_word_scores + self.epsilon) / \
                                                        torch.abs(scale_word_score)
                        elif self.local_metric == 'cosine':
                            max_word_scores = 1 - torch.min(pos_distance, dim=2, keepdim=True).values * self.temperature
                            scale_word_score = torch.max(max_word_scores)
                            word_inverse_scaling = (np.pi - torch.acos(scale_word_score + self.epsilon)) / \
                                                        (np.pi - torch.acos(max_word_scores) + self.epsilon)
                        else:
                            NotImplementedError
                        word_inverse_scaling = word_inverse_scaling.repeat(1,1,max_num_regions)
                        ## Comparing each (Batch sz) positive img-cap pair to all negative
                        ## img-cap pairs (Batch sz - 1)
                        neg_distance_r2w = torch.masked_select(pos_neg_distance.transpose(0,1), diag_mask).reshape(
                                      [batch_size, batch_size - 1, max_num_words, max_num_regions]).transpose(0,1)
                        neg_sim_mask_r2w = torch.masked_select(sim_mask.transpose(0,1), diag_mask).reshape(
                                      [batch_size, batch_size - 1, max_num_words, max_num_regions]).transpose(0,1)
                        pos_distance_r2w = pos_distance[:, None, :, :].repeat(1, batch_size-1, 1, 1).transpose(0,1)
                        pos_sim_mask_r2w = pos_sim_mask[:, None, :, :].repeat(1, batch_size-1, 1, 1).transpose(0,1)
                        word_inverse_scaling = word_inverse_scaling[:, None, :, :].repeat(1, batch_size-1, 1, 1).transpose(0,1)
                        ## Comparing each positive {word}-{image region} pair to all negative
                        ## {all word}-{same image region} pairs in the negative image
                        neg_distance_r2w = neg_distance_r2w.repeat(1, 1, max_num_words, 1)
                        neg_sim_mask_r2w = neg_sim_mask_r2w.repeat(1, 1, max_num_words, 1)
                        pos_distance_r2w = pos_distance_r2w[:,:,:,None,:].repeat(1,1,1,max_num_words,1).reshape(
                                                    batch_size-1, batch_size, max_num_words**2, max_num_regions)
                        pos_sim_mask_r2w = pos_sim_mask_r2w[:,:,:,None,:].repeat(1,1,1,max_num_words,1).reshape(
                                                    batch_size-1, batch_size, max_num_words**2, max_num_regions)
                        word_inverse_scaling = word_inverse_scaling[:,:,:,None,:].repeat(1,1,1,max_num_words,1).reshape(
                                                    batch_size-1, batch_size, max_num_words**2, max_num_regions)
                        ## Calculating cost
                        cost_r2w = F.relu(- neg_distance_r2w + pos_distance_r2w + self.margin)
                        # cost_r2w = word_inverse_scaling * cost_r2w
                        ## Reduction: start doing mean and going backwords
                        net_mask = (pos_sim_mask_r2w * neg_sim_mask_r2w).to(torch.float)
                        cost_r2w = cost_r2w * net_mask
                        cost_r2w = cost_r2w.reshape(batch_size-1, batch_size, max_num_words, max_num_words, max_num_regions)
                        cost_r2w = torch.sum(cost_r2w, (0,3)) # (batch_size, max_num_words, max_num_regions)
                        net_mask = net_mask.reshape(batch_size-1, batch_size, max_num_words, max_num_words, max_num_regions)
                        net_mask = torch.sum(net_mask, (0,3))
                        # cost_r2w = cost_r2w / torch.max(net_mask, torch.ones_like(net_mask)) # (batch_size, max_num_words, max_num_regions)
                        ## Positive attention multiplied
                        attention_r2w = attention_r2w.reshape(batch_size, batch_size, max_num_words, max_num_regions)
                        pos_attention_r2w = torch.diagonal(attention_r2w, dim1=0, dim2=1).permute(2,0,1)
                        pos_attention_r2w = pos_attention_r2w * pos_sim_mask.to(torch.float)
                        cost_r2w = torch.sum(cost_r2w * pos_attention_r2w, 1) # (batch_size, max_num_regions)
                        cost_r2w = cost_r2w.sum(dim=1) / torch.max(pos_num_regions, other=torch.ones_like(pos_num_regions)) # (batch_size,)
                        cost_r2w = torch.masked_select(cost_r2w, (pos_num_words > 0) * (pos_num_regions > 0))
                if self.align_words:
                    losses['Curriulum Aware Triplet Loss (Align Words)'] = cost_w2r.mean()
                if self.align_regions:
                    losses['Curriculum Aware Triplet Loss (Align Regions)'] = cost_r2w.mean()
 
                self.log_dict(losses)
                self.log_dict(other_info)

                return other_info, losses

        # Begin training with original bag/MIL loss
        except NotCurriculumAwareLoss:
            pass
        # End of curriculum aware triplet training

        if self.global_metric == 'reconstruction_mse':
            if self.align_words:
                caption_rec = torch.bmm(attention_w2r, image_emb.transpose(1, 2))
                global_dist_w2r = ((caption_rec - caption_emb) ** 2).mean(dim=2)
                global_dist_w2r = (
                    (global_dist_w2r * caption_mask).sum(dim=1) /
                    torch.max(num_words, other=torch.ones_like(num_words))
                )
            if self.align_regions:
                image_rec = torch.bmm(caption_emb.transpose(1, 2), attention_r2w)
                global_dist_r2w = ((image_rec - image_emb) ** 2).mean(dim=2).mean(dim=1)
                global_dist_r2w = (
                    (global_dist_r2w * region_mask).sum(dim=1) /
                    torch.max(num_regions, other=torch.ones_like(num_regions))
                )

        elif self.global_metric == 'aligned_local':
            if self.align_words:
                attention_w2r = attention_w2r * caption_mask[:, :, None]
                global_dist_w2r = (
                    (attention_w2r * local_distance).sum(dim=2).sum(dim=1) /
                    torch.max(num_words, other=torch.ones_like(num_words))
                )
            if self.align_regions:
                attention_r2w = attention_r2w * region_mask[:, None, :]
                global_dist_r2w = (
                    (attention_r2w * local_distance).sum(dim=2).sum(dim=1) /
                    torch.max(num_regions, other=torch.ones_like(num_regions))
                )
        else:
            raise NotImplementedError

        if self.align_words:
            global_dist_w2r = torch.where(
                (num_words > 0) + (num_regions > 0),
                global_dist_w2r,
                global_dist_w2r.max().detach() + 100.0
            )
        if self.align_regions:
            global_dist_r2w = torch.where(
                (num_regions > 0) + (num_words > 0),
                global_dist_r2w,
                global_dist_r2w.max().detach() + 100.0
            )

        if self.align_words:
            self.log('global_dist_w2r', global_dist_w2r)
        if self.align_regions:
            self.log('global_dist_r2w', global_dist_r2w)

        losses = {}
        if self.loss_type == 'matching':
            if self.local_metric == 'dot':
                raise Exception('Matching loss is not defined for dot product\
                                 because dot product is unbounded')
            if self.align_words:
                losses['Image-Caption Matching Loss (Align Words)'] = global_dist_w2r.mean()
            if self.align_regions:
                losses['Image-Caption Matching Loss (Align Regions)'] = global_dist_r2w.mean()

        elif self.loss_type == 'cross_entropy':
            if self.align_words:
                pw_cost_w2r = global_dist_w2r.reshape(batch_size, batch_size)
                pw_logits_c_cap_w2r = torch.log_softmax(- pw_cost_w2r, dim=0)
                pw_logits_c_img_w2r = torch.log_softmax(- pw_cost_w2r, dim=1)
                losses['Cross-Entropy Loss (Align Words, Choose Caption)'] = (
                    torch.diag(- pw_logits_c_cap_w2r).mean())
                losses['Cross-Entropy Loss (Align Words, Choose Image)'] = (
                    torch.diag(- pw_logits_c_img_w2r).mean())
            if self.align_regions:
                pw_cost_r2w = global_dist_r2w.reshape(batch_size, batch_size)
                pw_logits_c_cap_r2w = torch.log_softmax(- pw_cost_r2w, dim=0)
                pw_logits_c_img_r2w = torch.log_softmax(- pw_cost_r2w, dim=1)
                losses['Cross-Entropy Loss (Align Regions, Choose Caption)'] = (
                    torch.diag(- pw_logits_c_cap_r2w).mean())
                losses['Cross-Entropy Loss (Align Regions, Choose Image)'] = (
                    torch.diag(- pw_logits_c_img_r2w).mean())

        elif self.loss_type == 'triplet':
            if self.align_words:
                pw_cost_w2r = global_dist_w2r.reshape(batch_size, batch_size)
                positive_dist_w2r = torch.diag(pw_cost_w2r)
                negative_cap_all_w2r = remove_diag(pw_cost_w2r, dim=0)
                negative_img_all_w2r = remove_diag(pw_cost_w2r, dim=1)
                if batch_size < 2:
                    negative_cap_dist_w2r = positive_dist_w2r + self.margin
                    negative_img_dist_w2r = positive_dist_w2r + self.margin
                elif self.negative_mining == 'hardest':
                    negative_cap_dist_w2r = negative_cap_all_w2r.min(dim=0).values
                    negative_img_dist_w2r = negative_img_all_w2r.min(dim=1).values
                elif self.negative_mining == 'easiest':
                    negative_cap_dist_w2r = negative_cap_all_w2r.max(dim=0).values
                    negative_img_dist_w2r = negative_img_all_w2r.max(dim=1).values
                elif self.negative_mining == 'random':
                    negative_cap_dist_w2r = negative_cap_all_w2r.gather(
                        index=torch.randint(batch_size - 1, (1, batch_size)).to('cuda'),
                        dim=0)[0, :]
                    negative_img_dist_w2r = negative_img_all_w2r.gather(
                        index=torch.randint(batch_size - 1, (batch_size, 1)).to('cuda'),
                        dim=1)[:, 0]
                losses['Triplet Loss (Align Words, Choose Caption)'] = torch.mean(
                    F.relu(positive_dist_w2r - negative_cap_dist_w2r + self.margin))
                losses['Triplet Loss (Align Words, Choose Image)'] = torch.mean(
                    F.relu(positive_dist_w2r - negative_img_dist_w2r + self.margin))
            if self.align_regions:
                pw_cost_r2w = global_dist_r2w.reshape(batch_size, batch_size)
                positive_dist_r2w = torch.diag(pw_cost_r2w)
                negative_cap_all_r2w = remove_diag(pw_cost_r2w, dim=0)
                negative_img_all_r2w = remove_diag(pw_cost_r2w, dim=1)
                if batch_size < 2:
                    negative_cap_dist_r2w = positive_dist_r2w + self.margin
                    negative_img_dist_r2w = positive_dist_r2w + self.margin
                elif self.negative_mining == 'hardest':
                    negative_cap_dist_r2w = negative_cap_all_r2w.min(dim=0).values
                    negative_img_dist_r2w = negative_img_all_r2w.min(dim=1).values
                elif self.negative_mining == 'easiest':
                    negative_cap_dist_r2w = negative_cap_all_r2w.max(dim=0).values
                    negative_img_dist_r2w = negative_img_all_r2w.max(dim=1).values
                elif self.negative_mining == 'random':
                    negative_cap_dist_r2w = negative_cap_all_r2w.gather(
                        index=torch.randint(batch_size - 1, (1, batch_size)).to('cuda'),
                        dim=0)[0, :]
                    negative_img_dist_r2w = negative_img_all_r2w.gather(
                        index=torch.randint(batch_size - 1, (batch_size, 1)).to('cuda'),
                        dim=1)[:, 0]
                losses['Triplet Loss (Align Regions, Choose Caption)'] = torch.mean(
                    F.relu(positive_dist_r2w - negative_cap_dist_r2w + self.margin))
                losses['Triplet Loss (Align Regions, Choose Image)'] = torch.mean(
                    F.relu(positive_dist_r2w - negative_img_dist_r2w + self.margin))
        else:
            raise NotImplementedError

        other_info = {}
        if self.loss_type == 'cross_entropy' or self.loss_type == 'triplet':
            if self.align_words:
                other_info['Batch Accuracy (Align Words, Choose Caption)'] = torch.mean(
                    (pw_cost_w2r.argmin(dim=0) ==
                     torch.arange(batch_size).to('cuda')
                    ).to(torch.float32))
                other_info['Batch Accuracy (Align Words, Choose Image)'] = torch.mean(
                    (pw_cost_w2r.argmin(dim=1) ==
                     torch.arange(batch_size).to('cuda')
                    ).to(torch.float32))
            if self.align_regions:
                other_info['Batch Accuracy (Align Regions, Choose Caption)'] = torch.mean(
                    (pw_cost_r2w.argmin(dim=0) ==
                     torch.arange(batch_size).to('cuda')
                    ).to(torch.float32))
                other_info['Batch Accuracy (Align Regions, Choose Image)'] = torch.mean(
                    (pw_cost_r2w.argmin(dim=1) ==
                     torch.arange(batch_size).to('cuda')
                    ).to(torch.float32))

        self.log_dict(losses)
        self.log_dict(other_info)

        return other_info, losses


import torch
import torch.nn.functional as F

MAX_SAMPLE_SIZE = 30

def ranking(context_hidden, next_hidden, next_top_k_ids, next_top_k_probs, alpha):
    '''
        context_hidden: beam_width x context_len x embed_dim
        next_hidden: beam_width x 1 x embed_dim
        next_top_k_ids: beam_width x 1
    '''
    beam_width, context_len, embed_dim = context_hidden.size()
    norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
    norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
    cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1,2)).squeeze(-1)
    scores, _ = torch.max(cosine_matrix, dim = -1)
    next_top_k_probs = next_top_k_probs.view(-1)
    scores = next_top_k_probs / (scores ** alpha)
    selected_idx = scores.multinomial(1)
    selected_idx = selected_idx.unsqueeze(0)
    next_id = torch.gather(next_top_k_ids, dim = 0, index=selected_idx)
    return next_id


def EncDecContrastiveDecodingOneStep(model, input_ids, decoder_input_ids, p, alpha):
    '''
        model: the generation model, e.g., gpt2
        input_ids: 1 x seqlen
    '''
    prev_hidden_states, logits = model.compute_logits_and_hidden_states(input_ids, decoder_input_ids)
    _, seqlen, embed_dim = prev_hidden_states.size()

    logit_for_next_step = logits[:,-1,:]
    next_probs = F.softmax(logit_for_next_step, dim = -1)

    sorted_probs, sorted_indices = torch.sort(next_probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs <= p
    sorted_indices_to_remove[:, 0] = 1
    top_p_count = min(MAX_SAMPLE_SIZE, sorted_indices_to_remove.sum())

    top_p_ids = sorted_indices[:, :top_p_count]
    top_p_probs = sorted_probs[:, :top_p_count]

        
    top_p_probs = torch.gather(next_probs, dim = 1, index=top_p_ids)
    beam_width = top_p_count

    # compute new hidden 
    expanded_context = [decoder_input_ids for _ in range(beam_width)]
    expanded_context = torch.cat(expanded_context, dim = 0)
    expanded_input_ids = [input_ids for _ in range(beam_width)]
    expanded_input_ids = torch.cat(expanded_input_ids)
    top_p_ids = top_p_ids.view(beam_width, 1)
    next_input_ids = torch.cat([expanded_context, top_p_ids], dim = -1)
    new_hidden_states, next_logits = model.compute_logits_and_hidden_states(expanded_input_ids, next_input_ids)
    context_hidden = new_hidden_states[:,:seqlen,:]
    next_hidden = new_hidden_states[:,seqlen:,:]

    next_id = ranking(context_hidden, next_hidden, top_p_ids, top_p_probs, alpha)       

    next_input_ids = torch.cat([decoder_input_ids, next_id], dim = -1)
    return next_input_ids


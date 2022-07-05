import sys
# import ipdb
import os
import operator
from operator import itemgetter
import torch
from torch import nn
import random
import argparse
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

train_fct = CrossEntropyLoss()
val_fct = CrossEntropyLoss(reduction='none')
class SimCTGBART(nn.Module):
    def __init__(self, model_name):
        super(SimCTGBART, self).__init__()
        from transformers import AutoTokenizer, BartForConditionalGeneration
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.vocab_size = len(self.tokenizer)
        self.embed_dim = self.model.config.hidden_size
        self.pad_token_id = self.tokenizer.pad_token_id

    def compute_logits_and_hidden_states(self, input_ids):
        # used for advanced decoding
        # input_ids: 1 x seqlen
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        logits = outputs.logits
        return last_hidden_states, logits

    def forward(self, input_ids, labels, **kwargs):
        bsz, seqlen = labels.size()
        outputs = self.model(input_ids=input_ids, output_hidden_states=True, labels=labels)
        return outputs
        # logits = outputs.logits
        # assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])
        # last_hidden_states = outputs.decoder_hidden_states[-1]
        # assert last_hidden_states.size() == torch.Size([bsz, seqlen, self.embed_dim])
        # return last_hidden_states, logits

    def eval_loss(self, input_ids, labels):
        bsz, seqlen = labels.size()
        outputs = self.model(input_ids=input_ids, output_hidden_states=True, labels=labels)
        logits = outputs.logits
        assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])
        mle_loss = val_fct(logits.view(-1, self.vocab_size), labels.view(-1))
        assert mle_loss.size() == torch.Size([bsz * seqlen])
        mask_tmp = labels.masked_fill(~labels.eq(-100), 1.0)
        mask = mask_tmp.masked_fill(mask_tmp.eq(-100), 0.0)
        # sum 
        mle_loss_sum = torch.sum(mle_loss)
        token_num_sum = torch.sum(mask)
        return mle_loss_sum, token_num_sum

    def save_model(self, ckpt_save_path):
        import os
        if os.path.exists(ckpt_save_path):
            pass
        else: # recursively construct directory
            os.makedirs(ckpt_save_path, exist_ok=True)
        # save model
        self.model.save_pretrained(ckpt_save_path)
        # save tokenizer
        self.tokenizer.save_pretrained(ckpt_save_path)
        
    @torch.no_grad()
    # decoding functions
    # ------------------------------------------------------- #
    def fast_contrastive_search(self, input_ids, decoder_ids, beam_width, alpha, decoding_len, output_hidden_states=False):
        '''
           input_ids: prefix input; 1 x prefix_len
           decoding_len: how many tokens to generate
           beam_width: size of candidate pool during decoding
           alpha: regulates importance of model confidence and degeneration penalty
        '''
        self.model.eval()
        from utlis import EncDecContrastiveDecodingOneStepFast
        # sanity check
        assert alpha >= 0. and alpha <= 1.0
        
        batch_size, seqlen = input_ids.size()
        generated = []
        past_key_values = None
        last_hidden_states = None
        logits = None
        input_embeds = None
        for step in range(decoding_len):
            decoder_ids, past_key_values, last_hidden_states, logits, input_embeds = EncDecContrastiveDecodingOneStepFast(
                self.model,
                input_ids,
                decoder_ids,
                beam_width,
                alpha,
                past_key_values,
                last_hidden_states,
                self.tokenizer,
                logits,
                first_step=step == 0,
                input_embeds=input_embeds,
            )
            token = decoder_ids.squeeze(dim=-1).item()
            generated.append(token)
        return (generated, last_hidden_states) if output_hidden_states else generated

    def greedy_search(self, input_ids, decoding_len):
        output = self.model.generate(
                            input_ids=input_ids, 
                            max_length=decoding_len)
        return output[0]

    def beam_search(self, input_ids, beam_width, decoding_len):
        output = self.model.generate(
                            input_ids=input_ids, 
                            max_length=decoding_len, 
                            num_beams=beam_width)
        return output[0]
    
    def nucleus_sampling(self, input_ids, nucleus_p, decoding_len):
        output = self.model.generate(
                            input_ids, 
                            do_sample=True, 
                            max_length=decoding_len, 
                            top_p=nucleus_p,
                            top_k=0)

        return output[0]

    def slow_sampling_contrastive_search(self, input_ids, p, alpha, decoding_len):
        '''
           input_ids: prefix input; 1 x prefix_len
           decoding_len: how many tokens to generate
           p: probability filtering
           alpha: regulates importance of model confidence and degeneration penalty
        '''

        from utils_prob import ContrastiveDecodingOneStep
        for step in range(decoding_len):
            input_ids = ContrastiveDecodingOneStep(self, input_ids, p, alpha)
        return input_ids[0]
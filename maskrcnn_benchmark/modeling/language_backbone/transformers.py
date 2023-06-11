from copy import deepcopy
import numpy as np
import torch
from torch import nn
from transformers import BertModel, BertTokenizer, BertConfig

class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()
        self.config = config
        self.bert_config = BertConfig.from_pretrained('bert-base-uncased')
        self.update_bert_config()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if self.config.MODEL.LANGUAGE_BACKBONE.USE_PRETRAINED:
            self.bert_model = BertModel.from_pretrained(
                'bert-base-uncased', config=self.bert_config)    
        else:
            self.bert_model = BertModel(self.bert_config)
        self.freeze()
        self.out_channels = self.bert_config.hidden_size
        head_config = self.config.MODEL.MMSS_HEAD.TRANSFORMER
        self.mlm = head_config.MASKED_LANGUAGE_MODELING
        self.mlm_prob = head_config.MASKED_LANGUAGE_MODELING_PROB
        self.mlm_prob_mask = head_config.MASKED_LANGUAGE_MODELING_PROB_MASK
        self.mlm_prob_noise = head_config.MASKED_LANGUAGE_MODELING_PROB_NOISE
        self.mlm_during_validation = head_config.MASKED_LANGUAGE_MODELING_VALIDATION
        self.embeddings = self.bert_model.embeddings.word_embeddings.weight
        self.word_and_noun_phrase_tokens = self.config.DATASETS.DATASET_ARGS.WORD_N_NOUN_PHRASE
        cls_token_id = self.tokenizer.convert_tokens_to_ids('[CLS]')
        sep_token_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.start_end_token_embeddings = self.embeddings[torch.tensor([cls_token_id, sep_token_id]).cuda()]


    def forward(self, text_list):
        tokenized_batch = self.tokenizer.batch_encode_plus(text_list, 
            add_special_tokens=True, 
            pad_to_max_length=True,
            return_special_tokens_mask=True,
        )
        if self.mlm:
            tokenized_batch['target_ids'] = deepcopy(tokenized_batch['input_ids'])
            tokenized_batch['mlm_mask'] = []
            for i, item in enumerate(tokenized_batch['input_ids']):
                mlm_mask = []
                for j in range(len(item)):
                    if (tokenized_batch['special_tokens_mask'][i][j] or
                        not tokenized_batch['attention_mask'][i][j] or
                        not (self.training or self.mlm_during_validation)):
                        mlm_mask.append(0)
                        continue
                    prob = np.random.rand()
                    if prob < self.mlm_prob:
                        mlm_mask.append(1)
                        prob /= self.mlm_prob
                        if prob < self.mlm_prob_mask:
                            item[j] = self.tokenizer.convert_tokens_to_ids(
                                self.tokenizer.mask_token)
                            tokenized_batch['special_tokens_mask'][i][j] = 1
                        elif prob < self.mlm_prob_mask + self.mlm_prob_noise:
                            item[j] = np.random.randint(len(self.tokenizer))
                    else:
                        mlm_mask.append(0)
                tokenized_batch['mlm_mask'].append(mlm_mask)

        tokenized_batch = {k: torch.tensor(v).cuda() for k, v in tokenized_batch.items()}
        bert_output = self.bert_model(
            input_ids=tokenized_batch['input_ids'],
            attention_mask=tokenized_batch['attention_mask'],
        )
        tokenized_batch['encoded_tokens'] = bert_output[0]
        tokenized_batch['input_embeddings'] = self.embeddings[tokenized_batch['input_ids']]

        if self.word_and_noun_phrase_tokens:
            text_list_len = []
            linear_text_list = []
            for text in text_list:
                words = text.split()
                text_list_len.append(len(words))
                linear_text_list.extend(words)

            linear_tokenized_batch = self.tokenizer.batch_encode_plus(linear_text_list, 
                add_special_tokens=True, 
                pad_to_max_length=True,
                return_special_tokens_mask=True,
            )

            linear_special_tokens_mask = (1 - torch.tensor(linear_tokenized_batch['special_tokens_mask']).cuda()).float()
            linear_input_embeddings = self.embeddings[torch.tensor(linear_tokenized_batch['input_ids']).cuda()]
            linear_input_embeddings = (linear_input_embeddings * linear_special_tokens_mask[:,:,None]).sum(1) / linear_special_tokens_mask.sum(1)[:,None]

            grounded_input_embeddings = torch.zeros(len(text_list), max(text_list_len)+2, linear_input_embeddings.shape[-1]).cuda()
            grounded_special_tokens_mask = torch.zeros(len(text_list), max(text_list_len)+2).cuda()
            grounded_attention_mask = torch.zeros(len(text_list), max(text_list_len)+2).cuda()

            start_idx = 0
            for batch_idx, text_len in enumerate(text_list_len):
                grounded_input_embeddings[batch_idx, 0, :] = self.start_end_token_embeddings[0,:] 
                grounded_input_embeddings[batch_idx, 1:text_len+1, :] = linear_input_embeddings[start_idx:start_idx+text_len, :]
                grounded_input_embeddings[batch_idx, text_len + 1, :] = self.start_end_token_embeddings[1,:] 

                grounded_special_tokens_mask[batch_idx, 0] = 1
                grounded_special_tokens_mask[batch_idx, 1:text_len+1] = 0
                grounded_special_tokens_mask[batch_idx, text_len + 1] = 1
                 
                grounded_attention_mask[batch_idx, 0] = 0
                grounded_attention_mask[batch_idx, 1:text_len+1] = 1
                grounded_attention_mask[batch_idx, text_len + 1] = 0

                start_idx = start_idx + text_len

            tokenized_batch['grounded_input_embeddings'] = grounded_input_embeddings
            tokenized_batch['grounded_special_tokens_mask'] = grounded_special_tokens_mask
            tokenized_batch['grounded_attention_mask'] = grounded_attention_mask

        else:
            tokenized_batch['grounded_input_embeddings'] = tokenized_batch['input_embeddings']
            tokenized_batch['grounded_special_tokens_mask'] = tokenized_batch['special_tokens_mask']
            tokenized_batch['grounded_attention_mask'] = tokenized_batch['attention_mask']
            
        return tokenized_batch


    def freeze(self):
        for p in self.bert_model.pooler.parameters():
            p.requires_grad = False
        if self.config.MODEL.LANGUAGE_BACKBONE.FREEZE:
            for p in self.parameters():
                p.requires_grad = False


    def update_bert_config(self):
        pass
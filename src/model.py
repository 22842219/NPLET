from audioop import cross
from matplotlib.pyplot import margins
from sklearn.datasets import make_sparse_coded_signal
from sklearn.metrics import hinge_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck

from luke.model import LukeEntityAwareAttentionModel

import pysdd
from pysdd.sdd import Vtree, SddManager, WmcManager
from pysdd.iterator import SddIterator
from array import array
from pathlib import Path
here = Path(__file__).parent

class EntityTyping(LukeEntityAwareAttentionModel):
    def __init__(self, args, num_labels, is_sdd):
        super(EntityTyping, self).__init__(args.model_config)

        self.num_labels = num_labels
        self.is_sdd = is_sdd
        self.dropout = nn.Dropout(args.model_config.hidden_dropout_prob)
        self.typing = nn.Linear(args.model_config.hidden_size, num_labels)
        self.apply(self.init_weights)

    def forward(
        self,
        word_ids,
        word_segment_ids,
        word_attention_mask,
        entity_ids,
        entity_position_ids,
        entity_segment_ids,
        entity_attention_mask,
        labels=None,
    ):
        encoder_outputs = super(EntityTyping, self).forward(
            word_ids,
            word_segment_ids,
            word_attention_mask,
            entity_ids,
            entity_position_ids,
            entity_segment_ids,
            entity_attention_mask,
        )

        emb_vector = encoder_outputs[1][:, 0, :]
        emb_vector = self.dropout(emb_vector)  # R[Batch, Emb]=[Batch, 1024]
        logits = self.typing(emb_vector)   

        if self.is_sdd and labels is not None:
            cmpe = CircuitMPE(bytes(here/"sdd_input"/"ontonotes_modified"/"et.vtree"), bytes(here/"sdd_input"/"ontonotes_modified"/"et.sdd"))
            norm_y_hat = torch.sigmoid(logits)  
            updated_query = cmpe.compute_wmc(norm_y_hat) 
            cross_entropy= torch.nn.BCELoss()
            loss = cross_entropy(updated_query, labels.to(torch.float32))
            return (loss, )

        if labels is None:
            return logits
        
        return (F.binary_cross_entropy_with_logits( logits.view(-1), labels.view(-1).type_as(logits)),)

class CircuitMPE:	
    def __init__ (self, vtree_file, sdd_file, test = False):

        self.vtree = Vtree.from_file(vtree_file)
        self.sdd_mgr = SddManager.from_vtree(self.vtree)
        self.et_logical_formula = self.sdd_mgr.read_sdd_file(sdd_file)
        self.length = self.sdd_mgr.var_count()       	
        # self.wmc_mgr = self.et_logical_formula.wmc(log_mode = False)
        self.lits = [None] + [self.sdd_mgr.literal(i) for i in range(1, self.sdd_mgr.var_count() + 1)]        


    def compute_wmc(self, norm_y_hat):
        updated_query = torch.empty(norm_y_hat.size()).cuda()	
        for index, every in enumerate(norm_y_hat):		
            lenght = every.size()[0]
            wmc_mgr = self.et_logical_formula.wmc(log_mode=False) 
            for i in range(0, lenght):
                weights = torch.cat((every, every))
                for j, p in enumerate(every):	
                    weights[j] = 1.0 - every[len(every)-1-j]
                weights = array('d',weights)		
                wmc_mgr.set_literal_weights_from_array(weights)
                wmc = wmc_mgr.propagate()
       
                wmc_mgr.set_literal_weight(self.lits[i+1], 1)
                wmc_mgr.set_literal_weight(-self.lits[i+1], 0)
                every_conditioned_wmc = wmc_mgr.propagate()	

                query_atom = every_conditioned_wmc*every[i]
                updated_ = torch.div(query_atom, wmc)
                updated_query[index][i] = updated_

        return    updated_query
 

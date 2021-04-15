"""
Taken from https://github.com/Andras7/word2vec-pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

"""
    u_embedding: Embedding for center word.
    v_embedding: Embedding for neighbor words.
"""


def is_in(batch_ids, target_ids):
    o = np.in1d(batch_ids.detach().cpu().numpy(), target_ids)
    return torch.from_numpy(o).to(batch_ids.device)


class SkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension, frozen_embeddings=None, frozen_embedding_ids=None):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.frozen_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        if frozen_embeddings is not None:
            self.frozen_embedding_ids = frozen_embedding_ids.cpu().numpy()
            self.frozen_embeddings.weight.data[self.frozen_embedding_ids, :] = frozen_embeddings.float().cpu()
        else:
            self.frozen_embedding_ids = np.array([])
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, pos_u, pos_v, neg_v):
        learned_emb_u = self.u_embeddings(pos_u)
        frozen_emb_u = self.frozen_embeddings(pos_u).detach()
        mask = is_in(pos_u, self.frozen_embedding_ids).unsqueeze(1).expand(-1, learned_emb_u.size(1))
        emb_u = torch.where(mask, frozen_emb_u, learned_emb_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)

    def save_embedding(self, id2word, file_name):
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        embedding[self.frozen_embedding_ids, :] = self.frozen_embeddings.weight.data[self.frozen_embedding_ids, :].detach().cpu().numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))

# Time-stamp: <11/09/2019>
# --------------------------------------------------------------------
# File Name          : matcher.py
# Original Author    : jiessie.cao@gmail.com
# Description        : Mathers for caculating the similarities of two encoding, in both 1:1 case or 1:N selecting cases
# --------------------------------------------------------------------

import torch
import torch.nn as nn

class DotProductMatcher(nn.Module):
    """
    Dot Product Match for the similarity of encoding
    """
    def __init__(self, encoding_dim):
        super(DotProductMatcher, self).__init__()
        self.inputDim = encoding_dim
        self.outputDim = 1

    def forward(self, ori_plan_encoding, candidate_plan_encodings):
        # batch_size * 1 * plan_encoding_dimension
        expanded_ori_plan_encoding = ori_plan_encoding.unsqueeze(1)
        if candidate_plan_encodings.dim() == 2:
            expanded_candidate_plan_encodings = candidate_plan_encodings.unsqueeze(1)
            expanded_plan_encoding = expanded_ori_plan_encoding
        else:
            # batchsize * num * dim
            expanded_candidate_plan_encodings = candidate_plan_encodings
            expanded_plan_encoding = expanded_ori_plan_encoding.repeat(1,candidate_plan_encodings.size()[1],1)
        # Baseline method1, dot product simialrity
        # batch_size *1 * 10 => batch_size * 10
        similarities = torch.bmm(expanded_plan_encoding.view(-1,1, self.inputDim), expanded_candidate_plan_encodings.view(-1, 1, self.inputDim).transpose(1,2)).view(-1)
        return similarities


class BilinearMatcher(nn.Module):
    """
    Bilinear layer than dot product for matching, the bilinear lay may learn something more than similarity (e.g. optimization)
    """
    def __init__(self, encoding_dim):
        super(BilinearMatcher, self).__init__()
        self.inputDim = encoding_dim
        self.outputDim = 1
        self.bilinear = nn.Linear(self.inputDim ,self.inputDim)

    def forward(self, ori_plan_encoding, candidate_plan_encodings):
        # batch_size * 1 * plan_encoding_dimension
        expanded_ori_plan_encoding = ori_plan_encoding.unsqueeze(1)
        if candidate_plan_encodings.dim() == 2:
            expanded_candidate_plan_encodings = candidate_plan_encodings.unsqueeze(1)
            expanded_plan_encoding = expanded_ori_plan_encoding
        else:
            # batchsize * num * dim
            expanded_candidate_plan_encodings = candidate_plan_encodings
            expanded_plan_encoding = expanded_ori_plan_encoding.repeat(1,candidate_plan_encodings.size()[1],1)

        # (batch_size * N * plan_encoding_dim) x (plan_encoding * plan_encoding)
        bilinear_ori_plan_encoding = self.bilinear(expanded_plan_encoding)
        # bilinear then dot product
        # batch_size *N * inputDim => (batch_size * N) * inputDim
        similarities = torch.bmm(bilinear_ori_plan_encoding.view(-1,1,self.inputDim), expanded_candidate_plan_encodings.view(-1,1, self.inputDim).transpose(1,2)).view(-1)
        return similarities


class ConcatenateMatcher(nn.Module):
    """
    Bilinear layer than dot product for matching, the bilinear lay may learn something more than similarity (e.g. optimization)
    """
    def __init__(self, encoding_dim):
        super(ConcatenateMatcher, self).__init__()
        self.inputDim = encoding_dim
        self.outputDim = 4 * self.inputDim

    def forward(self, ori_plan_encoding, candidate_plan_encodings):
        # batch_size * 1 * plan_encoding_dimension
        expanded_ori_plan_encoding = ori_plan_encoding.unsqueeze(1)
        if candidate_plan_encodings.dim() == 2:
            expanded_candidate_plan_encodings = candidate_plan_encodings.unsqueeze(1)
            expanded_plan_encoding = expanded_ori_plan_encoding
        else:
            # batchsize * num * dim
            expanded_candidate_plan_encodings = candidate_plan_encodings
            expanded_plan_encoding = expanded_ori_plan_encoding.repeat(1,candidate_plan_encodings.size()[1],1)

        # A; B
        concatenated_encoding = torch.cat((expanded_plan_encoding, expanded_candidate_plan_encodings), 2)

        # A-B;A*B;A;B;
        concatenated_encoding = torch.cat((expanded_plan_encoding - expanded_candidate_plan_encodings, expanded_plan_encoding * expanded_candidate_plan_encodings , concatenated_encoding), 2)
        if candidate_plan_encodings.dim() == 2:
            concatenated_encoding = concatenated_encoding.squeeze(1)

        return concatenated_encoding

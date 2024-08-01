import torch
import numpy as np

from fundus_prepro.algo.autobalance import autobalance
from fundus_prepro.algo.CLAHE_LAB import clahe_lab
from fundus_prepro.algo.CLAHE_MGG import clahe_max_green_gsc
from fundus_prepro.algo.CLAHE_RGB import clahe_rgb
from fundus_prepro.algo.graham_METH1 import graham_meth1
from fundus_prepro.algo.graham_METH2 import graham_meth2
from fundus_prepro.algo.sarki import sarki
from fundus_prepro.algo.seoud import seoud
from nntools.dataset import nntools_wrapper




def spatial_batch_normalization(batch: torch.Tensor):
    dims = batch.ndim - 1
    min_value = torch.amin(batch, (dims - 1, dims), keepdim=True)
    max_value = torch.amax(batch, (dims - 1, dims), keepdim=True)
    return (batch - min_value) / (max_value - min_value)


def get_preprocessing(name:str):
    match name: 
        case "autobalance":
            return nntools_wrapper(autobalance)
        case "clahe_lab":
            return nntools_wrapper(clahe_lab)
        case "clahe_rgb":
            return nntools_wrapper(clahe_rgb)
        case "clahe_max_green_gsc":
            return nntools_wrapper(clahe_max_green_gsc)
        case "graham_meth1":
            return nntools_wrapper(graham_meth1)
        case "graham_meth2":
            return nntools_wrapper(graham_meth2)
        case "sarki":
            return nntools_wrapper(sarki)
        case "seoud":
            return nntools_wrapper(seoud)
        



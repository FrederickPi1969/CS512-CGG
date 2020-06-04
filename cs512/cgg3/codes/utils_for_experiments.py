import networkx
import numpy as np
import torch

def load_all_params(operation_name):
    """
    load all pretrained model
    """
    vae = torch.load("vae.model")

    if operation_name == "density":
        w = torch.load("w_density.pt")
        a_w1 = torch.load("a_w1_density.pt")
        a_w2 = torch.load("a_w2_density.pt")
        a_b1 = torch.load("a_b1_density.pt")
        a_b2 = torch.load("a_b2_density.pt")


    else:
        raise NameError("operation model not trained")

    return vae, w, a_w1,a_w2, a_b1, a_b2

# def generate_one_set(alpha,):


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
import os
from math import pi
from math import cos


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED']=str(seed)


def get_encodings(model, dl):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    result = []
    with torch.no_grad():
        for x in dl:
            encodings,mu,var = model.encoder(x.to(device))
            result.append(encodings)
    return torch.cat(result, dim=0)


def get_decodings(model, dl):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    result = []
    with torch.no_grad():
        for x in dl:
            decodings, mu, var = model(x.to(device))
            result.append(decodings)
    return torch.cat(result, dim=0)
 

    
def KL_loss(mu, logvar):

    KLD = -0.5 * torch.mean(1 + logvar - mu**2 -  logvar.exp())
    return  KLD

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps*std + mu
 


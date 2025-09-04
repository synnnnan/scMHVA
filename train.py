import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
import os
import sys
import shutil
from util import KL_loss,get_encodings,setup_seed
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np



def pretrain_model(model,train_dl:torch.utils.data.dataloader.DataLoader,lr,epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=0.000002) 
    criterion = nn.MSELoss().to(device)
    model=model.to(device)
    model.train
    for epoch in tqdm(range(1, epochs + 1)):
        train_loss=0
        for x in train_dl:
            x = x.to(device)
            optimizer.zero_grad()

            x_prime, mu, var= model(x)
            loss = criterion(x_prime, x)
            loss.backward()
            optimizer.step()
            batch_size = x.shape[0]
        train_loss += batch_size*(loss.item())
    torch.save(model.state_dict(),"pretrained_model.pth")





def train_model(model, train_dl: torch.utils.data.dataloader.DataLoader ,  lr,  epochs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=0.000002) 
    criterion = nn.MSELoss().to(device)
    history = defaultdict(list)
    
    pretrained_dict = torch.load("pretrained_model.pth")  
    model.load_state_dict(pretrained_dict)

    
    embedding=[]
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        nsamples_train = 0
        train_loss=0
     
           
        for x in train_dl:
            optimizer.zero_grad()
            x = x.to(device)
            x_prime, mu, var= model(x)
            loss = criterion(x_prime, x) + 0.0001*(KL_loss(mu,var))

            # Backward pass
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
            loss.backward()
            optimizer.step()

            # log losses
            batch_size = x.shape[0]
            nsamples_train += batch_size
            train_loss += batch_size*(loss.item())
       
        train_loss = train_loss / nsamples_train
        history['train'].append(train_loss)

        if epoch == epochs:
            simulated_data_ls = get_encodings(model,train_dl)
            temp= pd.DataFrame(simulated_data_ls.cpu().numpy())
            embedding.append(temp)
         
    return model,history,embedding
   



      
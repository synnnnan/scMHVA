import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion import Fusion


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class LinBnDrop(nn.Sequential):
    def __init__(self, n_in, n_out, bn=True, p=0., act=None, lin_first=True):
        layers = [nn.BatchNorm1d(n_out if lin_first else n_in)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        if act is not None: lin.append(act)
        layers = lin+layers if lin_first else layers+lin
        super().__init__(*layers)


class MultiHead_SelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads,dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads 
        assert input_dim % num_heads == 0, "Input dimension must be divisible by the number of heads."

        # Linear layers for the query, key, and value projections for each head
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

        self.output_linear = nn.Linear(input_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim,eps=1e-6)

    def forward(self,x):
        batch_size=x.shape[0]
        seq_len=x.shape[1]
        #Split the output vector into multiple heads after matrix multiplication
        query=self.query(x).view(batch_size,seq_len,self.num_heads,self.head_dim)
        key=self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        value=self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        query=query.transpose(1,2)
        key=key.transpose(1,2)
        value=value.transpose(1,2)
       
        attention_scores=torch.matmul(query,key.transpose(2,3))/torch.sqrt(torch.tensor(self.head_dim,dtype=torch.float))
        attention_weights=torch.softmax(attention_scores,dim=-1)
    
        attention=torch.matmul(attention_weights,value)
        attention=attention.transpose(1,2).contiguous().view(batch_size,seq_len,self.num_heads*self.head_dim)
        output = self.dropout(self.output_linear(attention))
        output = self.layer_norm(x + output)
        return output




class Encoder(nn.Module):
    def __init__(self, num_features: list, num_hidden_features: list, z_dim: int=128):
        super().__init__()
        
        self.features=num_features
        self.num_hidden_features=num_hidden_features
        self.encoder_eachmodal= nn.ModuleList([LinBnDrop(num_features[i], num_hidden_features[i], p=0.2, act=nn.ReLU())
                                 for i in range(len(num_hidden_features))]).to(device) 
        self.fusion = Fusion(
            n_views=len(num_hidden_features), 
            input_sizes=[(num_hidden_features[i],) for i in range(len(num_hidden_features))]
        ).to(device)
        self.encoder = LinBnDrop(sum(self.fusion.output_size), z_dim, act=nn.ReLU()).to(device)
        self.weights=[]
        for i in range(len(num_features)):
            self.weights.append(nn.Parameter(torch.rand(1,num_features[i]) * 0.001, requires_grad=True).to(device))

        self.fc_mu =nn.Sequential( LinBnDrop(z_dim,z_dim, p=0.1),
                                 ).to(device)
        self.fc_var =nn.Sequential( LinBnDrop(z_dim,z_dim, p=0.1),
                                     ).to(device)

        self.attention = MultiHead_SelfAttention(input_dim=sum(self.fusion.output_size)//5,num_heads=4,dropout=0.1).to(device)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    


    def forward(self, x):
        X = []
        startfeature=0
        num_hidden_features=self.num_hidden_features
        for i, eachmodal in enumerate(self.encoder_eachmodal):
            tmp=eachmodal(x[:,startfeature:(startfeature+self.features[i])])
            startfeature=startfeature+self.features[i]
            X.append(tmp)
        x = self.fusion(X)
        x=x.view(x.shape[0],5,sum(self.fusion.output_size)//5)
        x = self.attention(x)
        x=x.view(x.shape[0],sum(self.fusion.output_size))
        x = self.encoder(x)
        mu = self.fc_mu(x)
        var = self.fc_var(x)
        x = self.reparameterize(mu, var)
        return x,mu,var
    


    
class Decoder(nn.Module):
    def __init__(self, num_features: list, z_dim: int = 100):
        super().__init__()
        self.features=num_features
        self.z_dim=z_dim
        self.attention = MultiHead_SelfAttention(input_dim=z_dim//5,num_heads=4,dropout=0.1).to(device)
        self.decoder_eachmodal= nn.ModuleList([ LinBnDrop(z_dim, num_features[i], act=nn.ReLU()) for i in range(len(num_features))]).to(device) 

    def forward(self, x):
        X = []
        z_dim=self.z_dim
        for i, deachmodal in enumerate(self.decoder_eachmodal):
            x=x.view(x.shape[0],5,z_dim//5)
            tmp = self.attention(x)
            tmp=tmp.view(x.shape[0],z_dim)
            tmp=deachmodal(tmp)
            X.append(tmp)
        x = torch.cat(X, 1)
        return x

    
class mhVAE(nn.Module):
    def __init__(self, num_features: list, num_hidden_features: list, z_dim: int = 100):
        super().__init__()
        self.encoder = Encoder(num_features, num_hidden_features, z_dim)
        self.decoder = Decoder(num_features, z_dim)

    
    def forward(self, x):
        x_encoder,mu,var= self.encoder(x)
        x= self.decoder(x_encoder)
        return x,mu,var
          



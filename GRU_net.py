import os
import torch as T
import torch.nn as nn
import torch.optim as optim

class GRU_model(nn.Module):
    
    def __init__(self, input_size=11, hidden_size=128, num_stacked_layers=3, alpha=0.001):
        super(GRU_model, self).__init__()
        
        #----- GRU -----
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_stacked_layers, batch_first=True)
        
        #----- Fully connected part -----
        self.fc1   = nn.Linear(hidden_size, 2048)
        self.act1  = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        
        self.fc2   = nn.Linear(2048, 1024)
        self.act2  = nn.ReLU()
        self.drop2 = nn.Dropout(0.3)
        
        self.fc3   = nn.Linear(1024, 256)
        self.act3  = nn.ReLU()
        self.drop3 = nn.Dropout(0.3)
        
        self.fc4   = nn.Linear(256, 64)
        self.act4  = nn.ReLU()
        self.drop4 = nn.Dropout(0.3)
        
        self.fc5   = nn.Linear(64, 5)  # Dernière couche sans activation
        self.act5  = nn.ReLU()  # peut etre sans

        #----- Optimiseur & Scheduler -----
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')       
        self.to(self.device)

    def forward(self, x):
        batch_size = x.size(0)
        
        h0 = T.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        
        out, _ = self.gru(x, h0)  # Pas de cellule c dans GRU
        
        x = self.act1(self.fc1(out[:, -1, :]))
        x = self.drop1(x)
        
        x = self.act2(self.fc2(x))
        x = self.drop2(x)
        
        x = self.act3(self.fc3(x))
        x = self.drop3(x) 

        x = self.act4(self.fc4(x))
        x = self.drop4(x) 
        
        x = self.fc5(x)
        
        return x
    
    def save_checkpoint(self, iteration, ckpt_file_name):
        if not os.path.isdir(ckpt_file_name):
            os.makedirs(ckpt_file_name)
        T.save(self.state_dict(), ckpt_file_name + '/policy_network_GRU' + str(iteration) + '.pt')

    def load_checkpoint(self, ckpt_file_name):
        self.load_state_dict(T.load(ckpt_file_name, map_location=T.device('cpu')))

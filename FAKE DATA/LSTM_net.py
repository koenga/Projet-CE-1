#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import torch as T
import torch.nn as nn
import torch.optim as optim

class LSTM_model(nn.Module):
    
    def __init__(self, input_size=11, hidden_size=64, num_stacked_layers=2, alpha=0.001):
    
        super(LSTM_model, self).__init__()
        
        #----- LSTM -----
        
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        
        #----- Fully connected part -----
        
        self.fc1   = nn.Linear(hidden_size, 2048)
        #T.nn.init.xavier_uniform_(self.fc1.weight)
        self.act1  = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)
        
        self.fc2   = nn.Linear(2048, 1024) 
        #T.nn.init.xavier_uniform_(self.fc2.weight)
        self.act2  = nn.ReLU()
        self.drop2 = nn.Dropout(0.5)
        
        self.fc3   = nn.Linear(1024, 256) 
        #T.nn.init.xavier_uniform_(self.fc3.weight)
        self.act3  = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)
        
        self.fc4   = nn.Linear(256, 64) 
        #T.nn.init.xavier_uniform_(self.fc4.weight)
        self.act4  = nn.ReLU()
        self.drop4 = nn.Dropout(0.5) 
        
        self.fc5   = nn.Linear(64,5)
        #T.nn.init.xavier_uniform_(self.fc5.weight)
        self.act5  = nn.ReLU()
        
        #----- Remaining parameters -----

        self.optimizer = optim.Adam(self.parameters(), lr = alpha)
        self.scheduler = T.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')       
        self.to(self.device)
        
        #----- Put yourself on the appropriate device -----

        self.to(self.device)

    def forward(self, x):
        
        #----- LSTM -----
        
        batch_size = x.size(0)
        
        h0 = T.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        c0 = T.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        #----- Fully connected part -----
        
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
            
        T.save(self.state_dict(), ckpt_file_name + '/policy_network_' + str(iteration) + '.pt')

        
    def load_checkpoint(self, ckpt_file_name):
        
        #print('... loading checkpoint ...')
        
        self.load_state_dict(T.load(ckpt_file_name, map_location=T.device('cpu')))   

        
        
    


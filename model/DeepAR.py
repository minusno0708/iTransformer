import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.hidden_size = 32

        self.lstm = nn.LSTM(
            input_size=configs.enc_in,
            hidden_size=self.hidden_size,
            num_layers=2,
            dropout=configs.dropout,
            batch_first=True
        )
        self.output_mu = nn.Linear(self.hidden_size, configs.c_out)
        self.output_sigma = nn.Linear(self.hidden_size, configs.c_out)
       
    def forecast(self, x):
        lstm_out, _ = self.lstm(x)
        mu = self.output_mu(lstm_out)
        #sigma = F.softplus(self.output_sigma(lstm_out))
        sigma = torch.exp(self.output_sigma(lstm_out))
        return mu, sigma

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        mu_out, sigma_out = self.forecast(x_enc)
        mu_out = mu_out[:, -self.pred_len:, :]
        sigma_out = sigma_out[:, -self.pred_len:, :]

        return mu_out, sigma_out
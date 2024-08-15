import torch
import torch.nn as nn

import sys
from src.exception import CustomException
from src.logger import logging

# Define the neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        try:
            super(NeuralNet, self).__init__()
            self.l1 = nn.Linear(input_size, hidden_size)
            self.l2 = nn.Linear(hidden_size, hidden_size)
            self.l3 = nn.Linear(hidden_size, num_classes)
            self.relu = nn.ReLU()
            logging.info('Init of neural net function')
        except Exception as e:
            raise CustomException(e,sys)

    def forward(self, x):
        try:
            out = self.l1(x)
            out = self.relu(out)
            out = self.l2(out)
            out = self.relu(out)
            out = self.l3(out)
            logging.info('Return the output of the neural net function')
            return out
        except Exception as e:
            raise CustomException(e,sys)
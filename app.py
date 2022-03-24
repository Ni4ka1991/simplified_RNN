#!/usr/bin/env python3
from rnn import *
from data import *
from helper_func import *
import torch, torch.nn as nn


text = loadTextFromFile( "data/Bacteria in Daily Life.txt" )
#print( text[:1000] )
text = encodeText( text )
#print( text[:10] )


any_numb_positive = 1.4

x_size = len( alphabet )
#h_size = x_size * any_numb_positive
h_size = x_size

model = RNN( x_size, x_size, h_size )
criterion = nn.MSELoss()
optimizer = torch.optim.SGD( model.parameters(), lr = 0.005 )

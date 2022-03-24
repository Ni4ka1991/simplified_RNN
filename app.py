#!/usr/bin/env python3
from rnn import *
from data import *
from helper_func import *
import torch, torch.nn as nn


text = loadTextFromFile( "data/Bacteria in Daily Life.txt" )
plain_text = loadTextFromFile( "data/Bacteria in Daily Life.txt" )
print( text[:1000] )
text = encodeText( text )
#print( text[:10] )


any_numb_positive = 4 # int! when the type is float appear error

x_size = len( alphabet )
h_size = x_size * any_numb_positive
#h_size = x_size

model = RNN( x_size, x_size, h_size )
criterion = nn.MSELoss()
optimizer = torch.optim.SGD( model.parameters(), lr = 0.005 )

print()
print( f"first letter of the text >>> {plain_text[:1]}" )
print( f"first letter of the text in vector forman >>> {text[:1]}" )
X = torch.Tensor( text[:1] )
print( f"first letter of the text in Torch tensor format >>>\n{X}" )
H = model.init_hidden() # initialisation a hidden neuron with zerros
print( f"X.shape alphabet len >>>   {X.shape} >>> 27"   )
print( f"H.shape alphabet*4 len >>> {H.shape} >>> 27*4 = 108" )

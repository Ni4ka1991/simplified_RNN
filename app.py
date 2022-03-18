#!/usr/bin/env python3

from data import *
from helper_func import *
import torch, torch.nn as nn


text = loadTextFromFile( "data/Bacteria in Daily Life.txt" )
print( text[:1000] )
text = encodeText( text )
print( text[:10] )

x_size = len( alphabet )
h_size = x_size

class RNN( nn.Module ):
    
    def __init__( self, input_size, output_size, hidden_size ):
        super( RNN, self ).__init__()


        self.hidden_size = hidden_size


        self.hidden = nn.Linear( input_size + hidden_size, hidden_size )
        self.output = nn.Linear( input_size + hidden_size, output_size )
        self.softmax = nn.LogSoftmax( dim = 1 )


import torch, torch.nn as nn

class RNN( nn.Module ):
    
    def __init__( self, input_size, output_size, hidden_size ):
        super( RNN, self ).__init__()


        self.hidden_size = hidden_size


        self.hidden = nn.Linear( input_size + hidden_size, hidden_size )
        self.output = nn.Linear( input_size + hidden_size, output_size )
        self.softmax = nn.LogSoftmax( dim = 1 )

    def forward( self, X, H ):
        x = torch.cat(( X, H ), 1 )

        h = self.hidden( x )
        y = self.output( x )
        y = self.softmax( y )

        return y, h

    def init_hidden( self ):
        return torch.zeros( 1, self.hidden_size )

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

Y, H = model( X, H )
print( "Data after model >>>" )
print( f"Y.shape alphabet len >>>   {Y.shape} >>> 27"   )
print( f"H.shape alphabet*4 len >>> {H.shape} >>> 27*4 = 108" )
print( f"Y torch tensor -> X, H after model(X, H) >>>\n{Y}" )
print()
print( f"H torch tensor after model(X, H) >>>\n{H}" )

def train( X ):
    h = model.init_hidden()
    model.zero_grad()
    print( f"len(X) >>> {len( X )}" )
    for i in range( len(X) - 1 ):
        x = torch.Tensor( [X[i]] )
        y = torch.Tensor( [X[i+1]] )
        if i == 0 or oneHotVectorToCharacter(x) == " ":
            h = model.init_hidden()
    
        optimizer.zero_grad()
        h = h.detach() #([[-3.2002, -3.3569, -3.2780, -3.3148, -3.1958, -3.2348, -3.1634, -3.2344]], grad_fn=<LogSoftmaxBackward0>)   :  detach >>> grad_fn=<LogSoftmaxBackward0>
        yp, h = model( x, h )
#        print( f"yp = {yp}" )
        loss = criterion( yp, y )
        loss.backward()
        optimizer.step()
    return loss.item()

epochs = 10

for i in range( epochs ):
    loss_value = train( text )
    if  i % 1 == 0:
        print( f"epoch {i} Loss {loss_value}" )


#model.load_state_dict( torch.load( "models/rnn_word_prediction" ))

textTest = encodeText("a")
model.eval()  #trash clearing
future = 15
h = model.init_hidden()
x = torch.Tensor( [textTest[0]])
print( oneHotVectorToCharacter( x ), " -> " )

for i in range( future ):
    yp, x = model( x , h )
    x = yp
    c = oneHotVectorToCharacter(yp) 
    print( c )
    if c == " ":
        break
    textTest = encodeText( c )
    x = torch.Tensor([textTest[0]])






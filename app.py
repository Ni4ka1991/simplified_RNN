#!/usr/bin/env python3

from helper_func import *
import torch, torch.nn as nn


text = loadTextFromFile( "data/Bacteria in Daily Life.txt" )
print( text[:1000] )
text = encodeText( text )
print( text[:10] )


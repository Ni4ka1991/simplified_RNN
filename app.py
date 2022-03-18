#!/usr/bin/env python3

import re

alphabet = ' abcdefghijklmnopqrstuvwxyz'


def characterToOneHotVector( character ):
    try:
        index = alphabet.index( character )
    except:
        index = 0  
    return [0] * index + [1] + ( len( alphabet ) - index - 1)  * [0]



def oneHotVectorToCharacter( vector ):
    return alphabet[torch.argmax( vector )]


def loadTextFromFile( name ):
    file = open( name, "r" )
    text = file.read()
    file.close()


    textProcessed = ""

    for i in text.lower():
        c = alphabet.find( i )
        if c == -1:
            textProcessed += " "
        else:
            textProcessed += i

#    textProcessed = re.sub( "page", " ", textProcessed )
    textProcessed = re.sub( " +", " ", textProcessed )
                                                                                
    return textProcessed

text = loadTextFromFile( "data/Bacteria in Daily Life.txt" )
print( text )

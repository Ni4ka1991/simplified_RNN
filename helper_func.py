#module helper_func

import re
from data import *

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


def encodeText( text ):
    textEncode = []
    for c in text:
        textEncode.append( characterToOneHotVector( c ))
    return textEncode


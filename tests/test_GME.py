import tensorflow as tf
import sinact as sa

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

import numpy as np
from numpy import genfromtxt

def load_data( filepath = "GME.csv" ):
    alldata = genfromtxt(filepath, delimiter=',')
    return alldata

def test( nwaves, data ):
    A_in = Input(shape=(5,), name='A_in')
    out = sa.MultipleSineWaveActivation(nwaves)(A_in)
    model = Model(inputs=[A_in], outputs=out)
    model.compile(optimizer='adam', loss='mean_squared_error')
    print( model.summary() )

    indata = data[:,0]
    outdata = data[:,1]

    model.fit( x=indata, y=outdata, epochs=10 )

if __name__ == '__main__':
    data = load_data()
    print( data.shape )
    indata = data[:,0]
    print( indata.shape )

    test( 1, data )

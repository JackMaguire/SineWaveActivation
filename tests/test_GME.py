import tensorflow as tf
import sinact as sa

import math

import keras
from keras.layers import *
from keras.models import Model

import numpy as np
from numpy import genfromtxt

class MultipleSineWaveActivationAnalyzer:
  def __init__( self, layer ):
    self.weights = [ i.numpy()[0] for i in layer.trainable_weights ]

  def analyze( self, x ):
    vals = [ self.weights[0], ]
    for i in range( 1, len(self.weights), 3 ):
        vals.append( self.weights[i] * math.sin( self.weights[i+1] * x + self.weights[i+2] ) )
    vals.append( sum(vals) )
    return vals


def load_data( filepath = "GME.csv" ):
    alldata = genfromtxt(filepath, delimiter=',')
    return alldata

def group_list(l, group_size):
    """
    :param l:           list
    :param group_size:  size of each group
    :return:            Yields successive group-sized lists from l.
    """
    for i in range(0, len(l), group_size):
        yield l[i:i+group_size]

def test( nwaves, data ):
    A_in = Input(shape=(1,), name='A_in')
    out = sa.MultipleSineWaveActivation(nwaves, use_bias = True, name="outwaves")(A_in)
    model = Model(inputs=[A_in], outputs=out)
    model.compile(optimizer='adam', loss='mean_squared_error')
    print( model.summary() )

    indata = data[:,0]
    outdata = data[:,1]

    lr = keras.callbacks.ReduceLROnPlateau(
        monitor="loss",
        factor=0.1,
        patience=10,
        verbose=0,
        mode="auto",
        min_delta=0.0001,
        cooldown=0,
        min_lr=0
    )

    stop = keras.callbacks.EarlyStopping(monitor="loss",patience=10)

    model.fit( x=indata, y=outdata, epochs=1000, callbacks=[lr,stop] )

    xs = np.linspace(-1.0,1.0,1001)
    xs = xs.reshape( (1001, 1) )
    print( xs.shape )

    ys = model.predict(xs)
    print( xs.shape, ys.shape )

    analyzer = MultipleSineWaveActivationAnalyzer( model.get_layer("outwaves") )

    for i in range( 0, len(xs) ):
        #print( "DATA", xs[i][0], ys[i][0], *analyzer.analyze(xs[i][0]) )
        print( "DATA", xs[i][0], ys[i][0] )

    model.save( "model.h5" )

    #print( [ i.numpy()[0] for i in model.trainable_weights ] )
    #print( model.get_layer("outwaves") )
    #print( model.get_layer("outwaves").trainable_weights )

if __name__ == '__main__':
    data = load_data()

    # sanity check:
    #print( data.shape )
    #indata = data[:,0]
    #print( indata.shape )

    test( 5000, data )

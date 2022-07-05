import tensorflow as tf
import sinact as sa

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def test1():
    A_in = Input(shape=(5,), name='A_in')
    out = sa.SingleSineWaveActivation()(A_in)
    #out = sa.PReLU()(A_in)
    #out = PReLUcopy()(A_in)
    model = Model(inputs=[A_in], outputs=out)
    model.compile(optimizer='adam', loss='mean_squared_error')
    print( model.summary() )

def test2( nwaves ):
    A_in = Input(shape=(5,), name='A_in')
    out = sa.MultipleSineWaveActivation(nwaves)(A_in)
    model = Model(inputs=[A_in], outputs=out)
    model.compile(optimizer='adam', loss='mean_squared_error')
    print( model.summary() )


if __name__ == '__main__':
    test1()
    test2(1)
    test2(2)
    test2(5)

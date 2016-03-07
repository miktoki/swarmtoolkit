import nose.tools as nt
import swtools
import numpy as np 
import datetime as dt
import unittest
import spacepy

import swtools.ops 
import swtools.aux 
import swtools.sph 
import swtools.ops 


swtools.debug_info(-1)#as error will be generated when testing if errors work, stdout logging is suppressed

 
def test_get_l_maxmin():
  assert swtools.sph.get_l_maxmin(65,lmin= 4) == ( 8, 4)
  assert swtools.sph.get_l_maxmin(57,lmax=28) == (28,28)
  assert swtools.sph.get_l_maxmin(24,lmin= 1) == ( 4, 1)
  assert swtools.sph.get_l_maxmin(25        ) == (12,12)
  assert swtools.sph.get_l_maxmin(15        ) == ( 3, 1)
  assert swtools.sph.get_l_maxmin(21,lmax= 4) == ( 4, 2)


def test_read_shc():
  shc_f=swtools.read_shc('simple_model.shc')

  assert shc_f[1] == 1
  assert shc_f[2] == 2
  np.testing.assert_allclose(shc_f[3],np.array([2000,2010]),atol=1e-8)
  np.testing.assert_allclose(shc_f[0][0],np.array([ 
    4.1,  3.9,  0.1,  0.1,  0.2,  0.5,  0.3,  0.2,  0. ,  0. ,  0.8,
    0.1,  0.2,  0. ,  0.2,  0. ,  0. ,  0. ,  0.3,  0. ,  0. ]),
    atol=1e-8)

  shc_f=swtools.read_shc('simple_model.shc',cols=[3])
  np.testing.assert_allclose(shc_f[0][0],np.array([
    5. ,  1. ,  4.4,  0.1,  0.2,  0.5,  0.4,  0.2,  0. ,  0. ,  0.1,  
    0.2,  0.1,  0.1,  0.1,  0. ,  0. ,  0. ,  0.4,  0. ,  0. ]),
    atol=1e-8)


def test_bnec():
  lat = np.arange(0.5,2)#0.5,1.5
  lon = np.arange(0.8,2)#0.8,1.8

  Bnec1 = swtools.get_Bnec('simple_model.shc',lat,lon,lmin_file=2)
  Bnec2 = swtools.get_Bnec('simple_model.shc',lat,lon,r=1.2,lmin_file=2)
  
  B0x = Bnec1[0,0]
  B1x = Bnec2[1,0]
  B0y = Bnec1[0,1]
  B1y = Bnec2[1,1]
  B0z = Bnec1[0,2] 
  B1z = Bnec2[1,2]

  np.testing.assert_allclose(B0x[0,0],np.array([-6.11950869]),atol=1e-8)
  np.testing.assert_allclose(B0y[0,1],np.array([-0.28767098]),atol=1e-8)
  np.testing.assert_allclose(B0z[1,0],np.array([ 3.23692366]),atol=1e-8)
  
  np.testing.assert_allclose(B1x[0,0],np.array([-0.58513366]),atol=1e-8)
  np.testing.assert_allclose(B1y[0,1],np.array([-0.33464752]),atol=1e-8)
  np.testing.assert_allclose(B1z[1,0],np.array([ 3.64336225]),atol=1e-8)
  

def test_legendre():
  lat_rad = np.arange(0.5,2)*np.pi/180
  P, dP = np.zeros((2,3,3)), np.zeros((2,3,3))
  for la in range(len(lat_rad)):
    P[la], dP[la] = swtools.sph._get_legendre(2,2,np.pi/2-lat_rad[la],True)

  Pcomp = np.array([[
            [1., 0., 0.],
            [0.00872654, 0.99996192, 0.],
            [-0.49988577, 0.01511423, 0.86595945]    
        ],[
            [1., 0., 0.],
            [0.02617695, 0.99965732, 0.],
            [-0.49897215, 0.04532427, 0.86543197]
        ]])
  dPcomp = np.array([[
            [0., 0., 0.],
            [-0.99996192, 0.00872654, 0.],
            [-0.02617861, -1.73178701, 0.01511423]    
        ],[
            [0., 0., 0.],
            [-0.99965732, 0.02617695, 0.],
            [-0.07850393, -1.72967709, 0.04532427]
        ]])
  np.testing.assert_allclose( P, Pcomp,atol=1e-8)
  np.testing.assert_allclose(dP,dPcomp,atol=1e-8)



if __name__ == '__main__':
  import nose
  import os

  simple_model = """#simple model for testing purposes
#
#
2 4 2 1 0
    2000.0 2010.0
2  0 4.1 5.0
2  1 3.9 1.0
2 -1 0.1 4.4
2  2 0.1 0.1
2 -2 0.2 0.2
3  0 0.5 0.5
3  1 0.3 0.4
3 -1 0.2 0.2
3  2 0.0 0.0
3 -2 0.0 0.0
3  3 0.8 0.1
3 -3 0.1 0.2
4  0 0.2 0.1
4  1 0.0 0.1
4 -1 0.2 0.1
4  2 0.0 0.0
4 -2 0.0 0.0
4  3 0.0 0.0
4 -3 0.3 0.4
4  4 0.0 0.0
4 -4 0.0 0.0"""
  with open('simple_model.shc') as f:
    f.write(simple_model)
  
  nose.main()
  
  os.remove('simple_model.shc')
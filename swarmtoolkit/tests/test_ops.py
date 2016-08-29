import nose.tools as nt
import swarmtoolkit as st
import numpy as np 
import datetime as dt
import unittest
import spacepy

import swarmtoolkit.ops 
import swarmtoolkit.aux 
import swarmtoolkit.sph 
import swarmtoolkit.ops 


st.debug_info(-1)#as error will be generated when testing if errors work, stdout logging is suppressed

 
class Shift_params_test(unittest.TestCase):
  """Testcases for shift param"""
  def to_sec(self,a):
    s = np.vectorize(lambda x:(x-dt.datetime(1970,1,1,0,0,0,0)).
      total_seconds()) 
    return s(a)

  def setUp(self):
    "set up test fixtures"
    N=10
    datet1,datet2=np.empty(N,dtype=object),np.empty(N,dtype=object)
    param1,param2=np.arange(N),np.arange(N//2,N+N//2)
    for h in range(N):
      datet1[h]=dt.datetime(1234,5,6,h,0,0,0)
      datet2[h]=dt.datetime(1234,5,6,h,0,0,0)
    self.inval=[param1,param2,datet1,datet2]

  def tearDown(self):
    self.inval=None
  
  def test_null(self):
    inval=self.inval
    inval.append(0)
    outval=st.shift_param(*inval)
    for i in range(4):
      f=lambda x:x
      if i>1:
        f=self.to_sec
      np.testing.assert_allclose(f(outval[i]),f(inval[i]),rtol=1e-8,
        err_msg="array {} in {} not within rtol='{}' of expected values".format(i,__name__,1e-8))
  
  def test_simple_p(self): #shift so step_c=0
    n=3
    inval=self.inval
    inval.append(inval[2][n]-inval[2][0])
    outval=st.shift_param(*inval)

    nt.eq_(len(outval[0]),(len(inval[0])-n),"Expected p1 array to have length {}".format(len(inval[0]-n)))
    nt.eq_(len(outval[1]),(len(inval[1])-n),"Expected p2 array to have length {}".format(len(inval[1]-n)))
    
    nt.eq_(outval[3][0],inval[3][n],"Unexpected value of unshifted datetime array[0] of equal length: '{}'!='{}'"
      .format(outval[3][0],inval[3][n]))
    nt.eq_(outval[3][-1],inval[3][-1],"Unexpected value of unshifted datetime array[-1] of equal length: '{}'!='{}'"
      .format(outval[3][-1],inval[3][-1]))
    
    nt.eq_(outval[1][0],inval[1][n],"Unexpected value of unshifted param array[0] of equal length: '{}'!='{}'"
      .format(outval[1][0],inval[1][n]))
    nt.eq_(outval[1][-1],inval[1][-1],"Unexpected value of unshifted param array[-1] of equal length: '{}'!='{}'"
      .format(outval[1][-1],inval[1][-1]))
    
    
    np.testing.assert_allclose(self.to_sec(inval[2][n:]),self.to_sec(outval[2]),rtol=1e8,
      err_msg="Unexpected value of shifted datetime array of equal value")
    np.testing.assert_allclose(inval[0][:-n],outval[0],rtol=1e8,
      err_msg="Unexpected value of shifted param array of equal value")
  
  def test_simple_n(self): #shift so step_c=0
    n=3
    inval=self.inval
    #negative shift using delta in seconds
    inval.append((inval[2][0]-inval[2][n]).total_seconds())
    outval=st.shift_param(*inval)
    
    nt.eq_(len(outval[0]),(len(inval[0])-n),"Expected p1 array to have length {}".format(len(inval[0]-n)))
    nt.eq_(len(outval[1]),(len(inval[1])-n),"Expected p2 array to have length {}".format(len(inval[1]-n)))
    
    nt.eq_(outval[3][0],inval[3][0],"Unexpected value of unshifted datetime array[0] of equal length: '{}'!='{}'"
      .format(outval[3][0],inval[3][0]))
    nt.eq_(outval[3][-1],inval[3][-n-1],"Unexpected value of unshifted datetime array[-1] of equal length: '{}'!='{}'"
      .format(outval[3][-1],inval[3][-n-1]))
    
    nt.eq_(outval[1][0],inval[1][0],"Unexpected value of unshifted param array[0] of equal length: '{}'!='{}'"
      .format(outval[1][0],inval[1][0]))
    nt.eq_(outval[1][-1],inval[1][-n-1],"Unexpected value of unshifted param array[-1] of equal length: '{}'!='{}'"
      .format(outval[1][-1],inval[1][-n-1]))
    
    np.testing.assert_allclose(self.to_sec(inval[2][:-n]),self.to_sec(outval[2]),rtol=1e8,
      err_msg="Unexpected value of shifted datetime array of equal value")
    np.testing.assert_allclose(inval[0][n:],outval[0],rtol=1e8,
      err_msg="Unexpected value of shifted param array of equal value")
   
  def test_out_of_range(self):#too large/small shift
    inval=self.inval
    inval.append((inval[2][-1]-inval[2][0]).total_seconds() +1) #too large (positive) shift
    outval=st.shift_param(*inval)
    nt.eq_(outval,None,"Expected arrays to have no overlap and return None-tuple(+)")

    inval[4] = (inval[2][0]-inval[2][-1]).total_seconds() -1 #too large (negative) shift
    outval=st.shift_param(*inval)
    nt.eq_(outval,None,"Expected arrays to have no overlap and return None")
  
  def test_disjoined(self):#datet1[-1]<datet2[0]
    inval=self.inval
    delta=(inval[2][1]-inval[2][0])*11
    inval[3]+=delta
    inval.append(delta)
    
    outval=st.shift_param(*inval)
    np.testing.assert_allclose(self.to_sec(outval[2]-delta),self.to_sec(inval[2]),rtol=1e-5,
      err_msg="Unexpected value of shifted datetime array for disjoined arrays")
    np.testing.assert_allclose(self.to_sec(outval[3]),self.to_sec(inval[3]),rtol=1e-8,
      err_msg="Unexpected value of shifted param array for disjoined arrays")
    
  def test_ineq_len(self):#length arrays are different
    N=10
    Np=15
    datet1,datet2=np.empty(Np,dtype=object),np.empty(N-2,dtype=object)
    param1,param2=np.arange(Np),np.arange(N-2,2*(N-2))
    for h in range(2,N):
      datet2[h-2]=dt.datetime(1234,5,6,h,0,0,0)
    for h in range(Np):
      datet1[h]=dt.datetime(1234,5,6,h,0,0,0)

    delta=datet1[1]-datet1[0]

    outval=st.shift_param(param1,param2,datet1,datet2,delta)
    nt.eq_(outval[2][0],datet2[0],"Unexpected value of shifted datetime array[0] of inequal length: '{}'!='{}'"
      .format(outval[2][0],datet2[0]))
    nt.eq_(outval[2][-1],datet2[-1],"Unexpected value of shifted datetime array[-1] of inequal length: '{}'!='{}'"
      .format(outval[2][-1],datet2[-1]))
    
    nt.eq_(outval[3][0],datet2[0],"Unexpected value of unshifted datetime array[0] of inequal value: '{}'!='{}'"
      .format(outval[3][0],datet2[0]))
    nt.eq_(outval[3][-1],datet2[-1],"Unexpected value of unshifted datetime array[-1] of inequal value: '{}'!='{}'"
      .format(outval[3][-1],datet2[-1]))
    
    np.testing.assert_allclose(outval[0],param1[1:N-2+1],rtol=1e-8,err_msg="Unexpected value of shifted param array of inequal value")

  def test_auto(self):
    h=3600
    out=st.shift_param(*self.inval,auto=True,return_delta=True,v=0,dt_lim=[-6*h,0])
    delta_t=out[-1]
    np.testing.assert_allclose(out[0],out[1],rtol=1e-2,
        err_msg="array in {} not within rtol='{}' of expected values".format(__name__,1e-2))
    np.testing.assert_allclose((np.abs(out[2][0]-out[3][0]).total_seconds()),np.zeros(len(out[2])),atol=h/10,
        err_msg="array in {} not within atol='{}' of expected values".format(__name__,h/10))
    assert abs(delta_t)-5*h<0.1*h

    
class Concatenation_tests(unittest.TestCase):
  def setUp(self):
    "set up test fixtures"
    N=10
    self.d11=np.arange(10)
    self.t1 = np.array([dt.datetime(2000,1,1)+dt.timedelta(hours=x) for x in range(0, 30)],dtype=object)
    self.d12=np.arange(-10,20)

    self.d21=np.eye(3)
    self.d22=np.eye(3,4)
    self.d23=np.eye(4,3)

    self.d31=np.ones((3,3,2))
    self.d32=np.ones((3,4,2))*2
    self.d33=np.arange(30).reshape((3,5,2))

  def tearDown(self):
    pass

  def test_1d(self):
    #len checks
    out1=st.sw_io.concatenate_values(self.d11,self.d12)
    nt.eq_(len(out1),(len(self.d11)+len(self.d12)),"Incorrect concatenation length {}!={}+{}"
      .format(len(out1),len(self.d11),len(self.d12)))
    
    out2=st.sw_io.concatenate_values(self.d11,self.d12,self.d11)
    nt.eq_(len(out2),(len(self.d11)+len(self.d12)+len(self.d11)),
      "Incorrect concatenation length {}!={}+{}+{}"
      .format(len(out2),len(self.d11),len(self.d12),len(self.d11)))
    
    #value checks
    nt.eq_(out1[0],0,"Unexpected value in concatenated array: {}!={}"
      .format(out1[0],0))
    nt.eq_(out1[-1],19,"Unexpected value in concatenated array: {}!={}"
      .format(out1[-1],19))
    nt.eq_(out2[-1],9,"Unexpected value in concatenated array: {}!={}"
      .format(out1[-1],9))
  
  def test_2d(self):
    
    out1=st.sw_io.concatenate_values(self.d21,self.d21*2,self.d21)
    out2=st.sw_io.concatenate_values(self.d21,self.d22*3)
    out3=st.sw_io.concatenate_values(self.d21,self.d23*4)
    out4=st.sw_io.concatenate_values(self.d21,self.d21*10,axis=0)
    
    #shape checks
    nt.eq_(np.shape(out1),(3,9),"Unexpected shape of concatenated array: {}!={}"
      .format(np.shape(out1),(3,9)))
    nt.eq_(np.shape(out2),(3,7),"Unexpected shape of concatenated array: {}!={}"
      .format(np.shape(out2),(3,7)))
    nt.eq_(np.shape(out3),(7,3),"Unexpected shape of concatenated array: {}!={}"
      .format(np.shape(out3),(7,3)))
    nt.eq_(np.shape(out4),(6,3),"Unexpected shape of concatenated array: {}!={}"
      .format(np.shape(out4),(6,3)))
    
    
    #value checks
    nt.eq_(int(out1[0][3]),2,"Unexpected value in concatenated array: {}!={}"
      .format(out1[0][3],2))
    nt.eq_(int(np.sum(out1)),12,"Unexpected value of sum of concatenated array: {}!={}"
      .format(int(np.sum(out1)),12))
    nt.eq_(int(out2[1][4]),3,"Unexpected value in concatenated array: {}!={}"
      .format(out2[1][4],3))
    nt.eq_(int(out3[3][0]),4,"Unexpected value in concatenated array: {}!={}"
      .format(out3[3][0],4))
    nt.eq_(int(np.sum(out4)),33,"Unexpected value of sum of concatenated array: {}!={}"
      .format(int(np.sum(out4)),33))
    

  def test_3d(self):
    out1=st.sw_io.concatenate_values(self.d31,3*self.d31,axis=0)
    out2=st.sw_io.concatenate_values(self.d31,self.d32)
    out3=st.sw_io.concatenate_values(self.d31,self.d33)
    out4=st.sw_io.concatenate_values(self.d31,self.d32,self.d33)

    #shape checks
    nt.eq_(np.shape(out1),(6,3,2),"Unexpected shape of concatenated array: {}!={}"
      .format(np.shape(out1),(6,3,2)))
    nt.eq_(np.shape(out2),(3,7,2),"Unexpected shape of concatenated array: {}!={}"
      .format(np.shape(out2),(3,7,2)))
    nt.eq_(np.shape(out3),(3,8,2),"Unexpected shape of concatenated array: {}!={}"
      .format(np.shape(out3),(3,8,2)))
    nt.eq_(np.shape(out4),(3,12,2),"Unexpected shape of concatenated array: {}!={}"
      .format(np.shape(out4),(3,12,2)))
    
    #value checks
    def rsum(mv):
      """sum of range(mv)"""
      return int((mv+1)*mv/2)

    def prod(a):
      """product of values in a"""
      s=1
      for i in a:
        s*=i
      return s

    nt.eq_(int(out1[3][2][1]),3,"Unexpected value in concatenated array: {}!={}"
      .format(int(out1[3][2][1]),3))
    nt.eq_(int(np.sum(out4)),prod(np.shape(self.d31)) + prod(np.shape(self.d32))*2 + rsum(prod(np.shape(self.d33))-1),"Unexpected value of sum of concatenated array: {}!={}"
      .format(int(np.sum(out4)),prod(np.shape(self.d31)) + prod(np.shape(self.d32))*2 + rsum(prod(np.shape(self.d33))-1)))

 

  
def test_where_overlap():
  t1=np.array([dt.datetime(1990,2,4),dt.datetime(2000,3,8),dt.datetime(2010,1,1),dt.datetime(2020,1,1),dt.datetime(2030,1,1)],dtype=object)
  t2=np.array([dt.datetime(1990,2,3),dt.datetime(2000,3,8),dt.datetime(2010,1,1)],dtype=object)
  t3=np.array([dt.datetime(1990,2,4),dt.datetime(2000,3,9),dt.datetime(2010,2,1),dt.datetime(2020,2,1),dt.datetime(2030,1,1)],dtype=object)
  delta=dt.timedelta(days=7000)#approx 19 yrs
  out1=st.ops.where_overlap(t1,t2)
  out2=st.ops.where_overlap(t1,t3)
  out3=st.ops.where_overlap(t1,t2,delta)
  out4=st.ops.where_overlap(t1,t3,-delta)
  
  nt.eq_(out1,(range(3),range(1,3)))
  nt.eq_(out2,(range(5),range(5)))
  nt.eq_(out3,(range(1),range(2,3)))
  nt.eq_(out4,(range(2,5),range(3)))


def test_cyclic_rising():
  from numpy import pi
  np.random.seed(40)
  A = 0.75
  ATOL = 1e-8
  for N in [50,51]:
    for v0 in [0,pi/4,pi/3,pi/2-pi/10]:
      if N==50 and v0==0:#one hi_res test
        N = 500
      a_0 = A*np.sin(np.linspace(v0,6*pi+2*np.random.random(),N))
      a_r = st.cyclic2rising(a_0,lim=[-A,A])
      a_c = st.rising2cyclic(a_r,lim=[-A,A])
      
      not_rising = np.count_nonzero((a_r<=np.roll(a_r,1))[1:]) 

      assert not_rising == 0, 'Not all values are rising: {} of {} are not '.format(not_rising,N)


      np.testing.assert_allclose(a_0,a_c,atol=ATOL,
        err_msg="rising -> cyclic -> rising. {} did not reproduce original array satisfactorily[atol={}]".format(test_cyclic_rising.__name__,ATOL))
  pass


def test_fourier_transform():
  from numpy import pi
  ATOL=1e-8
  N = 100
  T=N/15
  w=2*pi/T
  a_0 = np.exp(1j*w*np.arange(N))
  a_fft,f_fft=st.fourier_transform(a_0,1)
  assert abs(f_fft[1]-0.01)<ATOL

  for i in range(N):
    if i==15:
      assert abs(a_fft[i]-N)<ATOL
    else:
      assert abs(a_fft[i])<ATOL 

  a_out=np.fft.ifft(a_fft)
  
  np.testing.assert_allclose(a_0,a_out,atol=ATOL,
    err_msg="fft inverse of fourier did not retrieve original")


def test_where_diff():
  a = np.ones(10)
  i = 5
  d = 3
  a[i] = d

  out1=st.where_diff(a)[0]
  out2=st.where_diff(a,rtol=1)[0]
  out3=st.where_diff(a,atol=1)[0]
  out4=st.where_diff(a,rtol=1,atol=1,no_jump=True)[0]

  assert out1[0]==i
  assert out1[1]==(i+1)
  assert out2[0]==(i+1)
  assert (out3==out1).all()
  assert (out4==np.array([0,1,2,3,4,5,7,8,9])).all()

  b = np.ones((5,2))
  i = 3,1
  d = 3
  b[i]=d

  out5_comp=np.array([[0, 0, 1, 1, 2, 2, 3, 4],[0, 1, 0, 1, 0, 1, 0, 0]])
  out6_comp=(np.array([3, 3]), np.array([0, 1]))
  out5=st.where_diff(b,no_jump=True,axis=0)
  out6=st.where_diff(b,axis=1)

  np.testing.assert_allclose(out5,out5_comp)
  np.testing.assert_allclose(out6,out6_comp)
  

def test_interpolate2d_sphere():
  pass


class Align_params_test(unittest.TestCase):
  """Testcases for shift param"""
  def to_sec(self,a):
    s = np.vectorize(lambda x:(x-dt.datetime(1970,1,1,0,0,0,0)).
      total_seconds()) 
    return s(a)

  def setUp(self):
    "set up test fixtures"
    N=10
    datet1,datet2=np.empty(N,dtype=object),np.empty(N,dtype=object)
    param1,param2=np.arange(N)**2,np.arange(N//2,N+N//2)**2
    for h in range(N):
      datet1[h]=dt.datetime(1234,5,6,h,0,0,0)
      datet2[h]=dt.datetime(1234,5,6,h,0,0,0)
    self.inval=[param1,param2,datet1,datet2]

  def tearDown(self):
    self.datet1,self.datet2,self.param1,self.param2,self.inval = (None,)*5

  def test_null(self):
    out  = st.align_param(*self.inval)
    atol = 1e-8
    for i in range(2):
      np.testing.assert_allclose(out[i],self.inval[i],atol=atol,
        err_msg="p array {} in {} not aligned".format(i,self.test_null.__name__))
      np.testing.assert_allclose(self.to_sec(out[2]),self.to_sec(self.inval[i+2]),atol=atol,
        err_msg="t array {} in {} not aligned".format(i,self.test_null.__name__))

  def test_t_align(self):
    inval = self.inval
    tdiff = dt.timedelta(seconds=(inval[2][1]-inval[2][0]).total_seconds()/2)
    inval[2] = inval[2]-tdiff
    out = st.align_param(*inval)

    np.testing.assert_allclose(out[0],inval[0][1:],atol=1e-8
      ,err_msg="p array in {} not aligned".format(self.test_t_align.__name__))
    np.testing.assert_allclose(self.to_sec(out[2]),self.to_sec(inval[2])[1:],atol=1e-8
      ,err_msg="t array in {} not aligned".format(self.test_t_align.__name__))

  def test_sampling(self):
    p1,p2,t1,t2 = self.inval
    t2l = t2[::3]#change frequency
    p2l = p2[::3]

    out1 = st.align_param(p1,p2l,t1,t2l,align_to=True)#upsample
    out2 = st.align_param(p1,p2l,t1,t2l,align_to=False)#downsample
    
    np.testing.assert_allclose(p1,out1[0],atol=1e-8,
      err_msg="p array in {} not aligned/sampled correctly".format(self.test_sampling.__name__))
    np.testing.assert_allclose(p2,out1[1],atol=1e-8,
      err_msg="p array in {} not aligned/sampled correctly".format(self.test_sampling.__name__))
    np.testing.assert_allclose(p1[::3],out2[0],atol=1e-8,
      err_msg="p array in {} not aligned/sampled correctly".format(self.test_sampling.__name__))
    np.testing.assert_allclose(p2l,out2[1],atol=1e-8,
      err_msg="p array in {} not aligned/sampled correctly".format(self.test_sampling.__name__))
    

  def test_spline(self):
    inval = self.inval
    tdiff = dt.timedelta(seconds=(inval[2][1]-inval[2][0]).total_seconds()/2)
    inval[2] = inval[2]-tdiff
    outk1 = st.align_param(*inval,k=1)
    outk3 = st.align_param(*inval,k=3)

    assert outk1[1][5]!=outk3[1][5], 'spline k=1  gives same results as spline k=3'
    np.testing.assert_allclose(outk1[1],outk3[1],atol=1e-1,rtol=1e-2
      ,err_msg="<{} difference in splines in {}".format(1e-2,self.test_sampling.__name__))


if __name__ == '__main__':
  import nose
  nose.main()

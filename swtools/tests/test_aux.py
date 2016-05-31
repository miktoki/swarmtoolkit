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


class In2range_tests(unittest.TestCase):

  def test_basic(self):
    i='2'
    assert swtools.aux._in2range(10,i)==[2]

  def test_hyphen(self):
    i='-'
    assert swtools.aux._in2range(10,i)==range(1,11)
    i='2-'
    assert swtools.aux._in2range(10,i)==range(2,11)
    i='-3'
    assert swtools.aux._in2range(10,i)==range(1,4)


  def test_comma(self):
    i='2,4'
    assert list(swtools.aux._in2range(10,i))==[2,4]
    i='4,2,4'
    assert list(swtools.aux._in2range(10,i))==[4,2,4]
    
  @nt.raises(ValueError)
  def test_errors1(self):
    list(swtools.aux._in2range(10,',2'))
  @nt.raises(ValueError)
  def test_errors2(self):
    list(swtools.aux._in2range(10,'2,'))

  def test_combined(self):
    i='2-,3'
    assert list(swtools.aux._in2range(10,i))==(list(range(2,11))+[3])
    i='-2,3-'
    assert list(swtools.aux._in2range(10,i))==(list(range(1,11)))
    i='-2,4-5,9-'
    assert list(swtools.aux._in2range(10,i))==[1,2,4,5,9,10]


def test_to_list():
  a='s'
  b=[1,2]
  c=[a]
  d=np.array(b)
  e=1
  f={'a':e}
  assert swtools.aux._tolist(a)==c
  assert swtools.aux._tolist(b)==b
  assert swtools.aux._tolist(c)==c
  assert (swtools.aux._tolist(d)==d).all()
  assert swtools.aux._tolist(e)==[e]
  assert swtools.aux._tolist(f)==[e]


def test_open_list():
  a='s'
  b=[1,2]
  c=[a]
  d=np.array(b)
  e=1
  f={'a':e}
  assert swtools.aux._single_item_list_open(a)==a
  assert swtools.aux._single_item_list_open(b)==b
  assert swtools.aux._single_item_list_open(c)==a
  assert (swtools.aux._single_item_list_open(d)==d).all()
  assert swtools.aux._single_item_list_open(e)==e
  assert swtools.aux._single_item_list_open(f)==e


class datetime_aux_test(unittest.TestCase):
  """Testcases for aux datetime functions"""
  def setUp(self):
    self.t2=dt.datetime(2001,3,2,23,59,59)
    self.t1=dt.datetime(2001,2,1,22,58,58)
    self.d21=self.t2-self.t1

  def tearDown(self):
    #
    pass

  def test_period(self):
    d1 = dt.timedelta(days=1)
    
    p1=swtools.aux._set_period(self.t1,self.t2,self.d21)
    p2=swtools.aux._set_period(self.t1,self.t2)
    p3=swtools.aux._set_period(self.t1,duration=self.d21)
    p4=swtools.aux._set_period(end_t=self.t2,duration=self.d21)
    p5=swtools.aux._set_period(start_t=self.t2,duration=1)
    
    comp1=[self.t1,self.t2]
    comp2=[self.t2,self.t2+d1]

    assert p1==comp1
    assert p2==comp1
    assert p3==comp1
    assert p4==comp1
    assert p5==comp2

    assert swtools.aux._is_in_period(self.t1 + 5*d1,comp1)
    assert swtools.aux._is_in_period(self.t1 + 0*d1,comp1)
    assert swtools.aux._is_in_period(self.t2 - 1*d1,comp1)
    assert swtools.aux._is_in_period(self.t2 - 0*d1,comp1)

    assert not swtools.aux._is_in_period(self.t2 + 1*d1,comp1)
    assert not swtools.aux._is_in_period(self.t1 - 1*d1,comp1)

  def test_MJD(self):
    EPOCH  = swtools.aux.EPOCH 
    s_day  = swtools.aux.SECONDS_PER_DAY
    ms_day = swtools.aux.MILLISECONDS_PER_DAY
    
    d10  = self.t1-EPOCH
    d20 = d10 + self.d21
    
    d10s = d10.total_seconds()
    d20s = d20.total_seconds()

    d10_day = d10s//s_day
    d10_ms  = (d10s*1000)%ms_day
    
    dd = np.array([self.t1,self.t2],dtype=object)
    ds = np.array([d10s,d20s])
    print((swtools.aux._MJD2000_datetime(d10_day,d10_ms)-EPOCH - d10).total_seconds())
    assert abs((swtools.aux._MJD2000_datetime(d10_day,d10_ms)-self.t1).total_seconds())<1e-8
    assert (swtools.aux._MJD2000sec_datetime(d10s)-self.t1).total_seconds()<1e-8
    assert isinstance(swtools.aux._MJD2000sec_datetime(ds),np.ndarray)

  def test_str2dt(self):
    s = '2001 3 2 23 59 59'
    assert self.t2 == swtools.aux._str2dt(s)

  def test_timedelta_conv(self):
    t = 10

    d = dt.timedelta(seconds=t)

    assert swtools.aux._from_timedelta(d,1) == t
    assert swtools.aux._from_timedelta(d,1e3) == t*1e3
    assert swtools.aux._to_timedelta(t,'seconds') == d
    assert swtools.aux._to_timedelta(t,'days').total_seconds() == t*swtools.aux.SECONDS_PER_DAY

  def test_datetime_conv(self):
    t1 = "2001-03-02"
    t2 = "20010302"
    t3 = "20010302T235959"
    t4 = "20010302235959"
    t5 = self.t2
    t6 = (self.t2-swtools.aux.EPOCH).total_seconds()/swtools.aux.SECONDS_PER_DAY
    t7 = (self.t2-swtools.aux.EPOCH).total_seconds()//swtools.aux.SECONDS_PER_DAY

    for t in [t1,t2,t3,t4,t5,t6,t7]:
      assert swtools.aux._to_datetime(t).day==self.t2.day
      assert swtools.aux._to_datetime(t).month==self.t2.month
      assert swtools.aux._to_datetime(t).year==self.t2.year

  def test_dec_year(self):
    t1, t2 = self.t1, self.t2
    tlist  = swtools.aux._to_dec_year_list([t1,t2])
    t1_dec = t1.year+t1.month/12 + (t1.day +(t1.hour +\
       (t1.minute + (t1.second+ t1.microsecond/1e6)/60)/60)/24)/365
    t2_dec = t2.year+t2.month/12 + (t2.day +(t2.hour +\
       (t2.minute + (t2.second+ t2.microsecond/1e6)/60)/60)/24)/365

    assert tlist[0] == t1_dec
    assert tlist[1] == t2_dec


def test_importable():### 
  #ensure something(ie datetime) is not imported, then run function
  #check that it is imported

  try:
    math.sqrt(1)
    del math
  except NameError:
    pass
  assert swtools.aux._importable('math')


  pass


if __name__ == '__main__':
  import nose
  nose.main()
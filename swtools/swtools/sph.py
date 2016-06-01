#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
#from numba import jit
from scipy.interpolate import splrep,splev

from . import aux 

__all__=[ 'get_Bnec',
          'get_l_maxmin',
          'read_shc']


def _get_legendre(PdP, theta, schmidtnormalize, out):
    """Legendre functions and its derivatives
    
        The functions are computed iteratively, using an algorithm from 
        "Spacecraft Attitude Determination and Control" by James Richard Wertz
        (http://books.google.no/books?id=GtzzpUN8VEoC&lpg=PP1&pg=PA781#v=onepage)
    """
    
    
    if hasattr(theta,'__len__'):
        theta = theta[0]
        schmidtnormalize = schmidtnormalize[0]

    shp = PdP.shape
    nmax = shp[1]-1
    mmax = shp[2]-1

    S = np.zeros((nmax+1,mmax+1))

    #shorthand notation
    c=np.cos(theta)
    s=np.sin(theta)
    
    #Initialization
    S[0, 0] = 1.
    PdP[0, 0, 0] = 1.
    #dP[0,0]= 0
    
    #n=1 m=0
    PdP[:,1,0]  = c * PdP[:,0,0]
    PdP[1,1,0] -= s * PdP[0,0,0]
    S[1,0] = S[0,0]
    
    #n=1,m=1
    if mmax:
      PdP[:, 1, 1]  = s * PdP[:,0,0]
      PdP[1, 1, 1] += c * PdP[0,0,0]
      S[1, 1] = S[1,0]
            
    for n in range(2, nmax +1):
        #m=0
        S[n, 0] = S[n - 1, 0] * (2.*n - 1)/n
        Knm = ((n - 1)**2) / ((2*n - 1)*(2*n - 3))
        #P[n,0]  = c * P[n-1,0] - Knm*P[n-2,0]
        #dP[n,0] = c * dP[n-1,0] - s * P[n-1,0] - Knm * dP[n-2,0]
        PdP[:,n,0] = c *PdP[:,n-1,0] - Knm*PdP[:,n-2,0] 
        PdP[1,n,0] -= s*PdP[0,n-1,0]
        #for m in range(1, min(n + 1, mmax + 1)):
        #    S[n, m] = S[n, m - 1] * \
        #      np.sqrt((n - m + 1)*(int(m == 1) + 1.)/(n + m))
        #    if n == m:
        #        PdP[:, n, n]  = s * PdP[:, n - 1, n - 1]
        #        PdP[1, n, n] += c * PdP[0, n - 1, n - 1]
        #    else:
        #        Knm = ((n - 1)**2 - m**2) / ((2*n - 1)*(2*n - 3))
        #        PdP[:, n, m]  = c * PdP[:, n -1, m] - Knm*PdP[:, n - 2, m]
        #        PdP[1, n, m] -= s * PdP[0, n - 1, m]
        m_arr = np.arange(1,min(n + 1, mmax + 1))
        
        S[n,m_arr] = S[n,m_arr-1] * np.sqrt((n - m_arr + 1)*((m_arr==1)+1)/(n+m_arr))
        Knm = ((n - 1)**2 - m_arr**2) / ((2*n - 1)*(2*n - 3))
        PdP[:, n, m_arr]  = c * PdP[:, n -1, m_arr] - Knm*PdP[:, n - 2, m_arr]
        PdP[1, n, m_arr] -= s * PdP[0, n - 1, m_arr]

        #overwrite for n==m
        PdP[:, n, n]  = s * PdP[:, n - 1, n - 1]
        PdP[1, n, n] += c * PdP[0, n - 1, n - 1]
        
 

    if schmidtnormalize:
        # now apply Schmidt normalization
        PdP *= S
    
    out=PdP
    


#@jit#(cache=True) #uint8(uint32,uint8,uint8)
def get_l_maxmin(arr_len,lmax=0,lmin=0,suppress=False):
  """
  Semi-brute force attempt to get a reasonable value of lmax 
  (maximum degree) based on array length

  Idea based on the fact that the array length will never exceed 
  ``lmax**2``, but ``lmax`` will never be larger than 
  ``array length/2``. The algorithm then favours solutions with lower 
  ``lmax`` where the array length does not correspond to a unique
  ``(lmax,lmin)`` pair. ``lmax`` and/or ``lmin`` may be set. If no pair
  is found, an error is raised.``suppress=True`` suppresses logger 
  output.
  """

  if lmax<0:
    lmax=0
  if lmin<0:
    lmin=0

  #return input if all specified
  if lmax and lmin:
    if arr_len==(lmax*(lmax+2)-(lmin-1)*(lmin+1)):
      return lmax,lmin
    else:
      if not suppress:
        aux.logger.error("(lmax,lmin) pair are unsuitable.")
      raise ValueError
  #find lmin given lmax
  if lmax:
    for lmin in range(1,lmax+1):
      if arr_len==(lmax*(lmax+2)-(lmin-1)*(lmin+1)):
        return lmax,lmin
    if not suppress:
      aux.logger.error("Could not find suitable lmin given lmax")
    raise ValueError
    
  lmax_est=int(np.sqrt(arr_len))
  
  #find lmax given lmin
  if lmin:
    for lmax in [lmax_est,lmax_est+1]:
      if arr_len==(lmax*(lmax+2)-(lmin-1)*(lmin+1)):
        return lmax,lmin
    for lmax in range(int(arr_len/2+1)):
      if arr_len==(lmax*(lmax+2)-(lmin-1)*(lmin+1)):
        return lmax,lmin
    if not suppress:
      aux.logger.error("Could not find suitable lmin given lmax")
    raise ValueError
    
  #find both lmax and lmin
  for lmax in [lmax_est,lmax_est+1]:
    for lmin in range(lmax):
      if arr_len==(lmax*(lmax+2)-(lmin-1)*(lmin+1)):
        return lmax,lmin
  for lmax in range(int(arr_len/2+1)):
    for lmin in range(1,lmax+1):
      if arr_len==(lmax*(lmax+2)-(lmin-1)*(lmin+1)):
        return lmax,lmin
   
  if not suppress:
    aux.logger.error("Could not find a suitable (lmax,lmin) pair")
  raise ValueError


def read_shc(shc_fn,cols='all'):
  """
  Read values of gaussian coefficients (g,h) from column(s) in file.

  File should be ascii file obeying the SHC format.

  Parameters
  ----------
  shc_fn : str
    Path of input SHC ascii file
  cols : list_like
    List of columns to read from file. This should correspond to the
    columns the different times values coefficients will be 
    read from. In a standard SHC file the first two columns (0 and 1)
    correspond to the degree (l) and order (m) of the harmonic and
    should not be included in `cols`. As such the default value 
    ``cols='all'`` corresponds to ``cols=range(2,2+N_times)``, where 
    ``N_times`` is the number of time snapshots in the file.  

  Returns
  -------
  Tuple
    Tuple with following values at given indices:

      0. numpy.ndarray of gaussian coefficients with such that 
        ``myarray[0]`` gives all coefficients at the first time point,
        given that there are multiple time snapshots. Otherwise 
        ``array[0]`` will only contain the first coefficient.
      1. spline order `k` as an integer used to reconstruct model from 
        time snapshots.
      2. number of columns as an integer.
      3. time of the temporal snapshots (in fractional years in the 
          standard SHC format) as 1D `numpy.ndarray`.

  Notes
  -----
  Missing data values marked as NaN are currently not handled.
  
  """
  with open(shc_fn) as f:
    headerlen=0
    header_fin=False
    for line in f:
      if header_fin:#finished reading header
        times=np.empty(N_times)
        c=0
        for t in line.split():
          times[c]=float(t)
          c+=1
        break
      else:
        if line.startswith('#'):
          headerlen+=1
        else:
          header_fin=True
          N_min,N_max,N_times,spline_order,N_step = \
            (int(v) for v in line.split()[:5])
  if cols=='all':
    cols=range(2,2+N_times)
  else:
    c_tmp=[c-2 for c in cols]
    times=times[c_tmp]
    N_times = len(times)
  gh=np.loadtxt(shc_fn,skiprows=headerlen+2,usecols=cols,unpack=True)
  if len(gh.shape)==1:
    gh=np.expand_dims(gh,0)
  
  # currently not passing on N_min,N_max,N_step
  return gh,spline_order,len(cols),times

def get_ghi(l,m,l_low=1):
    if abs(m)>l:
        return -1
    return l**2 + 2*abs(m) - 2 - (m>0) - l_low**2

def _Bnec_core(gh,lmax,lmin,lat_rad,lon_rad,spline_order,r,dB,n_times,lmin_file=1):
  """Compute magnetic field in North-East-Center reference system"""
  X,Y,Z=0,1,2
  llo,lla=len(lon_rad),len(lat_rad)
  lo_a_cs=np.empty((lmax+1,2,llo))
  for m in range(lmax+1):
    lo_a_cs[m,0]=np.cos(m*lon_rad)
    lo_a_cs[m,1]=np.sin(m*lon_rad)
  
  Bnec=np.zeros((n_times,3,lla,llo))
  la_arr = np.arange(lla,dtype=np.uint16)
  t_arr = np.arange(n_times,dtype=np.uint16)
  l_arr = np.arange(lmin,lmax+1,dtype=np.uint16)
  PdP = np.zeros((2,lmax+1,lmax+1),dtype=np.float32)
  #m arr goes from 1 to l
  
  for la in range(lla):#lat_rad loop
    #_th = np.array((np.pi/2-lat_rad[la],),dtype=np.float32)
    #_sn = np.array((True,),dtype=np.dtype(bool))
    _get_legendre(PdP,np.pi/2-lat_rad[la],True,PdP)
    P,dP = PdP
    for t in range(n_times):
      gh_i=(lmin-1)*(lmin+1) - (lmin_file-1)*(lmin_file+1)
    
      for l in range(lmin,lmax+1):#degree loop
        rpow=r**(-(l+2))
        #m=0 #=> sin(mX)=0,cos(mX)=1 out1=X*m=0
        Bnec[t,X,la]+=dP[l,0]*rpow*(gh[t,gh_i])
        #Bnec[t,Y,la]+=0
        Bnec[t,Z,la]-= P[l,0]*rpow*(gh[t,gh_i])*(l+1)
        gh_i+=1
        for m in range(1,l+1): #order loop
          Bnec[t,X,la]+=dP[l,m]*rpow\
              *(gh[t,gh_i]*lo_a_cs[m,0] + gh[t,gh_i+1]*lo_a_cs[m,1])  
        
          Bnec[t,Y,la]+= (P[l,m]/np.cos(lat_rad[la])*rpow*m\
              *(gh[t,gh_i]*lo_a_cs[m,1] - gh[t,gh_i+1]*lo_a_cs[m,0])) 
          
          Bnec[t,Z,la]-= P[l,m]*rpow*(l+1)\
              *(gh[t,gh_i]*lo_a_cs[m,0] + gh[t,gh_i+1]*lo_a_cs[m,1])  
          gh_i+=2
    #gh_i = (lmin-1)*(lmin+1) - (lmin_file-1)*(lmin_file+1)
    #m_arr = np.arange(-lmin,lmax+1,dtype=np.int16)
    #gh_i = get_ghi(l_arr,m_arr,lmin_file)
    #Bdiff1 = np.where(gh_i>0,gh[:,gh_i]*lo_a_cs .... ,np.zeros(gh_i.shape))
    

    #rpow = r**(-(l_arr+2))
    #Bnec[:,X,la]+=PdP[1,l,0]*rpow*(gh[:, ])
    #Bnec[:,Z,la]-= P[0,l,0]*rpow*(gh[:,gh_i])*(l_arr+1)

    #Bnec[:,X,la]+=PdP[1,l_arr,l_arr]*rpow\
    #          *(gh[t,gh_i]*lo_a_cs[m,0] + gh[t,gh_i+1]*lo_a_cs[m,1])
        
    #m=0


   
  return Bnec    

#optimizable?
def _B_nec_spl(inarr,times,outval,der,k,b,c,ext=2):
  """Spline for all values of B_nec inside a latitude loop"""
  #a: dimension of B (or dB) ie: Bx,By,Bz
  out=np.empty((len(outval),3,b,c))
  for ai in range(3):
      for bi in range(b):
        for ci in range(c):
          tck=splrep(times,inarr[:,ai,bi,ci],k=k)
          #ext=2 raise err, ext=0 extrapolates
          out[:,ai,bi,ci]=splev(outval,tck,der=der,ext=ext)
        
  return out


def get_Bnec(shc_fn,latitude,longitude,cols='all',lmax=-1,lmin=-1,lmin_file=1,r=1,h=0,t_out=[],k=-1,dB=False,ext=2):
  """
  Compute magnetic field components in NEC-frame from SHC ascii file.

  Get computation of the magnetic field components or its derivative
  for given latitude and longitude in the North-East-Center reference
  system given gaussian spherical harmonics coefficients file.

  Parameters
  ----------
  shc_fn : str
    Path of input SHC ascii file
  latitude : array_like
    latitude values to evaluate magnetic field at
  longitude : array_like
    longitude values to evaluate magnetic field at
  cols : list_like, optional
    List of columns to read from file. This should correspond to the
    columns the different times values coefficients will be 
    read from. In a standard SHC file the first two columns (0 and 1)
    correspond to the degree (``l``) and order (``m``) of the 
    harmonic and should not be included in `cols`. As such the default
    value ``cols='all'`` corresponds to ``cols=range(2,2+N_times)``, 
    where `N_times` is the number of time snapshots in the file.  
  lmax, lmin : int, optional
    Maximum and minimum  degree of harmonics :math:`l_{max}` 
    (:math:`l_{min}`). If non-positive, suitable values will be set 
    based on the number of coefficients (default ``-1``).
  lmin_file : int, optional
    Lowest value of degree in SHC file (default ``1``).
  r : float, optional
    Fractional radius at which to evaluate magnetic field. This is the
    radius divided by the reference radius 6371.2 km
    (see also `h`)(default ``1``).
  h : float, optional
    Height(in km) above reference radius 6371.2 km at which to evaluate
    magnetic field. A non-zero value of `h` will overwrite any 
    value of `r` (default ``0``).
  t_out : datetime.datetime, scalar or datetime/scalar list, optional
    Times at which to evaluate magnetic field. Float values should 
    correspond to fractional years. If left empty, times will be taken 
    from the SHC file (default ``[]``).
  k : int
    Spline order for temporal interpolation. If not set, spline_order 
    will be taken from SHC file. If `k` is greater than or equal to
     number of temporal snapshots, `k` will be reduced (default ``-1``,
     implying set by SHC file). 
  dB : bool
    Return interpolated magnetic field derivative dB/dt instead of 
    magnetic field (default ``False``).
  ext : { 0 | 1 | 2 | 3 }
    If interpolation is performed, determine the behaviour when 
    extrapolating: 

        0 : return extrapolated value
        1 : return 0
        2 : raise error (default)
        3 : return boundary value

  Returns
  -------
  numpy.ndarray with shape ``(N_times,3,latitude,longtitude)``

  See also
  --------
  get_l_maxmin, read_gh_shc
  
  """
  gh,spline_order,n_times,times = read_shc(shc_fn,cols)
  if k>0:
    spline_order=k
  
  if spline_order>=n_times and (t_out or dB):
    #if interpolation is intended: 
    #fit interpolation degree to what is possible
    spline_order=min(spline_order,n_times-1)
    aux.logger.info(
      'as n_times set is {} > spline order, spline order is set to: {}'
      .format(n_times,spline_order))  
  aux.logger.debug(
    'gh has shape: {}, n_times(chosen number of columns):'
    .format(np.shape(gh))+\
    ' {}, spline_order: {}'.format(n_times,spline_order))
  
  r0=6371.2
  if h:
    r=(h+r0)/r0
  if hasattr(gh[0][0],'__iter__'):
    arr_len = len(gh[0][0])
  else:
    arr_len = len(gh[0])

  lmax,lmin=get_l_maxmin(arr_len)#use whole array
  
  if lmax>0 or lmin>0:
    for length in range(arr_len,0,-1):
      try:
        lmax,lmin=get_l_maxmin(arr_len,lmax,lmin,True)
        if lmin>=lmin_file:
          break
      except ValueError:
        pass

  else:
    lmax,lmin=get_l_maxmin(arr_len,lmin=lmin_file)
  aux.logger.debug(
    'lmax and lmin set to be lmax={},lmin={}'.format(lmax,lmin))
  
  
  d2r=np.pi/180
  
  latitude =np.asarray( latitude)
  longitude=np.asarray(longitude)
  
  Bnec=_Bnec_core(gh,lmax,lmin,latitude*d2r,longitude*d2r,spline_order,
          r,dB,n_times,lmin_file)
  
  if not (t_out or dB):
    return Bnec
  else:
      t_out=aux._to_dec_year_list(t_out)
      llo,lla=len(longitude),len(latitude)
  
      if (not t_out) and dB:
        t_out=times
      return _B_nec_spl(Bnec,times,t_out,int(dB),spline_order,lla,llo)



if aux._importable('numba'):
    import numba
    _get_legendre = numba.guvectorize(
        ['void(float32[:,:,:],float32[:],boolean[:],float32[:,:,:])'],
        '(m,n,p),(),()->(m,n,p)')(_get_legendre)
    #get_l_maxmin = numba.jit(cache=True)(get_l_maxmin)
    #_Bnec_core = numba.jit(cache=True)(_Bnec_core)
    #_B_nec_spl = numba.jit(cache=True)(_B_nec_spl)
else:
    _get_legendre = np.vectorize(_get_legendre)
    pass

#@jit #(cache=True,nogil=True,nopython=True) 
#(array(float64, 2d, C) x 2)(uint8,uint8,float64,bool)

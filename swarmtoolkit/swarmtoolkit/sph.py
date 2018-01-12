#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import splrep,splev

from . import auxiliary 

__all__=[ 'get_Bnec',
          'get_Bparameter',
          'get_index',
          'get_l_maxmin',
          'read_shc']


def _get_legendre(theta, nmax, mmax, schmidtnormalize=True):
    """Legendre function and its first derivative
    
      Thetha are colatitudes in radians
        The functions are computed iteratively, using an algorithm from 
        "Spacecraft Attitude Determination and Control" by James Richard Wertz
        (http://books.google.no/books?id=GtzzpUN8VEoC&lpg=PP1&pg=PA781#v=onepage)
    """
    
    if hasattr(theta,'__len__'):
        theta = theta[0]
    
    PdP = np.zeros((2,nmax+1,mmax+1))
    S = np.zeros((nmax+1,mmax+1))

    #shorthand notation
    c=np.cos(theta)
    s=np.sin(theta)
    
    
    #Initialization
    S[0, 0] = 1.
    PdP[0, 0, 0] = 1.
    #dP[0,0]= 0
    #ddP[0,0]=0

    #n=1 m=0
    PdP[:,1,0]  = c * PdP[:,0,0]
    PdP[1,1,0] -= s * PdP[0,0,0]
    S[1,0] = S[0,0]
    
    #PdP[2,1,0] -= 2*s*PdP[1,0,0] + c*PdP[0,0,0]
    
    #n=1,m=1
    if mmax:
      PdP[:, 1, 1]  = s * PdP[:,0,0]
      PdP[1, 1, 1] += c * PdP[0,0,0] 
      #PdP[2,1,1] += 2*c*PdP[1,0,0] - s*PdP[0,0,0] 
      S[1, 1] = S[1,0]
    
    m_a = np.arange(1, mmax + 1)
    
                
    for n in range(2, nmax +1):
        #m=0
        S[n, 0] = S[n - 1, 0] * (2*n - 1)/n
        Knm = ((n - 1)**2) / ((2*n - 1)*(2*n - 3))
        #P[n,0]  = c * P[n-1,0] - Knm*P[n-2,0]
        #dP[n,0] = c * dP[n-1,0] - s * P[n-1,0] - Knm * dP[n-2,0]
        PdP[:,n,0]  = c*PdP[:,n-1,0] - Knm*PdP[:,n-2,0] 
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
        m = m_a[:min(n,mmax)]
        #S[n,m] = np.where(n-m+1>0,S[n,0]*(np.sqrt((n - m + 1)*((m==1)+1)/(n+m))**m),0)
        S[n,m] = S[n,0] * np.cumprod(np.sqrt((n - m + 1)*((m==1)+1)/(n+m)))
        
        #for i in range(len(m)):
        #  S[n,m[i]] = S[n,m[i]-1] * _[i] 
        #print(S[n,1:m[-1]+1].shape,_.shape,m.shape)
        #=_
        Knm = ((n - 1)**2 - m**2) / ((2*n - 1)*(2*n - 3))
        PdP[:, n, m]  = c * PdP[:, n -1, m] - Knm*PdP[:, n - 2, m]
        PdP[1, n, m] -= s * PdP[0, n - 1, m]
        
        #overwrite for n==m
        if n in m:
          PdP[:, n, n]  = s * PdP[:, n - 1, n - 1]
          PdP[1, n, n] += c * PdP[0, n - 1, n - 1]
        #print(PdP[0])
    
    if schmidtnormalize:
        # now apply Schmidt normalization
        PdP *= S
    
    return PdP
 

def _get_legendre_grad(theta, nmax, mmax, schmidtnormalize=True):
  """Legendre function and its first and second derivatives"""
  if hasattr(theta,'__len__'):
    theta = theta[0]
    
  PdP = np.zeros((3,nmax+1,mmax+1))
  S = np.zeros((nmax+1,mmax+1))

  #shorthand notation
  c=np.cos(theta)
  s=np.sin(theta)
  
  
  #Initialization
  S[0, 0] = 1.
  PdP[0, 0, 0] = 1.
  #dP[0,0]= 0
  #ddP[0,0]=0

  #n=1 m=0
  PdP[:,1,0]  = c * PdP[:,0,0]
  PdP[1,1,0] -= s * PdP[0,0,0]
  PdP[2,1,0] -= 2*s*PdP[1,0,0] + c*PdP[0,0,0]
  
  S[1,0] = S[0,0]
  
  
  #n=1,m=1
  if mmax:
    PdP[:, 1, 1]  = s * PdP[:,0,0]
    PdP[1, 1, 1] += c * PdP[0,0,0] 
    PdP[2,1,1] += 2*c*PdP[1,0,0] - s*PdP[0,0,0] 
    S[1, 1] = S[1,0]
  
  m_a = np.arange(0, mmax + 1)
  
              
  for n in range(2, nmax +1):
      #m=0
      S[n, 0] = S[n - 1, 0] * (2*n - 1)/n
      #Knm = ((n - 1)**2) / ((2*n - 1)*(2*n - 3))
      
      #PdP[:,n,0]  = c*PdP[:,n-1,0] - Knm*PdP[:,n-2,0] 
      #PdP[1,n,0] -= s*PdP[0,n-1,0]
      #PdP[2,n,0] -= 2*s*PdP[1,0,0] + c*PdP[0,0,0]
      
      m = m_a[:min(n,mmax)]

      S[n,m] = S[n,0] * np.cumprod(np.sqrt((n-m+1)*((m==1)+1)/(n+m)))
      Knm = ((n-1)**2 - m**2) / ((2*n - 1)*(2*n - 3))

      PdP[:, n, m]  = c * PdP[:,n-1,m] - Knm*PdP[:,n-2,m]
      PdP[1, n, m] -= s * PdP[0,n-1,m]
      PdP[2, n, m] -= 2*s*PdP[1,0,0] + c*PdP[0,0,0]
      
      #overwrite for n==m
      if n in m:
        PdP[:, n, n]  = s * PdP[:,n-1,n-1]
        PdP[1, n, n] += c * PdP[0,n-1,n-1]
        PdP[2, n, n] += 2*c*PdP[1,n-1,n-1] - s*PdP[0,n-1,n-1]

  if schmidtnormalize:
      # now apply Schmidt normalization
      PdP *= S
  
  return PdP


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

  Assumes maximum order value is the same as maximum degree
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
        auxiliary.logger.error("(lmax,lmin) pair are unsuitable.")
      raise ValueError
  #find lmin given lmax
  if lmax:
    for lmin in range(1,lmax+1):
      if arr_len==(lmax*(lmax+2)-(lmin-1)*(lmin+1)):
        return lmax,lmin
    if not suppress:
      auxiliary.logger.error("Could not find suitable lmin given lmax")
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
      auxiliary.logger.error("Could not find suitable lmin given lmax")
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
    auxiliary.logger.error("Could not find a suitable (lmax,lmin) pair")
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


def get_index(l,m,lmin=1,mmax=-1):
    """
    Get index of a Gauss coefficient in an array, from degree and order

    Order `m` needs to be less or equal to degree `l`.
  
    Parameters
    ----------
    l : int
      Degree of coefficient
    m : int
      Order of coefficient
    lmin : int, optional
      Lower bound of `l` in array (default 1).
    mmax : int, optional
      Upper bound of `m` in array. If `mmax` less than 0, then it is 
      assumed to only be restricted by `l` (default -1).

    Returns
    -------
    int
      Index of coefficent. If invalid input parameters are provided,
      -1 will be returned.

    Notes
    -----
    Index is given by:
    .. math: 

      l^2 - l_{\text{min}}^2 + 2|m|    , if m=0.
      l^2 - l_{\text{min}}^2 + 2|m| - 1, if m>0. 

    If mmax is set an additional term 
    :math:`-(l-m_\text{max}+1)(l-m_\text{max}+2)` is added.

    
    """
    idx = l**2 - lmin**2 + 2*abs(m) - (m>0)
    
    if abs(m)>l: 
      return -1
    elif mmax<0:
      return idx
    else:
      if m>mmax: 
        return -1
      t = l-mmax-1
      return idx + (t>0)*t*(t+1)
      


def _Bnec_core(gh,lmax,lmin,clat_rad,lon_rad,spline_order,r,dB,n_times,lmin_file=1,mmax=-1,isrc=True):
  """Compute magnetic field in North-East-Center reference system"""
  X,Y,Z = 0,1,2#N,E,C indices
  llo,lla = len(lon_rad),len(clat_rad)
  if mmax<0:
    mmax = lmax

  lo_a_cs = np.empty((mmax+1,2,llo))
  for m in range(mmax+1):
    lo_a_cs[m,0] = np.cos(m*lon_rad)
    lo_a_cs[m,1] = np.sin(m*lon_rad)
  
  
  Bnec = np.zeros((n_times,3,lla,llo))
  #m arr goes from 1 to l
  if isrc:
    s = 1
    l0 = -2
  else:
    s = -1
    l0 = -1

  m_arr = np.arange(mmax+1)
  l_arr = np.arange(lmin,lmax+1)

  if not isinstance(r,np.ndarray):
    r = np.full(lla,r)
  elif len(r)!=lla:
    auxiliary.logger.error('Unexpected shape of r or h')
    return


  u=np.newaxis
  for la in range(lla):#lat_rad loop
    s_inv = 1/np.sin(clat_rad[la])
    P,dP = _get_legendre(clat_rad[la],lmax,mmax,True)
    for t in range(n_times):
      ghcs = _compute_gh_cs(gh,lo_a_cs,lmax,lmin,t,llo,lmin_file,mmax)
      
      rpow = r[la]**(l0-s*(l_arr))


      for li,l in enumerate(range(lmin,lmax+1)):#degree loop
        m = m_arr[:min(l+1,mmax+1)]
        
        Bnec[t,X,la] += rpow[li]*(dP[l,m,u]*ghcs[li,m,0]).sum(0) 
        Bnec[t,Y,la] += rpow[li]* (P[l,m,u]*s_inv*m[:,u]*ghcs[li,m,1]).sum(0)
        Bnec[t,Z,la] -= rpow[li]* (P[l,m,u]*(l+isrc)*s*ghcs[li,m,0]).sum(0) 
         
  return Bnec 


def _Bnec_core_grad(gh,lmax,lmin,clat_rad,lon_rad,spline_order,r,dB,n_times,lmin_file=1,mmax=-1,isrc=True,grad='E'):
  """Compute gradient of magnetic field in North-East-Center reference system"""
  X,Y,Z = 0,1,2#N,E,C indices
  llo,lla = len(lon_rad),len(clat_rad)
  if mmax<0:
    mmax = lmax

  lo_a_cs = np.empty((mmax+1,2,llo))
  for m in range(mmax+1):
    lo_a_cs[m,0]=np.cos(m*lon_rad)
    lo_a_cs[m,1]=np.sin(m*lon_rad)
  
  Bnec = np.zeros((n_times,3,lla,llo))
  #m arr goes from 1 to l
  
  R_inv=1/(r*6371.2)
  

  if isrc:
    s = 1
    l0 = -2
  else:
    s = -1
    l0 = -1

  m_arr = np.arange(mmax+1)
  l_arr = np.arange(lmin,lmax+1)
  
  if not isinstance(r,np.ndarray):
    r = np.full(lla,r)
  elif len(r)!=lla:
    auxiliary.logger.error('Unexpected shape of r or h')
    return



  u=np.newaxis
  
  if grad=='N':
    for la in range(lla):#lat_rad loop
      s_inv = 1/np.sin(clat_rad[la])
      t_inv = 1/np.tan(clat_rad[la])
      PdP = _get_legendre_grad(clat_rad[la],lmax,mmax,True)
      for t in range(n_times):
        ghcs = _compute_gh_cs(gh,lo_a_cs,lmax,lmin,t,llo,lmin_file,mmax)
      
        rpow = r[la]**(l0-s*(l_arr))

        for li,l in enumerate(range(lmin,lmax+1)):#degree loop
          m = m_arr[:min(l+1,mmax+1)]
        
          Bnec[t,X,la] += rpow[li]*(ghcs[li,m,0]\
              *(PdP[2,l,m,u] - s*(l+isrc)*PdP[0,l,m,u])).sum(0)
        
          Bnec[t,Y,la] -= rpow[li]*(ghcs[li,m,1]*m[:,u]*s_inv\
              *(PdP[1,l,m,u]-t_inv*PdP[0,l,m,u])).sum(0)
          
          Bnec[t,Z,la] += s*rpow[li]*(ghcs[li,m,0,u]*PdP[1,l,m,u]*(l+isrc+s)).sum(0)
          
  elif grad=='E':
    for la in range(lla):#lat_rad loop
      s_inv = 1/np.sin(clat_rad[la])
      t_inv = 1/np.tan(clat_rad[la])
      coth_inv = 1/np.tanh(clat_rad[la])
      PdP = _get_legendre(clat_rad[la],lmax,mmax,True)
      for t in range(n_times):
        ghcs = _compute_gh_cs(gh,lo_a_cs,lmax,lmin,t,llo,lmin_file,mmax)
      
        rpow = r[la]**(l0-s*(l_arr))

        for l in range(lmin,lmax+1):#degree loop
          
          m = m_arr[:min(l+1,mmax+1)]
          m_s = m*s_inv

          Bnec[t,X,la] -= rpow[li]*(ghcs[li,m,1]*m_s[:,u]\
              *(PdP[1,l,m,u]-t_inv*PdP[0,l,m,u])).sum(0)
        
          Bnec[t,Y,la] += rpow[li]*(ghcs[li,m,0]\
              *(PdP[1,l,m,u]*coth_inv-(m_s[:,u]**2\
              +s*(l+isrc))*PdP[0,l,m,u])).sum(0)
          
          Bnec[t,Z,la] -= rpow[li]*(ghcs[li,m,1]*m_s[:,u]\
              *(l+1+isrc)*PdP[0,l,m,u]).sum(0)

  elif grad=='C':
    for la in range(lla):#lat_rad loop
      s_inv = 1/np.sin(clat_rad[la])
      t_inv = 1/np.tan(clat_rad[la])
      PdP = _get_legendre(clat_rad[la],lmax,mmax,True)
      for t in range(n_times):
        ghcs = _compute_gh_cs(gh,lo_a_cs,lmax,lmin,t,llo,lmin_file,mmax)
      
        rpow = r[la]**(l0-s*(l_arr))

        for l in range(lmin,lmax+1):#degree loop

          Bnec[t,X,la] += s*rpow[li]*(ghcs[li,m,0]*PdP[1,l,m,u]*(l+isrc+s)).sum(0)

          Bnec[t,Y,la] -= rpow*(ghcs[li,m,1]*m[:,u]*s_inv\
              *(l+1+isrc)*PdP[0,l,m,u])

          
          Bnec[t,Z,la] += rpow*(ghcs[li,m,0]\
              *(l+isrc)*(l+isrc+s)*PdP[0,l,m,u]).sum(0)

  return Bnec 


def _compute_gh_cs(gh,cs_lon,lmax,lmin,t_idx,llo,lmin_file=1,mmax=-1):
  
  ghcs = np.zeros((lmax+1-lmin,mmax+1,2,llo))
  
  gh_i = (lmin-1)*(lmin+1) - (lmin_file-1)*(lmin_file+1)
  for li,l in enumerate(range(lmin,lmax+1)):
    ghcs[li,0,0] = gh[t_idx,gh_i]
    gh_i += 1
    for m in range(1,min(l+1,mmax+1)):
      ghcs[li,m,0] = gh[t_idx,gh_i  ]*cs_lon[m,0]\
                   + gh[t_idx,gh_i+1]*cs_lon[m,1]
      ghcs[li,m,1] = gh[t_idx,gh_i  ]*cs_lon[m,1]\
                   - gh[t_idx,gh_i+1]*cs_lon[m,0]
      gh_i += 2
  return ghcs


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


def get_Bparameter(B,outp='FDI'):
  """
  Get Intensity, declination or inclination of magnetic field 

  Parameters
  ----------
  B : np.ndarray of floats
    numpy.ndarray of magnetic vector components. Components are 
    assumed to be B_N = B[0], B_E = B[1], B_C=B[2]. Expected
    shape of array is ``(Ntimes,3,dim1,dim2)`` or ``(3,?,?)``
    where ``?`` signifies optional dimensions.
  outp : str or list, optional
    Output parameter. Must be ``'F'``(intensity), ``'D'``
    (declination) or ``'I'``(Inclination) (default ``'FDI'``).

  Returns
  -------
  numpy.ndarray
    array with shape ``(N_times,len(outp),latitude,longtitude)``

  """
  Bsh = B.shape
  if (len(Bsh)==4 and Bsh[1]==3):
    out = np.empty((Bsh[0],len(outp),Bsh[2],Bsh[3]))
    for i in range(Bsh[0]):
      out[i] = get_Bparameter(B[i],outp)
    return out[i]
  elif Bsh[0]==3:
    shape = [len(outp)] + list(Bsh[1:])
    out = np.empty(shape)
  else:
    auxiliary.logger.error("Unexpected shape of input {}".format(Bsh))
    return

  X,Y,Z = 0,1,2
  for p_i,p in enumerate(outp):
    if p.upper() == 'F':
      out[p_i] = np.sqrt((B**2).sum(axis=0))
    elif p.upper() == 'D':
      out[p_i] = np.arctan(B[Y]/B[X])
    elif p.upper() == 'I':
      out[p_i] = np.arctan(B[Z]/np.sqrt(B[X]**2 + B[Y]**2))
    else:
      out[p_i] = 0
  return out


def get_Bnec(shc_fn_dict,latitude,longitude,cols='all',lmax=-1,lmin=-1,lmin_file=1,r=1,h=0,t_out=[],k=-1,dB=False,gradient='',source='internal',ext=2):
  """
  Compute magnetic field components in NEC-frame from SHC ascii file.

  Get computation of the magnetic field components or its derivative
  for given latitude and longitude in the North-East-Center reference
  system given gaussian spherical harmonics coefficients file.

  Parameters
  ----------
  shc_fn_dict : str
    Path of input SHC ascii file `or` dictionary containing fields 
    ``coeff`` (gauss coefficents), ``t`` (time[float]) and 
    ``lm`` (degree and order pairs) and optionally ``k`` (spline
    order, default 3, will be overwritten if ``k`` is specified as
    keyword argument).
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
  lmax,lmin : int, optional
    Maximum and minimum  degree of harmonics :math:`l_{max}` 
    (:math:`l_{min}`). If non-positive, suitable values will be set 
    based on the number of coefficients (default ``-1``).
  lmin_file : int, optional
    Lowest value of degree in SHC file (default ``1``).
  r : float or np.ndarray, optional
    Fractional radius at which to evaluate magnetic field. This is the
    radius divided by the reference radius 6371.2 km. If `r` is 
    provided as an array, it must have the same shape as `latitude`
    (see also `h`)(default ``1``).
  h : float or np.ndarray, optional
    Height(in km) above reference radius 6371.2 km at which to evaluate
    magnetic field. A non-zero value of `h` will overwrite any 
    value of `r`. If `h` is provided as an array, it must have the
    same shape as `latitude` (default ``0``).
  t_out : datetime.datetime, scalar or datetime/scalar list, optional
    Times at which to evaluate magnetic field. Float values should 
    correspond to fractional years. If left empty, times will be taken 
    from the SHC file (default ``[]``).
  k : int, optional
    Spline order for temporal interpolation. If not set, spline_order 
    will be taken from SHC file. If `k` is greater than or equal to
    number of temporal snapshots, `k` will be reduced (default ``-1``,
    implying set by SHC file). 
  dB : bool
    Return interpolated magnetic field derivative dB/dt instead of 
    magnetic field (default ``False``).
  gradient : { '' | 'X' | 'Y' | 'Z'}
    Compute the gradient of one of the magnetic field components. 
    Note that solutions are numerically unstable at the poles. This
    will stack with `dB`, such that if ``bool=True`` and 
    ``gradient='Y'``, the time derivatives of the gradient of the 
    east component will be computed. ``''`` implies no gradient
    (default ``''``).
  source : { 'internal' | 'external' }
    determine whether to interpret the SHC file as containing the
    internal or the external Gaussian coefficients 
    (default ``'internal'``).
  ext : { 0 | 1 | 2 | 3 }
    If interpolation is performed, determine the behaviour when 
    extrapolating: 

        0 : return extrapolated value
        1 : return 0
        2 : raise error (default)
        3 : return boundary value

  Returns
  -------
  numpy.ndarray
    array with shape ``(N_times,3,latitude,longtitude)``

  See also
  --------
  get_l_maxmin, read_shc, read_mma
  
  """
  if isinstance(shc_fn_dict,dict):
    gh = shc_fn_dict['coeff']
    times = shc_fn_dict['t']
    lm = shc_fn_dict['lm']
    lmin_file = min(lm[:,0])
    spline_order = 3
    n_times=len(times)
  gh,spline_order,n_times,times = read_shc(shc_fn_dict,cols)
  if k>0:
    spline_order=k
  
  if spline_order>=n_times and (t_out or dB):
    #if interpolation is intended: 
    #fit interpolation degree to what is possible
    spline_order=min(spline_order,n_times-1)
    auxiliary.logger.info(
      'as n_times set is {} > spline order, spline order is set to: {}'
      .format(n_times,spline_order))  
  auxiliary.logger.debug(
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
  auxiliary.logger.debug(
    'lmax and lmin set to be lmax={},lmin={}'.format(lmax,lmin))
  
  
  d2r=np.pi/180
  
  latitude =np.asarray( latitude)
  longitude=np.asarray(longitude)
  
  Bnec=_Bnec_core(gh,lmax,lmin,.5*np.pi-latitude*d2r,longitude*d2r,spline_order,
          r,dB,n_times,lmin_file)
  
  if not (t_out or dB):
    return Bnec
  else:
      t_out=auxiliary._to_dec_year_list(t_out)
      llo,lla=len(longitude),len(latitude)
  
      if (not t_out) and dB:
        t_out=times
      return _B_nec_spl(Bnec,times,t_out,int(dB),spline_order,lla,llo)


def mauersberger_lowes_spec(gh,r=1,lmax=-1,lmin=1):
    """The Mauersberger–Lowes spectrum"""
    ratio=1/r
    if lmax<1:
        lmax,lmin=get_l_maxmin(len(gh_vals))
    R_l=np.empty(lmax+1-lmin)
    gh_idx=0
    for l in range(lmin,lmax+1):
        gh_idx_n=gh_idx+2*l+1
        g_sq=np.sum(gh_vals[gh_idx:gh_idx_n]**2)
        gh_idx=gh_idx_n
        
        R_l[l-lmin] = (l+1)*ratio**(2*l+4)*g_sq
    return R_l


def degree_correlation(gh1,gh2,lmax=-1,lmin=1):
    """Correlation per spherical harmonic degree between two models 1 and 2"""
    if lmax<1:
        lmax1,lmin1=get_l_maxmin(len(gh1))
        lmax2,lmin2=get_l_maxmin(len(gh2))
        lmax = min(lmax1,lmax2)
        lmin = min(lmin1,lmin2)
    c12=np.empty(lmax+1-lmin)
    i=0
    for l in range(lmin,lmax+1):
        #m=0
        g12 = gh1[i]*gh2[i]
        g11 = gh1[i]**2
        g22 = gh2[i]**2
        i+=1
        for m in range(1,l+1):
            g12 += gh1[i]*gh2[i]
            g11 += gh1[i]**2
            g22 += gh2[i]**2
            i += 2
        c12[l-lmin] = g12/np.sqrt(g11*g22)
    return c12


def mean_sq_vdiff(gh1,gh2,r=1,lmax=-1,lmin=1):
    """Mean square vector field difference per spherical harmonic degree"""
    ratio=1/r
    if lmax<1:
        lmax1,lmin1=get_l_maxmin(len(gh1))
        lmax2,lmin2=get_l_maxmin(len(gh2))
        lmax = min(lmax1,lmax2)
        lmin = min(lmin1,lmin2)
    r12 = np.empty(lmax+1-lmin)
    i=0
    for l in range(lmin,lmax+1):
        #m=0
        dg = (gh1[i]-gh2[i])**2
        i += 1
        for m in range(1,l+1):
            dg += (gh1[i]-gh2[i])**2
            i+=2
        r12[l-lmin] = (l+1)*ratio**(2*l+4)*dg
    return r12


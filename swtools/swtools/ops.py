#!/usr/bin/python
# -*- coding: utf-8 -*-

import datetime as dt

import numpy as np
import spacepy
#in shift_param:
#import numexpr as ne
#from iminuit import Minuit 
from scipy.interpolate import InterpolatedUnivariateSpline,RectSphereBivariateSpline 

from . import aux


__all__=[ 'align_param',
          'shift_param',
          'where_overlap',
          'fourier_transform',
          'cyclic2rising',
          'rising2cyclic',
          'interpolate2d_sphere',
          'where_diff']


def align_param(p1,p2,t1,t2,k=3,align_to=False): 
  """
  .. _align_param:
  
    Interpolate parameters such that time values overlap.

  Given parameter arrays :math:`p_1(t_1)` and :math:`p_2(t_2)` with 
  their corresponding time arrays one can downsample the most 
  frequently sampled array, and align the arrays wrt. time (using
  interpolation) such that only one time array :math:`t` is required 
  for :math:`p_1` and :math:`p_2`. Only overlapping temporal regions 
  will be utilized. Upsampling may be forced using the `align_to` 
  argument. 

  Parameters
  ----------
  p1 : 1D numpy.ndarray
    First parameter to align. 
  p2 : 1D numpy.ndarray
    Second parameter to align.
  t1 : 1D numpy.ndarray of datetime.datetime objects
    time values of `p1`. Should have same length as `p1`. If 
    ``align_to=True``, `t1` will set the output frequency. 
  t2 : 1D numpy.ndarray of datetime.datetime objects
    time values of `p2`. Should have same length as `p2`.
  k : int, optional
    Degree of the smoothing spline. Must be ``1 <= k <= 5``
    (default ``3``).
  align_to : bool, optional
    Align `p2` to `p1` independent of respective frequency, instead of
    downsampling to lowest frequency of the two. This may be useful for
    aligning multiple parameters to a specific frequency, for 
    upsampling, or for handling datasets where one of the parameters is
    not uniformly sampled (default ``False``).

  Returns
  -------
  tuple 
    (p1',p2',t) where `p1'` and `p2'` are sampled at `t` instead of at 
    `t1` or `t2`. p1',p2' and t are `numpy.ndarray's`. As only the 
    temporal overlap is utilized, the temporal span of the array will
    in general be less than or equal to the smallest temporal span of
    the two.

  Raises
  ------ 
  ValueError
  
      - if respective p,t pairs are not of same length.
      - if length of t array is less than spline order.
      - if time arrays are not ordered ascending.
      - if unable to interpolate arrays.

  Notes
  -----
  To be able to interpolate, the spline order must be less than the
  total number of time steps. This funtion assumes uniform sampling,
  and the sampling times are currently solely determined by the step
  length between the first two timesteps. If one of the parameters is
  uniformly sampled while the other is not, or one of the arrays
  contains non-finite numbers, ``align_to=True`` may be used with
  the finite, uniformly sampled parameter as `p1`.
  
  See also 
  --------
  plot_align
  
  """
  t_dt=False
  k=int(k)
  if len(t1)<=k or len(t2)<=k:
    aux.logger.error(
      'Input arrays have length {} and {}, need at least a length of {}'
      .format(len(t1),len(t2),1+k))
    raise ValueError
  if len(p1)!=len(t1) or len(p2)!=len(t2):
    aux.logger.error(
      'Incompatible lengths of parameter to time array: p1:{}!=t1:{} or p2:{}!=t2:{}'
      .format(len(p1),len(t1),len(p2),len(t2)) )
    raise ValueError
  if isinstance(t1[0],dt.datetime):
    t_dt=True
  
  #compare to find which has lowest frequency
  d1=t1[1]-t1[0]
  d2=t2[1]-t2[0]
  try:
    d1d=d1.days
    d2d=d2.days
  except AttributeError:
    d1d=d1
    d2d=d2
  #could be done in a more complete way using sum, 
  #np.roll and bool check for values <=0
  #but then one would have to take into account seconds as well
  if d1d<0 or d2d<0:
      aux.logger.error(
        't1 or t2 object is not ordered ascending. Aborting align_param')
      raise ValueError
      
    
  if abs(aux._from_timedelta((d1-d2)))<1e-5:
    #approx equally sampled, only need to shift and crop array
    nansum1=np.sum(p1!=p1)
    nansum2=np.sum(p2!=p2)
    if nansum1>=nansum2:
      p_lf,t_lf=p1,t1
      p_hf,t_hf=p2,t2
      lf_eq_1=True
    else:
      p_hf,t_hf=p1,t1
      p_lf,t_lf=p2,t2
      lf_eq_1=False
    
    aux.logger.debug(
      'Approximately equally sampled. Sampling diff in us: {}'
      .format(abs(aux._from_timedelta((d1-d2),1e6))))
  else:
    if aux._from_timedelta(d1)<aux._from_timedelta(d2):
      #if d1 higher frequency:
      p_hf,t_hf=p1,t1
      p_lf,t_lf=p2,t2
      lf_eq_1=False
      aux.logger.debug(
        'p1 more frequent than p2. Diff in us: {}'
        .format(abs(aux._from_timedelta((d1-d2),1e6))))
    else:
      p_hf,t_hf=p2,t2
      p_lf,t_lf=p1,t1
      lf_eq_1=True
      aux.logger.debug(
        'p2 more frequent than p1. Diff in us: {}'
        .format(abs(aux._from_timedelta((d1-d2),1e6))))
  if align_to:
    p_lf,t_lf=p1,t1
    p_hf,t_hf=p2,t2
    lf_eq_1=True
    aux.logger.debug('p2 being aligned to p1')
  
  
  lh,ll=len(t_hf),len(t_lf)
      
  #cropping
  #index of first overlap:
  lo_edge_lf=np.argmax(t_hf[0]<=t_lf)
  lo_edge_hf=np.argmax(t_lf[0]<=t_hf)
  #index+1 of last overlap:
  hi_edge_lf=ll-np.argmax(t_hf[-1]>=t_lf[::-1])
  hi_edge_hf=lh-np.argmax(t_lf[-1]>=t_hf[::-1])
  
  #extend hi-freq part to avoid boundary effects
  if hi_edge_hf<lh-k:
    p_hf=p_hf[:hi_edge_hf+k]
    t_hf=t_hf[:hi_edge_hf+k]
  if lo_edge_hf>k:
    p_hf=p_hf[lo_edge_hf-k:]
    t_hf=t_hf[lo_edge_hf-k:]
  
  #crop low-freq
  t_lf=t_lf[lo_edge_lf:hi_edge_lf]
  p_lf=p_lf[lo_edge_lf:hi_edge_lf]
  
  #interpolate to find values of p_hf in range [t_lf[0],min(t_lf[-1],t_hf[-1])]
  if not np.array_equal(t_lf,t_hf):
    t0=t_hf[0]
    if t_dt:#if datetime: turn to float
      t_sec_hf=aux._to_sec_v(t_hf-t0)
      t_sec_lf=aux._to_sec_v(t_lf-t0) 
    else:
      t_sec_hf=t_hf
      t_sec_lf=t_lf
      
    f_hf=InterpolatedUnivariateSpline(t_sec_hf,p_hf,k=k,ext=2)
    try:
      p_hf=f_hf(t_sec_lf)
    except ValueError:
      aux.logger.error('Unable to interpolate. Check that arrays have'+\
      ' finite values and have some overlap in time. Aborting...')
      raise
  #return in same order they came in:
  if lf_eq_1:
    return p_lf,p_hf,t_lf
  else:
    return p_hf,p_lf,t_lf


def _shift_param(p1,p2,t1,t2,delta_t):
    """return values of p1 and p2 shifted by delta_t"""
    
    """
    #delta_t measured in s
    #output preserves len(x1)=len(x2) for x in [t,p]
    #assume t are numpy arrays of datetime objects and p are numpy arrays 
    #of float

    #do not assume equal length of t or p
    #assume equal steplength in t
    #assume t to be monotonous rising?
    
    #dt is how much to move t1 relative to t2
    """
    dt_us=aux._from_timedelta(delta_t,1e6)
    delta_t_dt = aux._to_timedelta(dt_us,'microseconds')
    step=aux._from_timedelta(t1[1]-t1[0],1e6)#step size
    
    #step correction 
    step_c=(dt_us%step)
    step_c_dt=aux._to_timedelta(step_c,'microseconds')
    len1,len2=len(t1),len(t2)
    
    if (t1[-1]+delta_t_dt)<t2[0] or (t1[0]+delta_t_dt)>t2[-1]:
        #will not work due to no overlap
        aux.logger.warning(
          'No overlap between the two time series for given shift: {} seconds'
          .format(dt_us/1e6))
        return 

    #find first and last overlap after shift:
    
    #index of first overlap on t1
    lo_edge1=np.argmax(t1+delta_t_dt>=t2[0]) 
    
    #index+1 of last overlap on t1
    hi_edge1=len1-np.argmax(t1[::-1]+delta_t_dt<=t2[-1]) 
    
    #index of first overlap on t2
    lo_edge2=np.argmax(t2>=t1[0]+delta_t_dt) 
    
    #index+1 of last overlap on t2
    hi_edge2=len2-np.argmax(t2[::-1]<=t1[-1]+delta_t_dt) 
    
    t1+=delta_t_dt
    
    return p1[lo_edge1:hi_edge1],p2[lo_edge2:hi_edge2],t1[lo_edge1:hi_edge1],t2[lo_edge2:hi_edge2]
 

def shift_param(p1,p2,t1,t2,delta_t=0,dt_lim=(-20,20),v=1,spline_points=1e7,eval_width=None,k=3,auto=False,useminos=True,imincall=1e4,bins=1e3,return_delta=False,show=False,ext=2):
    """
    .. _shift_param:

    Return values of `p1` and `p2` shifted by `delta_t`.

    Shift a parameter :math:`p_1(t_1)` wrt. a second parameter 
    :math:`p_2(t_2)` by a time step :math:`\Delta t`. The shift can
    also be done automatically to find best fit by using a minimizer
    based on 'SEAL Minuit'(`iminuit`) with interpolated values.

    Parameters
    ----------
    p1 : 1D numpy.ndarray
      Parameter to be shifted by `delta_t`
    p2 : 1D numpy.ndarray
      parameter to shift `p1` with respect to.
    t1 : 1D numpy.ndarray of datetime.datetime objects
      time values of `p1`. Should have same length as `p1`. 
    t2 : 1D numpy.ndarray of datetime.datetime objects
      time values of `p2`. Should have same length as `p2`.
    delta_t : int,float, optional
      time shift in seconds to shift `p1` by. If used together with the
      argument `auto=True`, this value will be used as a first guess to
      the best fit. It can then be set to `None` if `dt_lim` is set.
      A middle value will then be assumed (default ``0``).
    dt_lim : int/float list_like of length 2 or int/float.
      Maximum and minimum value of `delta_t` in seconds allowed for 
      minimizing algorithm. If `dt_lim` is a number, symmetry round
      `delta_t` will be assumed, eg ``[delta_t-dt_lim,delta_t+dt_lim]``.
      `int` or `float` must be non-negative. If ``dt_lim = None`` it 
      will be set to ``dt_lim=(delta_t - ((1-eval_ratio)/2)*abs(delta_t),
      delta_t + ((1-eval_ratio)/2)*abs(delta_t))``, where 
      ``eval_ratio=eval_width/len(p1)`` (default ``(-20,20)``). 
    v : int, optional
      Verbosity level of function. ``0 <= v <= 2`` (default ``1``).
    spline_points : int, optional
      Number of points used to make a spline fit of p1 with. Number
      will be reduced if p1 has fewer points. Float values will be 
      truncated (default ``1e7``).
    eval_width : int, optional
      Number of points in time to compare `p1` and `p2` values,
      centered around the value of ``t1+delta_t``. Number will be
      reduced by increasing span of dt_lim to accommodate for all
      possible values of `delta_t`. If set to ``eval_width=None`` a
      width corresponding to 60% of the length of p2 will be used 
      (default ``None``).
    k : int, optional
      Degree of the smoothing spline. Must be ``1 <= k <= 5``.
    auto : bool, optional
      Use minimizer to find best fit for `delta_t` (default ``False``).
    useminos : bool
      If ``auto=True``,run 
      `minos <http://iminuit.readthedocs.org/en/latest/api.html#iminuit.Minuit.minos>`_ 
      (default ``True``)
    imincall : int
      If ``auto=True``, number of calls to migrad/minos. Float values
      will be truncated (default ``1e4``)
    bins : int 
      If `auto=True`, number of bins for profile of solution space (if
      no solution is found from initial `delta_t`, divide dt_lim into
      `bins`, and find best solution out of these). Also applicable for
      when visualizing profile using ``show=True``. Float values will
      be truncated (default ``1e3``).
    return_delta : bool
      return `delta_t` as output (default ``False``).
    show : False
      show solution profile in a plot (see iminuit `draw_profile 
      <http://iminuit.readthedocs.org/en/latest/api.html#iminuit.Minuit.draw_profile>`_ 
      )(default ``False``).
    ext : int
      handling of values outside interpolation region:
      
      - extrapolation = 0
      - set to zero = 1 
      - raise error = 2
      - set to constant(equal to boundary) = 3

    Returns
    -------
    a tuple of numpy.ndarray's ``(p1,p2,t1+delta_t,t2)`` are returned.
    Only values with temporal overlap are returned. Output will be of
    equal length. If `return_delta=True`, a tuple 
    ``(p1,p2,t1+delta_t,t2,delta_t)`` will be returned, with delta_t as
    float.

    Raises
    ------
      ValueError

        - if length's are incompatible
        - if `eval_width`>length of p2
        - if neither `delta_t` nor `dt_lim` are provided.
        - if ``delta_t=None`` and `dt_lim` is a number.
        - if `dt_lim` is negative
      
      IndexError
        if `dt_lim` has length less than 2.

    Notes
    -----
    This function assumes uniform sampling rate, and may not give 
    desired results if this is not the case. As minimizing functions
    can be non-trivial, some tweaking of arguments may be necessary
    to get optimal results.
    
    See also
    --------
    align_param, where_overlap
    
    """
    import numexpr as ne
    from iminuit import Minuit 


    t2=t2.copy()
    t1=t1.copy()
    if not auto:
      if isinstance(delta_t,(int,float,dt.timedelta)):
        return _shift_param(p1,p2,t1,t2,delta_t)
      else:
        raise ValueError("Invalid timeshift delta_t provided")
    
    spline_points=int(spline_points)
    imincall=int(imincall)
    bins=int(bins)

    len1,len2=len(t1),len(t2)
    if len(p1)!=len(t1) or len(p2)!=len(t2):
      aux.logger.error(
        'incompatible lengths of parameter to time array:'+\
        ' p1:{}!=t1:{} or p2:{}!=t2:{}'
        .format(len(p1),len(t1),len(p2),len(t2)) )
      raise ValueError

    is_dt_dt=False 
    if isinstance(delta_t,dt.timedelta):
      delta_t=delta_t.total_seconds()
      is_dt_dt=True
    
    no_dt=False
    if delta_t==None:
      no_dt=True

    if eval_width!=None:
      if eval_width<=0 or eval_width>len2:
        aux.logger.error(
          "eval_width must be a positive integer less"+\
          " than the length of p2, you provided {}.".format(eval_width))
        raise ValueError
      eval_ratio=eval_width/len1
      if eval_ratio<0.1:
        low_ratio=True
    else:
      eval_width=int(0.6*len1)
      eval_ratio=eval_width/len1

    if hasattr(dt_lim,'__iter__'):
      if len(dt_lim)<2:
        aux.logger.error("dt_lim needs to be of length 2")
        raise IndexError
      for i in range(2):
        if dt_lim[i]==None:
          if no_dt:
            aux.logger.error(
              "If no delta_t is provided dt_lim tuple must be provided") 
            raise ValueError
        if isinstance(dt_lim[i],dt.timedelta):
          dt_lim[i]=dt_lim[i].total_seconds()

      dt_low,dt_high=dt_lim[:2]
      if not no_dt:
        if delta_t<dt_low or delta_t>dt_high:
          dt_low=delta_t - ((1-eval_ratio)/2)*abs(delta_t)
          dt_high=delta_t + ((1-eval_ratio)/2)*abs(delta_t)
          if v>0:
            aux.logger.info(
              "dt_lim has been changed from "+\
              "{} to {} due to specified delta_t being outside range."
              .format(dt_lim,(dt_low,dt_high)))
      dt_lim=(dt_low,dt_high)
    else:#assume number: max deviation from delta_t symmetrically
      if dt_lim!=None:
        if dt_lim<=0:
          aux.logger.error(
            "Float value dt_lim:{} out of bounds.".format(dt_lim)+\
            " Needs to be positive or tuple of numbers or timedelta objects.")
          raise ValueError
        dt_lim=(delta_t-dt_lim,delta_t+dt_lim)
      else:
        if no_dt:
          aux.logger.error(
            "If no delta_t is provided dt_lim tuple must be provided") 
          raise ValueError
        dt_lim=(delta_t - ((1-eval_ratio)/2)*abs(delta_t),
                delta_t + ((1-eval_ratio)/2)*abs(delta_t))
    step=(t1[1]-t1[0]).total_seconds()#one time step in seconds
   
    
    def check_eval_width(spline_points,dt_lim,eval_width,eval_ratio,tol=1e-3):
      #subset of p1 for splining
      if len1>spline_points:
        l_spl=len(p1[(l1-spline_points)//2:(l1+spline_points)//2])
      else:
        l_spl=len1
      if no_dt:
        #for slicing purposes define delta_t
        delta_t_=np.mean(dt_lim[:2])
      else:
        delta_t_=delta_t
      edge_dt_low =abs(dt_lim[1]-delta_t_)
      edge_dt_high=abs(dt_lim[0]-delta_t_)
    
      #dt_sec values converted to steps(rounded up):
      edge_dt_low =int(edge_dt_low //step + bool(abs(edge_dt_low %step)>tol))
      edge_dt_high=int(edge_dt_high//step + bool(abs(edge_dt_high%step)>tol))

      aux.logger.debug(
        "Step size is {} seconds. Edges need {}".format(step,edge_dt_low)+\
        "+{} steps to accommodate dt_lim values".format(edge_dt_high))
    
      is_valid=True
      if len2<(eval_width+edge_dt_low+edge_dt_high): 
        eval_width=len2-(edge_dt_low+edge_dt_high)
        eval_ratio=eval_width/len1
        aux.logger.debug(
          "eval_width adjusted to {} due to edge requirements"
          .format(eval_width))
        if eval_width<1:
          aux.logger.error(
            '\n\t'.join((
              'too few evaluation points. Consider reducing'+\
              ' spline_points or the span of dt_lim.'),
              'Number of  spline points: {}'.format(l_spl)+\
              'dt_lim: {}'.format(dt_lim)))
          is_valid=False
          if aux._is_interactive():
            print("Insert new values"+\
            " (If none specified, default values will be chosen")
            
            l_spl_tmp=input("spline points[{}]: ".format(l_spl))
            dt_lim0_tmp=input("dt_lim lower[{}]: ".format(dt_lim[0]))
            dt_lim1_tmp=input("dt_lim upper[{}]: ".format(dt_lim[1]))
            try:
              if l_spl_tmp.strip():
                l_spl = int(l_spl_tmp)
              if dt_lim0_tmp.strip():
                dt_lim[0]=float(dt_lim0_tmp)
              if dt_lim1_tmp.strip():
                dt_lim[1]=float(dt_lim1_tmp)
              if not (l_spl_tmp.strip() or \
                  dt_lim0_tmp.strip() or dt_lim1_tmp.strip()):
                raise ValueError
            except KeyboardInterrupt:
                aux.logger.error("KeyboardInterrupt - aborting...")
                raise 
            except Exception:
              aux.logger.error(
                "No input variables or incorrect input variables")
              raise
          else:
            raise ValueError(
              "too few evaluation points."+\
              " Consider reducing spline_points or the span of dt_lim") 
        if eval_width<10 and v:
          aux.logger.warning(
            'Only {} points used to evaluate goodness of fit.'
            .format(eval_width))
      
      return l_spl,dt_lim,edge_dt_low,edge_dt_high,eval_width,eval_ratio,is_valid 
    
    check_vars=spline_points,dt_lim,eval_width,eval_ratio,False
    while True:
      check_vars=check_eval_width(*check_vars[:-1])
      if check_vars[-1]:
        l_spl       = check_vars[0]
        dt_lim      = check_vars[1]
        edge_dt_low = check_vars[2]
        edge_dt_high= check_vars[3]
        eval_width  = check_vars[4]
        eval_ratio  = check_vars[5]
        
        delta_t=(dt_lim[0]+dt_lim[1])/2
        break


    p1_p=p1[(len1-l_spl)//2:(len1+l_spl)//2]
    t1_p=t1[(len1-l_spl)//2:(len1+l_spl)//2]
    aux.logger.debug(
      "number of spline points changed from '{}' to '{}'"
      .format(spline_points,l_spl))
    
    t0=t1_p[0]#arbitrarily chosen reference time
    t_sec1=aux._to_sec_v(t1_p-t0)
    p1_spl_f=InterpolatedUnivariateSpline(t_sec1,p1_p,k=k,ext=ext)
    #ext: 0=extrapolate;1=0;2=valueerror,3=const
   
    
    use_width=eval_width+edge_dt_low+edge_dt_high 
    eval_start=len2//2-(use_width//2 - edge_dt_low) -1
    eval_end  =len2//2+(use_width//2 - edge_dt_high) #-1
    if eval_start<0:
      eval_start=0
    t_sec2=aux._to_sec_v(t2[eval_start:eval_end]-t0) 
   
    fixing_vars = (
      ('Length of array splined',l_spl),
      ('Length of eval array',len(t_sec2)),
      ('eval_width',eval_width),('eval_ratio',eval_ratio),
      ('dt_lim',dt_lim),
      ('low_dt buf. len',edge_dt_low),
      ('upp_dt buf. len',edge_dt_high),
      ('initial delta_t guess',delta_t),
      ('dt set',not no_dt),
      ('eval_start index',eval_start),
      ('eval_end index+1',eval_end)
      )
    
    aux.logger.debug(
      'some variables related to the fitting listed:\n\t'+'\n\t'
      .join(('{:20s}:{:20s}'.format(fixing_vars[i][0],str(fixing_vars[i][1]))
       for i in range(len(fixing_vars)))))
    
    last_vars=[delta_t,None]
    
    def p_chisq_r(dt_candidate):
      p1_fit=p1_spl_f(t_sec2-dt_candidate)
      p2_slice=p2[eval_start:eval_end]
      chisq=ne.evaluate('sum((p2_slice-p1_fit)**2)')
      last_vars[:]=dt_candidate,chisq
      return chisq
    
    
    m = Minuit(p_chisq_r,
        dt_candidate=delta_t,
        limit_dt_candidate=dt_lim,
        print_level=bool(v>=2),
        error_dt_candidate=aux._from_timedelta(t1[1]-t1[0])/10,
        errordef=1)
    
    MAXTRIES=100
    tries=0
    def minuit_fail_warning():
      aux.logger.warning(
        "Unsuccessfil run of minimization algorithm, last recorded variables"+\
        " are delta_t: {:.4f}, chisq: {}.".format(*last_vars)+\
        " Aborted function 'shift_param'\nCheck that limits are appropriate,"+\
        " or adjust the ext parameter(currently ext={},".format(ext)+\
        " extrapolation=0,zero=1,raise error=2,constant=3)"+\
        " to handle values outside specified region")
    try:
      mout=m.migrad(ncall=imincall)
      while mout[0]['is_above_max_edm'] and not \
        mout[0]['has_reached_call_limit']:
      
        tries+=1
        mout=m.migrad(ncall=imincall)
        aux.logger.debug(
          "migrad did not converge; retrying. last vars:"+\
          " delta_t={:.4f} chisq:{}".format(*last_vars))
        if tries==MAXTRIES:
          raise ValueError('Maximum number of tries reached')
    except ValueError: 
      minuit_fail_warning()
      return

    if no_dt:
      try:
        prof=m.mnprofile('dt_candidate',bins=bins,bound=dt_lim)
      except ValueError:
        minuit_fail_warning()
      delta_t=prof[0][np.argmin(prof[1])]

    if useminos:
      try:
        mout=m.minos(maxcall=imincall)['dt_candidate']
      except ValueError:
        minuit_fail_warning()
        return
      delta_t=mout['min']
      is_valid=mout['is_valid']
    else:
      delta_t=mout[1][0].value
      is_valid=mout[0]['is_valid']
    

    if not is_valid:
        aux.logger.info(
          "Validity of solution is questionable; iminuit output dump: {}"
          .format(mout)) 
    elif mout['at_lower_limit']:
      aux.logger.info(
        "delta_t converged to solution near lower limit, consider rerunning"+\
        " with new limits")
    elif mout['at_upper_limit']:
      aux.logger.info(
        "delta_t converged to solution near upper limit, consider rerunning"+\
        " with new limits")
    if v:
      aux.logger.info('output delta_t: {}'.format(delta_t))
    
    if show:
      try:
        mout=m.draw_profile('dt_candidate',bins=bins,bound=dt_lim)
      except ValueError:
        minuit_fail_warning()
    if delta_t==None:
      return
    if return_delta:
      p1,p2,t1,t2=_shift_param(p1,p2,t1,t2,delta_t)
      if is_dt_dt:
        delta_t=dt.timedelta(seconds=delta_t)
      return p1,p2,t1,t2,delta_t 
    else:
      return _shift_param(p1,p2,t1,t2,delta_t)


def where_overlap(t1,t2,delta_t=0):
  """
  Find overlap between two datetime arrays, where one array may be 
  shifted by `delta_t`.

  This is essentially a convenience function to access 
  ``spacepy.toolbox.tOverlap(t1+delta_t,t2,presort=True)``.
  """
  try:
    if not (isinstance(t1[0],dt.datetime) and isinstance(t2[0],dt.datetime)):
      raise ValueError
  except Exception:
    aux.logger.error("{} was not supplied datetime arrays correctly"
      .format(where_overlap.__name__))
    raise 

  return spacepy.toolbox.tOverlap(t1+aux._to_timedelta(delta_t,'seconds'),t2,
            presort=True)


def fourier_transform(param,dt_t,norm=None):
  """
  Fourier transformation of 1d array with corresponding dt information.

  Parameters
  ----------
  param : array_like
    input parameter
  dt_t : float, datetime.timedelta or numpy.ndarray of datetime.datetime
    sample time or array of parameter sampling times.
  norm : {None,'ortho'} 
    None : no scaling
    'ortho': direct fourier transform scaled by ``1/sqrt(n)``, with 
    ``n`` being the length of `param` (default ``None``).
  
  Returns
  -------
  tuple
    `numpy.ndarray` of fourier transform of param, and `numpy.ndarray` of 
    corresponding frequencies.

  Raises
  ------
    TypeError
      if `dt_t` is array and content is not `datetime.datetime` objects
  
  Notes
  -----
  Requires uniform temporal sampling.
  
  """

  if isinstance(dt_t,np.ndarray):
    if isinstance(dt_t[0],dt.datetime):
      dt_t=dt_t[1]-dt_t[0]
    else:
      aux.logger.error('Content of dt_t array is not datetime objects')
      raise TypeError
  if isinstance(dt_t,dt.timedelta):
    dt_t=dt_t.total_seconds()
  out_param=np.fft.fft(param,norm=norm)
  out_freq=np.fft.fftfreq(len(param),dt_t)
  return out_param,out_freq


def cyclic2rising(a,lim=[-90,90]):
    """
    Return array with monotonic rising values, from array with cyclic 
    values (requires first indices to be rising, and assumes approx. 
    equidistant points).

    Parameters
    ----------
    a : array_like
      array of smooth cyclic values to be made monotonic rising.
    lim : list
      list of extremal(min,max) values within which `a` is cyclic
      (default ``[-90,90]``).

    Returns
    -------
    `numpy.ndarray`
      array of monotonic rising values
    """
    la=len(a)
    out=np.copy(a)
    maxdiff=lim[1]-lim[0]
    maxdiff2=maxdiff*2
  
    def adiff(a,b):
      return np.abs(a-b)
    
    #indices of local extremal points
    
    maxidx=np.where((a>np.roll(a,1)) &  (a>np.roll(a,-1)))[0]
    minidx=np.where((a<np.roll(a,1)) &  (a<np.roll(a,-1)))[0]
    
    #remove endpoint extremal points 
    try:
      if maxidx[0]==0:
        maxidx=maxidx[1:]
      if maxidx[-1]==(la-1):
        maxidx=maxidx[:-1]
    except IndexError:
      pass#lma=0
    try:
      if minidx[0]==0:
        minidx=minidx[1:]
      if minidx[-1]==(la-1):
        minidx=minidx[:-1]
    except IndexError:
      pass#lmi=0
    
    lma=len(maxidx)
    lmi=len(minidx)
    if lma+lmi==0:#already not cyclic
      return a


    #determine true point *before* extremal point 
    for i in range(lma):
        mi=maxidx[i]
        #if mi<la-1:
        if adiff(a[mi],a[mi+1])>adiff(a[mi],a[mi-1]):
                maxidx[i]=mi-1
    for i in range(lmi-1):
        mi=minidx[i]

        if adiff(a[mi],a[mi+1])>adiff(a[mi],a[mi-1]):
            minidx[i]=mi-1
  
    #add appropriate amount to make function monotonously rising
    if maxidx[0]<minidx[0]:#starts with max
        for m in range(lmi):
            out[maxidx[m]+1:minidx[m]+1]=maxdiff-a[maxidx[m]+1:minidx[m]+1]
        if lma!=lmi:
            out[maxidx[-1]+1:]=maxdiff-a[maxidx[-1]+1:]
        for m in range(lmi):
            out[minidx[m]+1:]+=maxdiff2
    else:
        for m in range(lma-1):#starts with min
            out[minidx[m]+1:maxidx[m+1]+1]=maxdiff-a[minidx[m]+1:maxidx[m+1]+1]
        if lma!=lmi:
            out[minidx[-1]+1:maxidx[-1]+1]=maxdiff-a[minidx[-1]+1:maxidx[-1]+1]
        for m in range(lmi):
            out[minidx[m]+1:]+=maxdiff2
    
    return out


def rising2cyclic(a,lim=[-90,90]):
    """
    Returns an array of cyclic values between two extremal values
    (requires first indices to be rising)

    Parameters
    ----------
    a : array_like
      array of monotonic rising values to be made cyclic
    lim : list
      list of extremal(min,max) values to make array cyclic within
      (default ``[-90,90]``).

    Returns
    -------
    `numpy.ndarray`
      array of cyclic values
    
    """
    #assume starts by rising
    out=np.empty(len(a))
    maxdiff=lim[1]-lim[0]
    out[:]=a%(2*maxdiff)
    q23=np.where((out>=  lim[1]) & (out < (lim[1]+maxdiff)))[0]
    q4=np.where(out>=(lim[1]+maxdiff))[0]
    out[q23]=maxdiff-out[q23]
    out[q4]=out[q4]-2*maxdiff
    return out


def interpolate2d_sphere(lat_rad, lon_rad, param,**kwargs):
  """
  Interpolate on rectangular mesh on sphere using radians
  
  Convenience function to call `RectSphereBivariateSpline 
  <http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectSphereBivariateSpline.html>`_
  
  Parameters
  ----------
  lat_rad : array_like
    1-D array of latitude coordinates in strictly ascending order.
    Coordinates must be given in radians, and lie within ``(0, pi)``.
  lon_rad : array_like
    1-D array of longitude coordinates in strictly ascending order.
    Coordinates must be given in radians and lie within the interval
    ``(0, 2*pi)``.
  param : array_like
    2-D array of parameter with shape ``(lat_rad.size, lon_rad.size)``
  
  Returns
  -------
  `scipy.interpolate.RectSphereBivariateSpline`
    Spline function to be used for evaluation of interpolation

  Notes
  -----
  Keyword arguments passed on to `RectSphereBivariateSpline`_.
  """
  ##more general(slower) interpolation in two dimensions:
  #interp2d
  #return interp2d(x,y,z,**kwargs)
  return RectSphereBivariateSpline(lat_rad,lon_rad,param,**kwargs)


def where_diff(values,atol=None,rtol=None,pdiff=[75,25],axis=0,no_jump=False):
  """
  Get indices of values which are significantly different from the
  preceding values.

  Function to find discontinuities using absolute tolerance, relative
  tolerance and percentile differences over an array.

  Parameters
  ----------
  values : array_like
    input array to be evaluated
  atol : float, optional
    absolute tolerance such that where the difference between any value
    and its preceeding value is larger than `atol` will be flagged as a
    discontinuity. May be combined with `rtol` to only flag 
    intersection of `atol` and `rtol` (default ``None``).
  rtol : float, optional
    relative tolerance such that where the difference between any value
    and its preceeding value divided by its value is larger than `rtol`
    , it will be flagged as a discontinuity. May be combined with 
    `atol` to only flag intersection of `atol` and  `rtol` 
    (default ``None``). 
  pdiff : list of float of length 2, optional
    Two values between 0 and 100. The percentile difference such that
    where the difference between any value and its preceeding value is
    larger than the difference between the values of the two 
    percentiles of the data, it will be flagged as a discontinuity 
    (default ``[75,25]``).
  axis : int
    Axis in array over which to evaluate (default ``0``).
  no_jump : bool
    Flag continuities instead of discontinuities (default ``False``).

  Returns
  -------
  ndarray or tuple of `numpy.ndarrays`
    Indices of flagged values 

  """
  values=np.asarray(values)
  val_diff=np.abs((values-np.roll(values,1,axis=axis)))
  val_diff[0]=0#unphysical
  if atol and rtol:
    jump= val_diff > abs(atol)
    jump*= val_diff/values > abs(rtol)
    if no_jump:
      return np.where((jump+1)%2)
    return np.where(jump)

  elif atol:
    diff=abs(atol)
    
  elif rtol:
    val_diff/=abs(values)
    diff=rtol
  else: #(use percetile) sort perc difference between values
    perc = np.percentile(values,pdiff[1]),np.percentile(values,pdiff[0])
    diff=abs(perc[1]-perc[0])
  if no_jump:
    return np.where(val_diff <= diff)  
  else:
    return np.where(val_diff > diff)   


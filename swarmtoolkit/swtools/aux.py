#!/usr/bin/python
# -*- coding: utf-8 -*-

import importlib
import datetime as dt
import logging

import numpy as np

__all__ = [ 'debug_info']

logger=None

#currently only some products avaliable
PRODUCT_FTP_LOC={
  'IBI':( 'ftp://swarm-diss.eo.esa.int/Level2daily/Current/IBI/TMS/Sat_A',
          'ftp://swarm-diss.eo.esa.int/Level2daily/Current/IBI/TMS/Sat_B',
          'ftp://swarm-diss.eo.esa.int/Level2daily/Current/IBI/TMS/Sat_C'),
  'TEC':( 'ftp://swarm-diss.eo.esa.int/Level2daily/Current/TEC/TMS/Sat_A',
          'ftp://swarm-diss.eo.esa.int/Level2daily/Current/TEC/TMS/Sat_B',
          'ftp://swarm-diss.eo.esa.int/Level2daily/Current/TEC/TMS/Sat_C'),
  'FAC':( 'ftp://swarm-diss.eo.esa.int/Level2daily/Current/FAC/TMS/Sat_A',
          'ftp://swarm-diss.eo.esa.int/Level2daily/Current/FAC/TMS/Sat_AC',
          'ftp://swarm-diss.eo.esa.int/Level2daily/Current/FAC/TMS/Sat_B',
          'ftp://swarm-diss.eo.esa.int/Level2daily/Current/FAC/TMS/Sat_C'),
  'EEF':( 'ftp://swarm-diss.eo.esa.int/Level2daily/Current/EEF/TMS/Sat_A',
          'ftp://swarm-diss.eo.esa.int/Level2daily/Current/EEF/TMS/Sat_B'),

  'MAG':( 'ftp://swarm-diss.eo.esa.int/Level1b/Current/MAGx_LR/Sat_A',
          'ftp://swarm-diss.eo.esa.int/Level1b/Current/MAGx_LR/Sat_B',
          'ftp://swarm-diss.eo.esa.int/Level1b/Current/MAGx_LR/Sat_C'
          'ftp://swarm-diss.eo.esa.int/Level1b/Current/MAGx_HR/Sat_A',
          'ftp://swarm-diss.eo.esa.int/Level1b/Current/MAGx_HR/Sat_B',
          'ftp://swarm-diss.eo.esa.int/Level1b/Current/MAGx_HR/Sat_C'),
  'EFI':( 'ftp://swarm-diss.eo.esa.int/Level1b/Current/EFIx_PL/Sat_A'
          'ftp://swarm-diss.eo.esa.int/Level1b/Current/EFIx_PL/Sat_B',
          'ftp://swarm-diss.eo.esa.int/Level1b/Current/EFIx_PL/Sat_C')
}
PRODUCT_DIC={#only lower case keys
  #IBI
  'bubble_index':'IBI',
  'bubble_probability':'IBI',
  #TEC
  'absolute_stec':'TEC',
  'relative_stec':'TEC',
  #FAC
  'fac':'FAC',
  'irc':'FAC',
  #EEF
  'eef':'EEF',
  #ACCxPOD
  #'acc_pod':None,
  #ACCx_AE
  #'acc_aero_obs':None,
  #DNSxWND
  #'local_solar_time':None,

  #MAGx_LR,MAGx_HR 
  'f':'MAG',
  'b_vfm':'MAG',
  'b_nec':'MAG',

  #EFI
  'v_ion':'EFI',
  't_ion':'EFI',
  't_elec':'EFI',
  'n':'EFI',
  'e':'EFI'
}

#
DL_FTP_MSG="""insert 0 to abort, 
    use ',' between values to select multiple files/directories,
    use '-' to select a range. ',' and '-' may be combined.
    Note that multiple downloads will result in immediate download.
    Select file(s)/directory from one of the above:"""

EPOCH = dt.datetime(2000,1,1,0,0,0,0)
SECONDS_PER_DAY=86400
MILLISECONDS_PER_DAY = SECONDS_PER_DAY*1000 
    
#conversion
def _tolist(obj):
  """turn into list if object is dict or not a non-string iterable"""
  if isinstance(obj, str) or not hasattr(obj,'__iter__'):
    return [obj]
  elif isinstance(obj,dict):
    return list(obj.values())
  else: 
    return obj


def _single_item_list_open(obj):
  """open return list/dict value if length is 1"""
  if isinstance(obj,list):
    if len(obj)==1:
      return obj[0]
  if isinstance(obj,dict):
    if len(obj)==1:
     return _single_item_list_open(list(obj.values()))
  return obj

#datetime conversions/functions
def _set_period(start_t=None,end_t=None,duration=None):
  """return a list of start time and end time given input"""
  if start_t and end_t:
    return [_to_datetime(start_t),_to_datetime(end_t)]
  if start_t and duration:
    return [
      _to_datetime(start_t),_to_datetime(start_t)+_to_timedelta(duration)]
  if end_t and duration:
    return [_to_datetime(end_t)-_to_timedelta(duration),_to_datetime(end_t)]

  return


def _is_in_period(date,dates):
  """check if 'date' is on same date as or between dates given in 'dates'"""
  #note using datetime.date and not datetime.datetime for comparison
  date =date.date()
  date0=dates[0].date()
  date1=dates[1].date()

  if date>=date0 and date<=date1:
    return True
  return False


def _MJD2000_datetime(days,milliseconds): 
    """extract datetime from (days,milliseconds) since 1.1.2000 UTC """
    #requires number or numpy.ndarray 
    days+= milliseconds/MILLISECONDS_PER_DAY
    if isinstance(days,(int,float)):
      return EPOCH+dt.timedelta(days=days)
    else:
      out=np.full(len(days),EPOCH,dtype=object)
      for i in range(len(milliseconds)):
        out[i]+=dt.timedelta(days=days[i])

      return out


def _MJD2000sec_datetime(sec):
    """extract datetime from seconds since 1.1.2000 UTC """
    days=sec//SECONDS_PER_DAY 
    milliseconds = (sec%SECONDS_PER_DAY)*1000
    return _MJD2000_datetime(days,milliseconds)


def _str2dt(s):
    """turn string 'year month day hour minute second to datetime"""
    tlist=s.split() #split time by whitespace
    tlist[:-1]=list(map(lambda y:int(y),tlist[:-1]))
    return dt.datetime(*tlist[:4])+\
      dt.timedelta(seconds=(tlist[4]*60+float(tlist[5])))


_str2dt_v=np.vectorize(_str2dt)
_float_us_to_timedelta_v=np.vectorize(lambda x:dt.timedelta(microseconds=x))
_to_sec_v=np.vectorize(lambda t:t.total_seconds())

     
def _from_timedelta(d,factor=1):
  """turn timedelta to seconds multiplied by a factor"""
  if isinstance(d,dt.timedelta):
    d=d.total_seconds()
  return d*factor  


def _to_timedelta(dt_input,t_type='days'):
  """turn input to timedelta object of type t_type"""
  if isinstance(dt_input,dt.timedelta):
    return dt_input
  if isinstance(dt_input,(int,float)):#assume number of days
    if t_type=='days':
      return dt.timedelta(days=dt_input)
    elif t_type=='seconds':
      return dt.timedelta(seconds=dt_input)
    else:
      #t_type : microseconds
      return dt.timedelta(microseconds=dt_input)
  
  logger.error("timedelta input not understood: '{}' of type '{}' "
    .format(dt_input,type(dt_input)))
  if type(dt_input) in (dt.timedelta,str,int,float):
    raise ValueError
  raise TypeError


def _to_datetime(date_input):
  """turn input to datetime object"""
  #currently highest resolution for string is days
  if isinstance(date_input,dt.datetime):
    return date_input
  if isinstance(date_input,str):
    date_input=date_input.strip()
    strlen=len(date_input)
    if strlen == 10: #assume yyyy-mm-dd
      year = int(date_input[0:4])
      month= int(date_input[5:7])
      day  = int(date_input[8:10])
    elif strlen == 8:#assume yyyymmdd
      year = int(date_input[0:4])
      month= int(date_input[4:6])
      day  = int(date_input[6:8])
    elif strlen in [14,15]:#assume yyyymmddThhmmss or yyyymmddhhmmss
      return _to_datetime(date_input[:8])
    return dt.datetime(year,month,day)
  if isinstance(date_input,(int,float)): #assume fractional days since MJD2000 
    return _MJD2000_datetime(int(date_input),
          (date_input-int(date_input))*86400000)
  
  logger.error("date input not understood: '{}' of type '{}' "
    .format(date_input,type(date_input)))
  if type(date_input) in (dt.datetime,str,int,float):
    raise ValueError
  raise TypeError(
    "Function '{}' only accepts datetime.datetime,str,int and float"
    .format(_to_datetime.__name__))

def _to_dec_year_list(ti):
  """datetime object list to decimal years list"""
  tl=_tolist(ti)
  if len(tl)==0: 
    return tl
  elif isinstance(tl[0],dt.datetime):
    #no leap years or leap seconds are taken into consideration
    return [t.year + t.month/12 + (t.day + (t.hour + (t.minute + (t.second
              + t.microsecond/1e6)/60)/60)/24)/365 for t in tl]
  else:
    return tl

#coloring
def _CSI(n):
  """format for CSI codes"""
  return '\x1b['+str(n)+'m'


def _CSI2str(n,*args):
  """format string(s) with a given CSI code"""
  p=''
  for a in args:
    p+=str(a)
  return '{}{}{}'.format(_CSI(str(n)),p,_CSI(0))

##misc
def _is_interactive():
  """determines if python is run interacively"""
  import __main__
  return not hasattr(__main__,'__file__')


def _in2range(totlen,i=None,msg=''):
    """convert ","/"-"-separated numbers to a list"""
    if not i:
        i=input(msg)
    try:
      for c in i:
        if not c.isdigit() and c not in ',-':
            raise ValueError(
              "input value '{}' is not a digit,hyphen or comma.".format(c))
    
      csp=i.split(',',1)
      if len(csp)>1:
            if len(i.split('-'))>1:
                import itertools 
                if not csp[0] or not csp[1]:
                    raise ValueError('Invalid use of commas')
                
                #start with splitting commas,then hyphens
                return itertools.chain(
                _in2range(totlen,csp[0]), _in2range(totlen,csp[1])) 
            #if commas but no hyphens return integers
            return (int(j) for j in i.split(','))  
    except Exception:
      logger.error('Invalid input, aborting...')
      raise
    if (len(i.split('-'))==1):#if no hyphens: must be int
        try:
            i=int(i)
        except Exception:
            logger.error("Could not convert '{}' to integer.".format(i))
            raise
        if not hasattr(i,'__len__'):
            return [i]
        
    else: 
        i=i.split('-')
        i0,i1=0,0
        if i[0]:#check if empty 
            i0=int(i[0])
        if i[1]:
            i1=int(i[1])
        if i0>totlen or i1>totlen:
            logger.error(
              "Value out bound: '{}' or '{}' larger than '{}'"
              .format(i0,i1,totlen))
            raise IndexError
        if (i0 and i1):
            s=(i1-i0)//abs(i1-i0) #determine sign
            return range(i0,i1+s,s)
        elif i0:
            return range(i0,totlen+1)
        elif i1:
            return range(1,i1+1)
        else:
            return range(1,totlen+1)    


def _importable(module_str):
    """check whether module can be imported"""
    try:
        l = importlib.find_loader(module_str)
        found = l is not None

    except Exception:
        logger.error("Could not import module: '{}'".format(module_str))
        raise ImportError
    return found


def _set_sw_logger(use_color=True):
  """set up logger"""
  global logger
  if not logger:
    logger = logging.getLogger('swarmtoolkit-logger')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s\t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    if use_color:
      #critical,error: red, warning: yellow, debug: blue, all: boldfont, 
      logging.addLevelName( 
        logging.CRITICAL, 
        _CSI2str("31;1",
        logging.getLevelName(logging.CRITICAL)))
      
      logging.addLevelName( 
        logging.ERROR, 
        _CSI2str("31;1",
        logging.getLevelName(logging.ERROR)))
      
      logging.addLevelName( 
        logging.WARNING, 
        _CSI2str("33;1",
        logging.getLevelName(logging.WARNING)))
      
      logging.addLevelName( 
        logging.INFO, 
        _CSI2str("1",
        logging.getLevelName(logging.INFO)))
      
      logging.addLevelName( 
        logging.DEBUG, 
        _CSI2str("34;1",
        logging.getLevelName(logging.DEBUG)))


def _get_sw_logger():
  return logger


def debug_info(activate=1):
  """Set the verbosity level of the logger.

  Parameters
  ----------
  activate : int
    Possible values:
    - ``activate>0`` : logger set to DEBUG (default)
    - ``activate<0`` : logger set to CRITICAL (virtually no logging)
    - ``activate=0`` : logger set to INFO (nominal value)

  """
  if activate>0:
    logger.setLevel(logging.DEBUG)
  elif activate<0:
    logger.setLevel(logging.CRITICAL)
  else:
    logger.setLevel(logging.INFO)

class _Conditional_decorator:
    def __init__(self, dec, condition):
        self.decorator = dec
        self.condition = condition

    def __call__(self, func):
        if not self.condition:
            return func
        return self.decorator(func)

#!/usr/bin/python
# -*- coding: utf-8 -*-

import zipfile
import tempfile
import os
import glob
import datetime as dt

import ftputil
import ftputil.session
import ftplib
import numpy as np
import spacepy.pycdf as pycdf

from . import aux

__all__ = [ 'concatenate_values',
            'dl_ftp',
            'extract_parameter',
            'getCDFlist',
            'getCDFparamlist',
            'getCDFparams',
            'param_peek',
            'read_sp3',
            'read_EFI_prov_txt',
            'unzip_file',
            'Parameter']


def getCDFparams(src,*params,**kwargs):
  """
  .. _getCDFparams:
  
  Extract parameters from cdf input as a list.

  See `getCDFlist`_ for a list of keyword arguments not specified here.

  Parameters
  ----------
  src : str
    Path or url of filename(s) or filename directory used to search for
    cdf file. 
  params : str 
    Names of parameters to be extracted from `src`. If names are not
    known, parameters within cdf files may be accessed using 
    `getCDFparamlist`_. Multiple parameters should be given as 
    separate, comma-separated strings. If no parameters are provided,
    all parameters from the first cdf found in `src` will be used
  
  Keyword Arguments
  -----------------
    param0 : str
      specify case-insensitive parameter name to use for filtering of files. If none 
      specified first parameter in `params` is assumed. Will not be 
      used unless ``filter_param=True`` is specified. Only supports values included
      in the dictionary ``swtools.aux.PRODUCT_DIC``.

  Returns
  -------
  list 
    `Parameter`_'s as ordered in `params`. If only one parameter is
    specified, `Parameter`_ will not be contained within a list. 

  Notes
  -----
    Other keyword arguments are passed on to `getCDFlist`_, 
    `extract_parameter`_ and `dl_ftp`_ where applicable.
  
  See also
  --------
  getCDFlist, extract_parameter, dl_ftp
  
  """
  if 'param0' not in kwargs:
    if params:
      kwargs['param0']=params[0]
      aux.logger.debug("{} chosen as 'param0'".format(params[0]))
    else:
      kwargs['param0'] = None
  if 'filter_param' not in kwargs:
    kwargs['filter_param']=False
  kwargs['src']=src
  cdflist=getCDFlist(**kwargs)
  if not params:
    if 'verbose' not in kwargs:
      kwargs['verbose'] = False
    params = getCDFparamlist(cdflist[0],**kwargs)
  
  aux.logger.debug("getting cdf files from:\n\t{}"
    .format('\n\t'.join(aux._tolist(src))))
  p_list=[]
  if cdflist:
    for par in params:
      p_list.append(extract_parameter(cdflist,par,**kwargs))

  return aux._single_item_list_open(p_list)


def unzip_file(input_file,output_loc):
    """
    .. _unzip_file:
    
    Unzip file to location if files in zip are not already present.

    Parameters
    ----------
    input_file : str
      path of zip file to extract content from.
    output_loc : str
      output path to extract content to.

    Returns 
    -------
      z : ``zipfile.ZipFile`` object
    """
    try:
      with zipfile.ZipFile(os.path.join(input_file), 'r') as z:
        all_extracted=True
        for zn in z.namelist():
          outdirlist=os.listdir(output_loc)
          if zn not in outdirlist:
            all_extracted=False
        if not all_extracted:
          z.extractall(output_loc)
          aux.logger.debug("Unzipping file '{}' in directory '{}'"
            .format(input_file,output_loc))
    except Exception:
      aux.logger.error(
        "Unable to unzip file {} to location {}\n\t".format(input_file,output_loc) +
        "Ensure that the files and locations are valid and that " +
        "the file is not corrupted"
        )
      raise

    return z


def _get_zip_content(zf,**zp):
  """extract zip file(s), return list of files with given format """
  if zp['temp']:
    zp['dst']=tempfile.mkdtemp(prefix="swtools_")
  z=unzip_file(zf, zp['dst'])
  return [os.path.join(zp['dst'],f) for f in z.namelist() 
    if f.upper().split('.')[-1] in zp['cdfsuffix']]


def getCDFlist(src=None, dst=os.curdir,sort_by_t=False, **kwargs):
  """
  .. _getCDFlist:
  
  Get a list of cdf's from a given location if zip or cdf.

  Parameters
  ----------
  src : str, optional
    Path or url of filename(s) or filename directory used to search 
    for cdf file. If set to ``None``, use current directory
    (default ``None``). 
  dst : str, optional
    Directory path of output files if input includes zip files or
    remote files. If directory does not exist it will be created.
    If set to ``None`` use same location as src 
    (default ``None``).
  sort_by_t : bool, optional
    Sort cdf filelist by start time of product (requires filenames
    to follow standard ESA Swarm naming convention)
    (default ``False``).
  
  Keyword Arguments
  -----------------
    cdfsuffix
      see getCDFparamlist_.
    temp : bool
      Specify whether to store cdf files extracted from zip
      temporarily or not (default ``False``).
    includezip : bool
      Traverse zip files if cdf files already in directory. Otherwise
      will only open zip files if no cdf files present. 
      (default ``False``).
    use_ftp : bool
      specify use of ftp server to locate cdf files. Will be 
      overruled (implicitly set to True) if `user` is specified
      (see `dl_ftp`_)(default ``False``).
    filter_param : bool
      Specify whether `param0` (if specified, see `getCDFparams`_) will
      be used for filtering files based on filenames or not 
      (default ``False``).
    sat : str or list of str
      Filter files based on satelite in filename, eg. 
      ``sat=[\'A\',\'C\']`` will only include files from Alpha or 
      Charlie, while ``sat=\'AC\'`` will only accept the joint 
      Alpha-Charlie product files (default ``None``).
    period : list of two `datetime.datetime` objects
      If a product is not (partially) within 
      time window specified, product will be filtered away based on
      filename. Can alternatively be set using two of `start_t`,
      `end_t` and `duration` (default ``None``).
    start_t, end_t : str, scalar or datetime.datetime
      Start/end time of filter. Scalars 
      will be interpreted as fractional days since MJD2000 and strings
      can follow the formats ``'yyyy-mm-dd'``, ``'yyyymmdd'``, 
      ``'yyyymmddThhmmss'`` or ``'yyyymmddhhmmss'`` (default ``None``).
    duration : scalar
      Duration after `start_t` or before `end_t` in fractional days 
      (default ``None``).
    param0
      see getCDFparams_ .

  Returns
  -------
    cdfl : list 
      list of strings of absolute paths to cdf files found in `src`.

  Notes
  -----
  The `src` argument will be overwritten by any `src` entry in 
  `kwargs`.

  If `start_t`. `end_t` and `duration` are all specified, `duration`
  will be ignored.
  
  """

  if not os.path.exists(dst):
    if len(os.path.basename(dst).split('.'))<2:
      os.makedirs(dst)
      aux.logger.debug("creating directory: '{}'".format(dst))
  
  if src is None:
    src = os.curdir
  zp=dict(cdfsuffix=['DBL','CDF'],temp=False,use_ftp=False,includezip=False,
          src=src,dst=dst,filter_param=False,param0=None,sat=None,
          period=None,start_t=None,end_t=None,duration=None)
  zp.update(kwargs)
  zp['cdfsuffix']=aux._tolist(zp['cdfsuffix'])
  if not 'filter_param' in zp:
    zp['param0']=None
  if zp['param0']: zp['param0']=str(zp['param0']).lower()
  if not zp['period']:
    zp['period']=aux._set_period(zp['start_t'],zp['end_t'],zp['duration'])
  
  if not zp['use_ftp'] and 'user' not in zp:
    if isinstance(zp['src'],str):
      zp['src']=os.path.abspath(zp['src'])
    else:
      for i,zloc in enumerate(zp['src']):
        zp['src'][i]=os.path.abspath(zloc)
  zp['dst']=os.path.abspath(zp['dst'])
  aux.logger.debug('kwargs set to {}'.format(zp))
  #if src is directory: glob it for cdf's otherwise for zip's
  def _locate_cdfs(loc):
    cdfl=[]
    zipl=[]
    if os.path.isdir(loc):
      aux.logger.debug(
        "searching through '{}' for CDF's or ZIP's".format(loc))
      for suffix in zp['cdfsuffix']:
        cdfl += glob.glob(loc+"/*.{}".format(suffix.upper())) +\
                glob.glob(loc+"/*.{}".format(suffix.lower()))
      if cdfl:
        aux.logger.debug(
          "Files with suffix '{}' found:\n\t{}"
          .format("' or '".join(zp['cdfsuffix']),'\n\t'.join(cdfl)))
      if (not cdfl) or zp['includezip']:
        zipl+=glob.glob(loc+"/*.ZIP") + glob.glob(loc+"/*.zip")
        if zipl:
          aux.logger.debug("Found following zip files:\n\t{}"
            .format('\n\t'.join(zipl)))
        
    #if zipfile
    elif loc.upper().endswith('ZIP'):
      zipl.append(loc)
    #if cdffile
    elif os.path.basename(loc).upper().split('.')[-1] in zp['cdfsuffix']:
      cdfl.append(loc)#src

    #turn .zip's into .cdf's
    for z in zipl:
      cdfl+=_get_zip_content(z,**zp)

    #aux.logger.debug("CDF's found in '{}':\n\t{}"
    #  .format(loc,'\n\t'.join(cdfl)))
    
    return cdfl

  cdflist=[]
  if isinstance(zp['src'],str):
    if zp['use_ftp'] or ('user' in zp):
        fl=aux._tolist(dl_ftp(zp['src'],**zp))
        if not fl:
          return []
        for fn in fl:
          cdflist+=_locate_cdfs(fn)
    else:
      cdflist=_locate_cdfs(zp['src'])
  else:
    for loc in zp['src']:
      if zp['use_ftp'] or ('user' in zp):
        fn=aux._tolist(dl_ftp(zp['src'],**zp))
        if fn:
          cdflist+=_locate_cdfs(fn)
      else:
        cdflist+=_locate_cdfs(loc)
  
  #check for duplicate basenames:
  bn=[os.path.basename(fn) for fn in cdflist]
  for bname in reversed(bn):
    if bn.count(bname)>1:
      aux.logger.debug('duplicate of {} found. Removing duplicate'.format(bname))
      i=bn.index(bname)
      del bn[i],cdflist[i]

  #filter out files by sat, param0 and period
  _filter_filelist(cdflist,**zp)

  if sort_by_t:
    try:
      cdflist=sorted(cdflist,key=lambda cdfn: _info_from_filename(cdfn)['t0'])
    except Exception: 
      aux.logger.debug('unable to sort by time due to unexpected filename')
  return cdflist


def getCDFparamlist(cdflist,cdfsuffix=['DBL','CDF'],unzip=False,verbose=True):
    """
    .. _getCDFparamlist:
    
    List parameters in the given files if unique.

    Prints a list of parameters for each unique product type based on 
    filename, and returns a list with corresponding information.

    Parameters
    ----------
    cdflist : str or list of str
        Path(s) of cdf file(s). If the only input path is a directory,
        its content will be explored.
    cdfsuffix : str or list of str, optional 
        case-insensitive list of strings of file extensions to accept
        as valid cdf files. If none found, zip files will be explored
        for the same file suffixes (default ``['DBL','CDF']``).
    unzip : bool, optional
        unzip any zip file in `cdflist` (default ``False``).
    
    Returns
    -------
      dict or list
        dictionary of product parameter lists. If only one product
        type is found, only a list of parameters is returned.


    Raises
    ------
      CDFError
        if unable to decode any cdf file specified.

    """
    zp = {'temp':False,'cdfsuffix':cdfsuffix}
    cdflist=aux._tolist(cdflist)
    if len(cdflist)==1 and os.path.isdir(cdflist[0]):
      dirname=cdflist[0] #unnecessary with abspath
      dir_content=os.listdir(dirname)
      zp['dst'] = dirname
      cdflist=[]
      for f in dir_content:
        for fmt in cdfsuffix:
          if f.lower().endswith(fmt):
            cdflist.append(os.path.join(dirname,f))
          if f.lower().endswith('.zip') and unzip:
            cdflist.append(_get_zip_content(os.path.join(dirname,f),**zp))
    else:
      for i in range(len(cdflist)-1,-1,-1):
        if cdflist[i].lower().endswith('.zip') and unzip:
          zp['dst'] = os.path.dirname(cdflist[i])
          cdflist.append(_get_zip_content(cdflist.pop(i),**zp))

    aux.logger.debug("CDF files:\n\t{}".format('\n\t'.join(cdflist)))
    products_listed={}
    for f in cdflist:
        try:
            cdf = pycdf.CDF(f)
        except Exception:
            aux.logger.error("Failed to decode cdf file '{}'".format(f))
            raise
        else:
            prod=_info_from_filename(f,'product')
            if prod not in products_listed:
              if verbose:
                aux.logger.info(
                  "\nList of parameters for file '{}':\n\t{}"
                  .format(os.path.basename(f),'\n\t'
                  .join(map(lambda x:x[0],cdf.items()))))
              products_listed[prod]=[k for k in cdf.items()]
    return aux._single_item_list_open(products_listed)


def getCDFattr(fn,*params,verbose=True,shape=True,fileattr=True):
  """
  .. _getCDFattr:

  Get file and/or parameter attributes of cdf file as a dictionary

  Extract meta-data contained in the cdf file about the file itself 
  and the parameters contained as a nested dictionary.

  Parameters
  ----------

  fn : str
    Path of cdf file
  params : str
    Names of parameters to be extracted from `fn`. If names are not
    known, parameters within cdf files may be accessed using 
    `getCDFparamlist`_. Multiple parameters should be given as 
    separate, comma-separated strings. If no parameters are provided,
    all will be retrieved.

  Keyword Arguments
  -----------------
      verbose : bool
        Print attributes to stdout (default ``True``)
      shape : bool
        Add shape of parameter as parameter attribute ``SHAPE`` 
        (default ``True``)
      fileattr : bool
        Include file attributes (as ``FILE_ATTRIBUTES`) in output
        (default ``True``)
  
  Returns
  -------
    dict
      dictionary of parameter/file dictionaries of attributes.
      If only one parameter is chosen and ``fileattr=False``, 
      only a dictionary of attributes is returned.

  Raises
  ------
    CDFError
      if unable to decode any cdf file specified
    KeyError
      if unable to find any parameter provided in cdf
  """

  attr = {}
  try:
    cdf = pycdf.CDF(f)
  except Exception:
    aux.logger.error("Failed to decode cdf file '{}'".format(f))
    raise
  else:
    if fileattr:
      attr.update({"FILE_ATTRIBUTES":{k:str(v) for k,v in 
        cdf.attrs.items()}})
    if params is None:
      for k,v in cdf.items():
        attr[k] = dict(v.attrs)
        if shape:
          attr[k].update({"SHAPE":str(cdf[k])})
    else:
      for p in params:
        try:
          attr[p] = dict(cdf[p].attrs)
          if shape:
            attr[p].update({"SHAPE":str(cdf[p])})  
        except Exception:
          aux.logger.error("Unable to find parameter {} in {}".format(
            p,fn))
          raise KeyError
    cdf.close()

    if verbose:
      s = 'Attributes of {}\n\n'.format(fn)
      for k,v in sorted(d.items()):
        s += "{}:\n".format(k)
        if isinstance(v,dict):
          for vk,vv in sorted(v.items()):
            s += "\t{}:\n\t\t{}\n".format(vk,vv)
        else:
          s += "\t{}\n".format(v)
      aux.logger.info(s)
  #if len(attr)==1:
  #  attr = attr[[next(iter(attr.keys()))]]
  return aux._single_item_list_open(attr)
    
def extract_parameter(cdflist, parameter,**kwargs):
    """
    .. _extract_parameter:
    
    Extract given parameter from cdf file.

    Extract a parameter's values, unit and name from a list of cdf
    files and concatenate values along a given axis to a 
    `numpy.ndarray`.

    Parameters
    ----------
    cdflist : str or list of str
      Path(s) of cdf file(s).
    parameter : str
      Name of parameter within cdf file. If name is not known, 
      parameters within cdf files may be shown using 
      `getCDFparamlist`_.
    
    Keyword Arguments
    -----------------
      cat : bool
        concatenate parameter values (default ``True``)
      axis : int
        concatenate along specified axis. If ``axis=None`` it will 
        concatenate along last axis. Ie. in a 5x4x2 array it will set
        axis to 3 (default ``None``).

    Returns
    -------
    list
      list of parameter lists, which includes values, units and names
      of parameters. If only one parameter, it will not be contained 
      within a list.

    Raises
    ------
      CDFError
        if unable to decode any cdf file specified
    """
    aux.logger.debug(
      "Attempt to extract parameter '{}' from cdf file list:\n\t{}"
      .format(parameter,'\n\t'.join(cdflist)))
    exdict=dict(cat=True,axis=None)
    exdict.update(kwargs)
    
    values_container={}
    first=True
    if not cdflist:
      aux.logger.debug("Empty cdf list: No values extracted")
      return None,None,None
    else:
      found = False
      unit = None
      for f in cdflist:
        values = None
        u = None
        
        prod=_info_from_filename(f,'product')
        if prod not in values_container:
          values_container[prod]=[]
        
        try:
            cdf = pycdf.CDF(f)
        except Exception:
            aux.logger.error("Failed to decode cdf file '{}'".format(f))
            raise
        else:
            lower_keys = {}

            for k in cdf.keys():
                lower_keys[k.lower()] = k

            if parameter.lower() in lower_keys.keys():
                if parameter != lower_keys[parameter.lower()]:
                    aux.logger.debug(
                      "Assuming parameter to be '{}' rather than '{}'"
                      .format(lower_keys[parameter.lower()], parameter))
                    parameter = lower_keys[parameter.lower()]
                try:
                    u = cdf[parameter].attrs['UNITS']
                    aux.logger.debug(
                      "Unit '{}' found for parameter '{}' in file '{}'"
                      .format(u, parameter, f))
                except Exception:
                    aux.logger.info(
                      "No unit found for parameter '{}' in file '{}'"
                      .format(parameter, f))

                found = True
                aux.logger.debug(
                  "Found parameter '{}' in file '{}'"
                  .format(parameter, f))
                values = cdf[parameter][...]
            if values is not None:
              if values[0] is not None:
                values_container[prod].append(values)
                unit=u
            cdf.close()
      if not found:
        aux.logger.info(
          "Parameter '{}' not found in file(s): \n\t{}"
          .format(parameter,'\n\t'.join(cdflist)))
        
    if exdict['cat']:
      values=[]
      for prod in values_container:
        if values_container[prod]:
          values.append(
            concatenate_values(*values_container[prod],axis=exdict['axis']))
      return Parameter(values,unit,parameter)
    else:
      return Parameter(aux._tolist(values_container),unit,parameter)


def concatenate_values(*an,axis=None):
  """
  .. _concatenate_values:

  Concatenate nD array's along `axis`.

  Concatenate along last axis if none specified."""
  if len(an)==0: 
    aux.logger.debug('No array given, cannot concatenate')
    return 
  if len(an)==1: 
    aux.logger.debug('1 array given, No concatenation needed')
    return an[0]
  
  sh_1=np.shape(an[0])
  similar=True
  for a in an[1:]:
    sh_n = np.shape(a)
    similar *= len(sh_1)==len(sh_n)#same dimension
    if axis is None:
      for i in range(len(sh_1)-1,-1,-1):
        if sh_1[:i]==sh_n[:i]:#same length along an axis
          axis=i
          break

            
  if axis>len(sh_1):
    axis=None#flatten arrays #behaviour not documented on numpy doc 
    aux.logger.debug(
      "axis value too high: {}>{}, might result in flattened array"
      .format(axis,len(sh_1)))
  if not similar:
    aux.logger.error("Some of the array's could not be concatenated")
    raise ValueError
  else:
    aux.logger.debug("concatenating arrays along axis {}".format(axis))
    return np.concatenate(an,axis)


def param_peek(in_arr_cdfl,parameter=None,n_show=5,axis=0,cataxis=None):
  """
  .. _param_peek:

  Show some values contained in given cdf(s)/array.

  Print values from a cdf file, `Parameter`_, list of cdf files or 
  `numpy.ndarray`:

  - cdf input or `Parameter`_ only: parameter name and units
  - 1D-array only :  first and last `n_show` values 
  - all : shape, max, min, mean, median and population
    standard deviation values. 
    number of zeros, NaN's, and largest jumps along given axis
  
  Parameters
  ----------
    in_arr_cdfl : str or list of str or `numpy.ndarray` or `Parameter`_
      path to cdf file(s), or array.
    parameter: str, optional
      if input is cdf file(s), parameter name should be specified. If
      names are not known, parameters within cdf files may be shown
      using `getCDFparamlist`_. (default ``None``).
    n_show : int, optional
      number of values to show from start and end of 1D-array 
      (default ``5``).
    axis : int
      axis to view largest jumps over (default ``0``).
    cataxis : int
      axis to concatenate over. If ``cataxis=None`` it will concatenate 
      along last axis, e.g. in a 5x4x2 array it will set axis to 3 
      (default ``None``).
  """
  u,n=None,None
  
  W_STR = 'Input not understood. Input should be cdf filename(s)'+\
        ' combined with a parameter in numpy.ndarray of dtype'+\
        ' float64 within file(s) OR numpy.ndarray of dtype float64' 
  if isinstance(in_arr_cdfl,str) and parameter:
    p=extract_parameter(aux._tolist(in_arr_cdfl),parameter,axis=cataxis,cat=True)
    p_arr = p.values
    u,n = p.unit,p.name
  elif isinstance(in_arr_cdfl,Parameter):
    p=in_arr_cdfl
    p_arr = p.values
    u,n = p.unit,p.name
  else:
    if not hasattr(in_arr_cdfl,'__iter__'):
      aux.logger.warning(W_STR)
        
      return

    if isinstance(in_arr_cdfl[0],str):
      p=extract_parameter(in_arr_cdfl,parameter,axis=cataxis,cat=True)
      p_arr = p.values
    else:

      p_arr=in_arr_cdfl
  try:
    if not p_arr.dtype==np.float64:
      aux.logger.warning(W_STR) 
      return
  except Exception:
    aux.logger.debug(
      "Input of type '{}' will be read as numpy.ndarray"
      .format(type(p_arr)))
    p_arr=np.array(p_arr)
  peek_info=[]
  sh=np.shape(p_arr)
  dim=len(sh)
  if u and n:
    peek_info.append("Array of '{}',\tunits: '{}'".format(n,u))
  peek_info.append('shape: '+str(sh))
  if dim==2:
    sh_min=min(sh),sh.index(min(sh))
    if sh_min[1]==1:
      p_a_view=p_arr.T
    else:
      p_a_view=p_arr
    for i in range(sh_min[0]):
      peek_info.append(
        '{}/{}\tmax: {:8f},\tmin: {:8f},\tmean: {:8f},\tmedian: {:8f},\tstddev: {:8f}'
        .format(i+1,sh_min[0],np.max(p_a_view[i]),np.min(p_a_view[i]),
          np.mean(p_a_view[i]),np.median(p_a_view[i]),np.std(p_a_view[i]))) 
  else:
      peek_info.append(
        'max: {:8f},\tmin: {:8f},\tmean: {:8f},\tmedian: {:8f},\tstddev: {:8f}'
        .format(np.max(p_arr),np.min(p_arr),np.mean(p_arr),np.median(p_arr),np.std(p_arr)))
    
  if dim==1:
    n_show=abs(int(n_show))
    if n_show:
      peek_info.append('first and last {} variable(s):'.format(n_show))
      for i in range(n_show):
        peek_info.append("{:2d}: {:10f}".format(i,p_arr[i]))
      for i in range(-n_show,0):
        peek_info.append("{:2d}: {:10f}".format(i,p_arr[i]))
  nz=np.count_nonzero(p_arr)
  nan=np.count_nonzero(np.isnan(p_arr))
  dparam_max=np.max(np.abs((p_arr-np.roll(p_arr,1,axis=axis))[1:]))
  totlen=1
  for i in sh:
    totlen*=i
  peek_info.append('Zeros: {:5d}\tFraction zero values: {:.2}'
    .format(totlen-nz,(totlen-nz)/totlen))
  peek_info.append("NaN's: {:5d}\tFraction NaN values:  {:.2}"
    .format(nan,(nan)/totlen))
  peek_info.append("Largest jump over 1 index[along axis={}]: {:10f}"
    .format(axis,dparam_max))

  aux.logger.info('\n\t'.join(peek_info))
  return


def dl_ftp(url='swarm-diss.eo.esa.int',**kwargs):
  """
  .. _dl_ftp:

  Download files from ftp server.

  Download files from a specified ftp url using filtering and/or
  interactive specification, if not already downloaded to given location.

  Parameters
  ----------
    url : str
      url of filename(s) or filename directory used to search for cdf file.
    
  Keyword Arguments
  -----------------
      user : str 
        ftp server username
      pw : str
        ftp server password
      dst : str
        location to download to
      cdfsuffix 
        see `getCDFparamlist`_
      use_filtering
        see `getCDFlist`_
      param0 
        see `getCDFparams`_
      sat
        see `getCDFlist`_
      period 
        see `getCDFlist`_
      use_passive_mode : bool, optional
        Try to set this to ``False`` if having difficulties with 
        connecting to ftp server (default ``True``). 
      use_current : bool, optional
        ignore directories named 'Previous' (default ``True``).
      use_color : bool, optional
        allow coloring in listing of files/directories 
        (default ``True``). 
  Returns
  -------
  str or list of str
    Paths to files specified for download. If only one file, a string 
    will be returned.

  Raises
  ------
  OSError
    if unable to connect to ftp server
  """
  
  pre='ftp://'
  
  ftpkw=dict( user=None,pw=None,use_color=True,dst=os.getcwd(),
              cdfsuffix=['DBL','CDF'],sat='',use_current=True,
              param0=None,period=None,use_filtering=True,
              use_passive_mode=True)

  
  ftpkw.update(kwargs)
  
  
  if ftpkw['param0']: ftpkw['param0']=str(ftpkw['param0']).lower()

  
  aux.logger.debug('url: {}'.format(url))
  aux.logger.debug('kwargs in dl_ftp: {}'.format(ftpkw))
  
  if url.startswith(pre):
    url=url[len(pre):]
  if not (ftpkw['user'] and ftpkw['pw']):
    if not ftpkw['user']:
      ftpkw['user']=input('username(ENTER if none required): ')
    ftpkw['pw']=input('password(ENTER if none required): ')
  
  s_factory = ftputil.session.session_factory(
        base_class=ftplib.FTP,
        port=21,
        use_passive_mode=ftpkw['use_passive_mode'],
        debug_level=None
        )

  try:
    baseurl,url_lloc=url.split('/',1)
  except Exception:
    baseurl,url_lloc=url,None
  try:
    aux.logger.debug('attempting to connect to {}...'.format(baseurl))
    dummy_host=ftputil.FTPHost(baseurl, ftpkw['user'], ftpkw['pw'],
      session_factory=s_factory)
    dummy_host.close()
  except OSError:
    aux.logger.error('Unable to connect to {}'.format(url) )
    raise

  with ftputil.FTPHost(baseurl, ftpkw['user'], ftpkw['pw']
                      ,session_factory=s_factory) as host:
    aux.logger.debug('successfully connected to {}'.format(baseurl))
    
    if url_lloc:
      aux.logger.debug('changing remote directory to {}'.format(url_lloc))
      host.chdir(host.path.dirname(url_lloc))
      
    found=False
    
    for fmt in ftpkw['cdfsuffix']+['.zip']:
      if url.endswith(fmt.lower()) or url.endswith(fmt.upper()):
        if host.path.isfile(host.path.basename(url)):
          found=True
          aux.logger.debug('Found file:{}'.format(url))
          if os.path.isfile(
              os.path.join(ftpkw['dst'],host.path.basename(url))):
            aux.logger.debug('File {} already downloaded.'
              .format(host.path.basename(url)))
            return os.path.join(ftpkw['dst'],host.path.basename(url))
        
    
    def list_files(names=None,color=True):
      if not names:
        names = host.listdir(host.curdir)
      for i,name in enumerate(names):
        digits=len(str(len(names)))
        colored=False
        if ftpkw['use_color'] and color:
          if host.path.isdir(name):
              colored=True
              print('({c:{f}{w}})'
                .format(c=i+1,f='',w=digits),aux._CSI2str(34,name+'/'))
          if not colored:
            for fmt in ftpkw['cdfsuffix']+['.zip']:
              if name.endswith(fmt.upper()) or name.endswith(fmt.lower()):
                colored=True
                print('({c:{f}{w}})'
                  .format(c=i+1,f='',w=digits),aux._CSI2str(32,name))
                break

        if not colored:    
          print('({c:{f}{w}}){n}'.format(c=i+1,f='',w=digits,n=name))
      return names
    def filter_remote_dirs(flist,**filters):
      #remove anything named 'Previous'
      if filters['use_current']:
          if 'Previous' in flist:
            aux.logger.debug("'Previous' filtered away")
            flist.remove('Previous')
            return
      #remove all folders not starting with param_dic[filters['param0']]
      #if any such folders found
      if filters['param0'] in aux.PRODUCT_DIC:
          dirnames_rm=[]
          found_param=False
          for f in flist:
            if host.path.isdir(f):
              if f[:3]==aux.PRODUCT_DIC[filters['param0'].lower()]:
                aux.logger.debug('selecting param dir {}'.format(f))
                found_param=True
              else:
                dirnames_rm.append(f)
          if found_param:
            for d in dirnames_rm:
              aux.logger.debug("'{}' filtered away".format(d))
              flist.remove(d)
            for f in reversed(flist):
              if not host.path.isdir(f):
                aux.logger.debug("'{}' filtered away".format(f))
                flist.remove(f)

            return
      #remove all 'Sat_X' folders where X not in filters['sat']
      s=aux._tolist(filters['sat'])
      if s[0]:
        for i,si in enumerate(s):
          s[i]=si.upper()
        for f in reversed(flist):
          if host.path.isdir(f):
            if f.startswith('Sat_') and \
                ''.join(f.split('_')[1:]) not in s:
              aux.logger.debug("'{}' filtered away".format(f))
              flist.remove(f)
      return 

    def read_remote_txt_file(txtpath):
      """read a .txt file on host and return output"""
      with host.open(txtpath) as f:
        return [line.strip() for line in f]
      
    
    while not found:
      flist = host.listdir(host.curdir)
      if ftpkw['use_filtering']:
        filter_remote_dirs(flist,**ftpkw)
      if len(flist)==1:
        aux.logger.debug('entering {}...'.format(flist[0]))
        selected=flist
      elif not flist:
        return flist
      else:#more than one file
        t0=None
        if host.path.isfile(flist[0]):
          try:
            t0=_info_from_filename(flist[0],'t0')
          except ValueError:
            pass
        if t0 and ftpkw['period']:
          found=True
          p=ftpkw['period']
          for i in range(len(flist)-1,-1,-1):
            info=_info_from_filename(flist[i])
            if 't0' in info:
              t0,t1=info['t0'],info['t1']
              if not ((p[0]<t0 and  p[1]>t0) and (p[0]<t1 and p[1]>t1)):
                del flist[i]
          #print('flist:',flist)
          selected=flist
          break
        else:
          flist=list_files(flist)
          selected=[flist[i-1] for i in \
            aux._in2range(len(flist),msg=aux.DL_FTP_MSG) if i is not 0]
          if not selected: break
      

      if host.path.isdir(selected[0]) and len(selected)>1:
        #multiple dirs selected #(>0 dirs + files)  
        found=True
      else:
        if selected[0].endswith('.txt') or selected[0].endswith('.TXT'):
          #txt selected #ignore other files than first
          flist=read_remote_txt_file(selected[0])
          list_files(flist)
          selected=[flist[i-1] for i in \
            aux._in2range(len(flist),msg=aux.DL_FTP_MSG) if i is not 0]
          found=True
        elif host.path.isfile(selected[0]):
            #file(s) selected #(>0 files + dirs)
            found=True
        else:#1 dir selected
            host.chdir(selected[0])
    if not selected:
      aux.logger.info('No file selected')
      return []
    f_dl,f_out=[],[]
    for s in selected:
      if host.path.isdir(s):
        #if directories inside: dl everything
        dir_content=host.listdir()
        for f in dir_content:
          if host.path.isdir(f):
            f_dl.append(s)
            break
        else:
          if use_filtering:
            _filter_filelist(dir_content,**ftpkw)
          f_dl += dir_content
      else:
        f_dl.append(s)
    if aux._is_interactive() and len(f_dl)>10:
      aux.logger.info(
        "{} entries will be downloaded to {}"
        .format(len(f_dl),ftpkw['dst']))
      answer='s'
      while answer.strip().lower()=='s':
        answer=input("Continue? [y/n/s shows entries] (y): ")
        if answer.strip().lower()=='s':
          aux.logger.info(
            'List of entries to be downloaded:\n\t{}'
            .format('\n\t'.join(f_dl)))
      if answer.strip('Yy '):
        return []
    for f in f_dl:
        urlloc = pre + baseurl + host.path.abspath(f)
        localloc = os.path.abspath(os.path.join(
          ftpkw['dst'],host.path.basename(f)))
        f_out.append(localloc)
        if os.path.isfile(localloc):
          aux.logger.info(
            '{} already in {}'.format(urlloc,os.path.dirname(localloc)))
          continue

        
        aux.logger.info('{} downloading to\n\t{}'.format(urlloc,localloc))
        host.download(f, localloc)
    return aux._single_item_list_open(f_out)


def read_sp3(fname,doctype=2,SI_units=True):
  """
  .. _read_sp3:
  
  Read SP3 ascii files to array.

  Read orbital format 'Standard Product # 3' (SP3) to numpy array of
  two SP3 document types  as shown by example 1 and example 2 in 
  https://igscb.jpl.nasa.gov/igscb/data/format/sp3_docu.txt . Output 
  may be converted to SI units. 

  Parameters
  ----------
  fname : str
    Path of SP3 file.
  doctype : int
    Two SP3 document formats:
    
      1. only position at given timestamps 
      2. position and velocity at given timestamps with 
         rate-of-change of clock correction
  
    `doctype` corresponds to these two options (default ``2``).
  SI_units : bool
    Convert to SI-units from SP3 units (default ``True``).

  Returns
  -------
  list of `numpy.ndarray`
    ``[x, y, z, t, header]`` for ``doctype=1``, 
    ``[x, y, z, vx, vy, vz, dt, t, header]`` for ``doctype=2``,
    where ``header`` is the first 22 lines of the SP3 document as a
    string.

  Raises
  ------
    EOFError
      If the specified SP3 file is empty.
  """
    
  with open(fname) as f:
      #read header to list
      try:
        header = [next(f)]
      except StopIteration:
        aux.logger.error("File is empty")
        raise EOFError('Empty file')
      c=1
      for line in f:
        header.append(line)
        c+=1
        if c==22:break

  def str2dt(s):
    tlist=s.split() #split time by whitespace
    tlist[:-1]=list(map(lambda y:int(y),tlist[:-1]))
    return dt.datetime(*tlist[:4])+dt.timedelta(
      seconds=(tlist[4]*60+float(tlist[5])))
  
  str2dt_v=np.vectorize(str2dt)
  
  km2m=1000
  dm2m=0.1
  us2s=1e-6
  
  if doctype==1:
    pattern=r'\*.{2}(.{28})\s*\n' \
      r'P.{3}(.{14})(.{14})(.{14})(.{14})\s*\n'
    dtype = [('date',np.str_,28),('x',np.float64),('y',np.float64),
            ('z',np.float64),('t',np.float64)]
    a=np.fromregex(fname,pattern,dtype)
    t=np.empty(len(a['date']),dtype=object)
    t=str2dt_v(a['date'])+aux._float_us_to_timedelta_v(a['t'])
    if SI_units:
      #km -> m
      a['x']*=km2m
      a['y']*=km2m
      a['z']*=km2m
      
    #if as_struct:
    #  return np.core.records.fromarrays(
    #    [a['x'],a['y'],a['z'],a['t'],t],
    #    names='x,y,z,t,dt,date')
    return [a['x'],a['y'],a['z'],t,header]

  elif doctype==2:
    pattern=r'\*.{2}(.{28})\s*\n' \
      r'P.{3}(.{14})(.{14})(.{14})(.{14})\s*\n' \
      r'V.{3}(.{14})(.{14})(.{14})(.{14})'

    dtype = [('date',np.str_,28),('x',np.float64),('y',np.float64),
            ('z',np.float64),('t',np.float64),('vx',np.float64),
            ('vy',np.float64),('vz',np.float64),('dt',np.float64)]

    a=np.fromregex(fname,pattern,dtype)
    t=np.empty(len(a['date']),dtype=object)
    t=str2dt_v(a['date'])+aux._float_us_to_timedelta_v(a['t'])
    
    km2m=1000
    dm2m=0.1
    us2s=1e-6

    if SI_units:
      #km -> m
      a['x']*=km2m
      a['y']*=km2m
      a['z']*=km2m
      
      #dm/s -> m/s
      a['vx']*=dm2m
      a['vy']*=dm2m
      a['vz']*=dm2m
      
      #us/s -> s/s
      a['dt']*=us2s


    #if as_struct:
    #  return np.core.records.fromarrays(
    #    [a['x'],a['y'],a['z'],a['t'],a['vx'],a['vy'],a['vz'],a['dt'],t],
    #    names='x,y,z,t,vx,vy,vz,dt,date')
    return [a['x'],a['y'],a['z'],a['vx'],a['vy'],a['vz'],a['dt'],t,header]


def _info_from_filename(fn,key=None):
  """get information about product based on filename"""
  #MM_CCCC_TTTTTTTTTT_yyyymmddThhmmss_YYMMDDTHHMMSS_vvvv.DBL
  
  fn=os.path.basename(fn)
  filelen=len(fn)
  if filelen not in [59,63,70]: 
    aux.logger.debug(
      "filename '{}' does not follow standard syntax set for ESA Swarm CDF's "
      .format(fn))
    if filelen<55:
      return 

  mission=fn[0:2]      #SW
  oper=fn[3:7]=='OPER' #operational or reprocessing #OPER/RPRO
  product=fn[8:18]        #product name
  version=fn[51:55]    #version

  try:
    #start time of validity   
    t0_y=int(fn[19:23])  #year
    t0_M=int(fn[23:25])  #month
    t0_d=int(fn[25:27])  #day
    t0_h=int(fn[28:30])  #hour
    t0_m=int(fn[30:32])  #minute
    t0_s=int(fn[32:34])  #second
    #end time of validity   
    t1_y=int(fn[35:39])  #year
    t1_M=int(fn[39:41])  #month
    t1_d=int(fn[41:43])  #day
    t1_h=int(fn[44:46])  #hour
    t1_m=int(fn[46:48])  #minute
    t1_s=int(fn[48:50])  #second
  except Exception:
    product_info = {'mission':mission,'oper':oper,'product':product,
                    'version':version}
  else:

    t0=dt.datetime(t0_y,t0_M,t0_d,t0_h,t0_m,t0_s)
    t1=dt.datetime(t1_y,t1_M,t1_d,t1_h,t1_m,t1_s)
    if filelen in [59,63]:
      product_info = {'t0':t0,'t1':t1,'mission':mission,
                      'oper':oper,'product':product,'version':version}
    else:
      dataset=fn[55:].split('.')[0]
      product_info = {'t0':t0,'t1':t1,'mission':mission,'oper':oper,
                      'product':product+dataset,'version':version}
  
  if key:
    if key in product_info:
      return product_info[key]
    else: 
      return
  
  return product_info


def _filter_filelist(fl,**kwargs):
  """get filelist subset based on filtering by filename"""
  filters={ 'sat':None,
            'param0':None,
            'period':None
  }
  filters.update(kwargs)
  
  #prod_in_dic=bool(filters['param0'] in aux.PRODUCT_DIC)
  prod_in_dic=filters['param0'] is not None
  if prod_in_dic: prod_in_dic = filters['param0'].lower() in aux.PRODUCT_DIC
  
  sat_in_dic=filters['sat'] is not None
  period_in_dic=filters['period'] is not None
  

  def keep_item(f,filters):
    prod_info=_info_from_filename(f)
    if (sat_in_dic and 
      prod_info['product'][3] not in aux._tolist(filters['sat'])):
        return False
    elif (prod_in_dic and 
        aux.PRODUCT_DIC[filters['param0'].lower()]!=prod_info['product'][:3]):
      return False
    elif (period_in_dic and 
        not (aux._is_in_period(prod_info['t0'],filters['period']) or
        (aux._is_in_period(prod_info['t1'],filters['period'])))):
      return False
    else:
      return True

  for f in reversed(fl):
    if not keep_item(f,filters):
      fl.remove(f)

  ### consider using pythons native filter(function,iterable) function which returns filtered list


def read_EFI_prov_txt(fname,*params,filter_nominal=False):
  """
  .. _read_EFI_prov_txt:
  
  Read provisional EFI products.

  Read an ascii file containing provisional EFI products.

  Parameters
  ----------
  fname : str
    Path and filename of provisional EFI ascii file.
  params : 
    Names of parameters to be extracted from `fname`. Possible values:
    - 'timestamp'
    - 'latitude'
    - 'longitude'
    - 'radius'
    - 'n' 
    - 't_elec'
    - 'u_sc'
    - 'flag'
    
    If no parameters are specified, all will be returned in a 
    dictionary. Multiple parameters should be given as separate, 
    comma-separated strings.
  filter_nominal : bool
    Only extract data where ``Flag=1`` (default ``False``).

  Returns
  -------
  List or dictionary 
    If parameters specified, returns list of `numpy.ndarray`'s; otherwise
    it will return dictionary of all parameter `numpy.ndarray`'s.

  Notes
  -----
  This function has not been subject to any optimization, and is slow. 
  Multiple file support or data concatenation is as of yet not 
  implemented.
  
  """
  
  filedata=np.loadtxt(fname,usecols=(2,3,4,5,7,10,11,12,13),unpack=True)
  flag=np.around(filedata[-1]).astype(int)
  filedata[0]= np.around(filedata[0]).astype(int)
  filedata[1]= np.around(filedata[1]).astype(int)
  fdata_list=[]
  if filter_nominal:
    nominal = flag==1
    for d in filedata:
      fdata_list.append(d[nominal])
  else:
    fdata_list=list(filedata)
  timestamp = aux._MJD2000_datetime(fdata_list[0],fdata_list[1])
  
  out_dic={ 'timestamp':timestamp,
            'latitude' :fdata_list[2],
            'longitude':fdata_list[3],
            'radius': fdata_list[4],
            'n': fdata_list[5],
            't_elec':fdata_list[6],
            'u_sc': fdata_list[7],
            'flag':fdata_list[8] }
  if len(params):
    out=[]
    try:
      for p in aux._tolist(params):
        out.append(out_dic[p.lower()])
        aux.logger.debug("param '{}' found".format(p))
      return aux._single_item_list_open(out)
    except Exception:
      pass
  return out_dic


class Parameter:
  """
  .. _Parameter:

  Container for a single parameter.

  Has 3 attributes:

    - values
    - name
    - unit

  `values` can also be accessed by calling the parameter
  (eg. ``myparam()``) or by accessing its indices (eg. ``myparam[2]``) 
  """
  def __init__(self,values,unit='',name=''):
    self.values=aux._single_item_list_open(values)
    self.unit=unit
    self.name=name
  def __call__(self):
    return self.values
  def __repr__(self):
    return ' '.join([Parameter.__name__,self.name,self.unit,'\n']) +\
        repr(self.values)
  def __getitem__(self, key):
    return self.values[key]
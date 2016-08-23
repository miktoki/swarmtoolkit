#!/usr/bin/python
# -*- coding: utf-8 -*-
import datetime as dt

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from . import aux
  
__all__ = [ 'plot_align',
            'plot',
            'plot_geo',
            'plot_scatter',
            'plot_twinx',
            'save_raw']


def plot(x,y,*xy,show=False,fmt_t=True,figsize=plt.rcParams["figure.figsize"],logx=False,logy=False,legends=[],lloc='best',lhide=False,lbox=False,lfontsize=15,colors=[],**plotkwargs):
  """
  .. _plot:
    
    Basic plot using `matplotlib`.

  A convenience function to use `matplotlib.pyplot.plot 
  <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot>`_ 
  with some set parameters. Of particular note this function handles an
  x-axis with datetimes better than the default behaviour in matplotlib.

  Parameters 
  ----------
  x : array_like
    Input x-values.
  y : array_like
    Input y-values.
  xy : optional
    Additional x- and y-values.
  show : bool, optional
    Show plot (default ``False``).
  fmt_t : bool, optional
    Format datetime x-ticks (see `matplotlib.figure.autofmt_xdate
      <http://matplotlib.org/api/figure_api.html?highlight=autofmt_xdate>`_)
      (default ``True``).
  figsize : tuple of length 2, optional
    Size of figure as tuple of width and height in inches 
    (default ``matplotlib.pyplot.rcParams["figure.figsize"]``).
  logx : bool, optional
    Set x-axis scale to log (default ``False``).
  logy : bool, optional
    Set y-axis scale to log (default ``False``).
  legends : list_like, optional
    Add legend(s)(default ``[]``).
  lloc : str or int, optional
    Location of legend. Can be one of:
        'best' : 0, (default)
        'upper right'  : 1,
        'upper left'   : 2,
        'lower left'   : 3,
        'lower right'  : 4,
        'right'        : 5,
        'center left'  : 6,
        'center right' : 7,
        'lower center' : 8,
        'upper center' : 9,
        'center'       : 10
  lhide : bool, optional
    Do not show legends. Useful to combine legends with twinx legends 
    (default ``False``).
  lbox : bool
    box legends in semi-transparent box (default ``False``)
  lfontsize : scalar
    fontsize of legend (default ``15``)
  colors : list_like, optional
    Color cycle to use in plot (eg. ``['r','g','b']`` will show plots 
    in red, green and blue (see 
    `matplotlib.colors <http://matplotlib.org/api/colors_api.html>`_ 
    for more examples). Default will use colormap set in the rcParams.
    (default ``[]``).    
  plotkwargs : optional
    Additional keyword arguments to pass on to 
    `matplotlib.pyplot.plot`_, these will be overwritten if conflicting 
    with other values.

  Returns
  -------
  tuple
    `matplotlib.figure.Figure 
    <http://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure>`_ 
    and
    `matplotlib.axes.Axes 
    <http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes>`_ 
    instances for plot

  See also
  --------
  plot_twinx, plot_align
  
  """
  fig,ax=plt.subplots(figsize=figsize)
  legends=aux._tolist(legends)
  if fmt_t and isinstance(x[0],dt.datetime):
    fig.autofmt_xdate()
    #plt.xticks(rotation=70)
  if legends:
    ax.plot(x,y,label=legends[0],**plotkwargs)
    try:
      legends[i+1]
    except Exception:
      legends.append(None)
    for i in range(len(xy)//2):
      ax.plot(xy[2*i],xy[2*i+1],label=legends[i+1],**plotkwargs)
  else:
    ax.plot(x,y,*xy,**plotkwargs)
        
  #import matplotlib.dates as mdates
  #ax.fmt_xdata = mdates.DateFormatter('%y%m%d-%H:%M:%S')
  
  if logx:
    ax.set_xscale('log')
  if logy:
    ax.set_yscale('log')
  if show:
    plt.show()
  if legends and not lhide: 
    boxarg={'framealpha':0.0}
    if lbox:
      boxarg={'fancybox':True, 'framealpha':0.75}
    ax.legend(labelspacing=0.25,loc=lloc, fontsize=lfontsize, **boxarg)
  if colors:
    ax.set_color_cycle(colors)
  return fig,ax
#style needs to be specified explcitly as: color='green', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=12 etc.


def plot_twinx(x,y,*xy,show=False,logy=False,legends=[],lloc='best',lall=True,lbox=False,lfontsize=15,ax=None, colors=[],**plotkwargs):
  """
  Overplot with a twin x-axis.

  Share same x-axis as another plot, but with separate y-axis values.
  Should be used in conjunction with another plot function 
  (eg. `plot`_).

  Parameters
  ----------
  x : array_like
    Input x-values.
  y : array_like
    Input y-values.
  xy : optional
    Additional x- and y-values.
  show : bool, optional
    Show plot (default False).
  logy : bool, optional
    Set y-axis scale to log (default ``False``).
  legends : list_like, optional
    Add legend(s) (default ``[]``).
  lloc : str or int, optional
    Location of legend. Can be one of::
        'best'         : 0 (default)
        'upper right'  : 1
        'upper left'   : 2
        'lower left'   : 3
        'lower right'  : 4
        'right'        : 5
        'center left'  : 6
        'center right' : 7
        'lower center' : 8
        'upper center' : 9
        'center'       : 10
  lall : bool, optional
    Combine legends from `ax` with `legends` (default ``True``).
  lbox : bool
    box legends in semi-transparent box (default ``False``)
  lfontsize : scalar
    fontsize of legend (default ``15``)
  ax : matplotlib.axes.Axes
    Axes instance of plot to share x-axis with. If ```ax=None```, get 
    current Axes instance (default None). 
  colors : list_like, optional
    Color cycle to use in plot (eg. ``['r','g','b']`` will show plots 
    in red, green and blue (see 
    `matplotlib.colors <http://matplotlib.org/api/colors_api.html>`_ 
    for more examples). Default will use colormap set in the rcParams.
    (default ``[]``).    
  plotkwargs : optional
    Additional keyword arguments to pass on to 
    `matplotlib.pyplot.plot`_, these will be overwritten if conflicting
    with other values.

  Returns
  -------
    `matplotlib.axes.Axes`_

  See also
  --------
  plot, plot_align
  """
  if not ax: 
    ax=plt.gca()

  ax2=ax.twinx()
  legends=aux._tolist(legends)
  
  if colors:
    ax2.set_color_cycle(colors)
  else:
    for i in range(len(ax.lines)):
      next(ax2._get_lines.prop_cycler)
    

  if legends:
    ax2.plot(x,y,label=legends[0],**plotkwargs)
    for i in range(len(xy)//2):
      try:
        legends[i+1]
      except Exception:
        legends.append(None)
      ax2.plot(xy[2*i],xy[2*i+1],label=legends[i+1],**plotkwargs)
  else:
    ax2.plot(x,y,*xy,**plotkwargs)
    

  if logy:
    ax2.set_yscale('log')
  
  if legends:
    boxarg={'framealpha':0.0}
    if lbox:
      boxarg={'fancybox':True, 'framealpha':0.75}
    lines2, labels2 = ax2.get_legend_handles_labels()
    if lall:
      lines, labels = ax.get_legend_handles_labels()
      ax2.legend(lines + lines2, labels + labels2,
        labelspacing=0.25,loc=lloc,fontsize=lfontsize,**boxarg)
    else:
      ax2.legend(lines2,labels2,
        labelspacing=0.25,loc=lloc,fontsize=lfontsize,**boxarg)
      
  return ax2


def plot_align(p1,p2,t1,t2,k=3,align_to=False,show=False,fmt_t=True,figsize=plt.rcParams["figure.figsize"],logx=False,logy=False,legends=[],lloc='best',lhide=False,colors=[],**plotkwargs):
  """
  Convenience function which combines `align_param`_ with `plot`_

  Align p1 and p2 using interpolation such that values will be sampled 
  on the same time steps. Output will be the same as for `plot`_.

  See `align_param`_ and `plot`_ for more information on arguments.
  """
  aligned=align_param(p1,p2,t1,t2,k=k,align_to=align_to)
  return plot(aligned[2],aligned[0],aligned[2],aligned[1],show=show,
      fmt_t=fmt_t,figsize=figsize,logx=logx,logy=logy,legends=legends,lloc=lloc,colors=colors,lhide=lhide,**plotkwargs)


def plot_scatter(x,y,param,show=False,fmt_t=True,figsize=plt.rcParams["figure.figsize"], vmax=None,vmin=None,cmap=plt.rcParams["image.cmap"],cbar=True,**scatterkwargs):
  """
  Scatterplot with colorbar using matplotlib.pyplot.scatter.

  Parameters
  ----------
  x : array_like
    x-coordinates of `param`.
  y : array_like
    y-coordinates of `param`.
  param : array_like
    value(determining colour) of `param` at each (x,y)-coordinate.
  show : bool, optional
    Show plot (default ``False``).
  fmt_t : bool, optional
    Format datetime x-ticks 
    (see `matplotlib.figure.autofmt_xdate`_ ) (default ``True``).
  figsize : tuple of length 2, optional
    Size of figure as tuple of width and height in inches 
    (default ``matplotlib.pyplot.rcParams["figure.figsize"]``).
  vmax : scalar, optional
    vmax sets the upper bound of the colour data. If either `vmin` or
    `vmax` are None, the min and max of the color array is used 
    (default ``None``).
  vmin : scalar, optional
    vmin sets the lower bound of the colour data. If either `vmin` or
    `vmax` are None, the min and max of the color array is used 
    (default ``None``).
  cmap : matplotlib.colors.ColorMap
    colormap to be used in plot 
    (default ``matplotlib.pyplot.rcParams["image.cmap"]``).
  cbar : bool, optional
    use colorbar (default ``True``).
  
  Keyword Arguments
  -----------------
  s : scalar or array_like 
    (size of points)**2 (default ``3``).
  linewidths : scalar
    (default ``0.0``).
  alpha : scalar
    blending value between 0(transparent) and 1(opaque).

  Returns
  -------
  tuple
    `matplotlib.figure.Figure`_ and `matplotlib.axes.Axes`_
     instances for plot as a tuple

  """
  s_kwargs={'s':3,'cmap':cmap,'linewidths':0.0,'alpha':1.0}
  s_kwargs.update(scatterkwargs)
  fig,ax=plt.subplots(figsize=figsize)
  
  if fmt_t and isinstance(x[0],dt.datetime):
    fig.autofmt_xdate()
  
  ax.set_xlim([x[0],x[-1]])
  ax.scatter(x,y,c=param,vmin=vmin,vmax=vmax,**s_kwargs)
  if show:
    plt.show()
  
  if cbar:
    import matplotlib as mpl
    norm=mpl.colors.Normalize(vmax=vmax, vmin=vmin)
    c_ax=fig.add_axes([0.96, 0.1, 0.025, 0.8])
    cb1 = mpl.colorbar.ColorbarBase(c_ax, cmap=cmap,norm=norm)#
  return fig,ax


def plot_geo(lat,lon,param,ptype='scatter',figsize=plt.rcParams["figure.figsize"],cmap=plt.rcParams["image.cmap"],cbar=True,dark_map=False,show=False,contourlevels=15,log_contour=False,show_lat=True,show_lon=False,show_grd=True,**kwargs):
  """
  Plot parameter on the globe using `mpl_toolkits.basemap.Basemap`.

  Parameters
  ----------
  lat : array_like
    Latitude of `param`.
  lon : array_like
    Longitude of `param`.
  param : array_like
    Value of `param` at each ``(lat,lon)``-coordinate.
  ptype : ``{'scatter'|'colormesh'|'contour'}``, optional
    Set plot type (default ``'scatter'``).
  figsize : tuple, optional
    Size of figure as tuple of width and height in inches 
    (default ``matplotlib.pyplot.rcParams["figure.figsize"]``).
  cmap : matplotlib.colors.ColorMap or str
    (Name of) colormap to be used in plot 
    (default ``matplotlib.pyplot.rcParams["image.cmap"]``).
  cbar : bool, optional
    use colorbar (default ``True``).
  dark_map : bool, optional
    draw map with darker tones of gray (default ``False``).
  show : bool, optional
    Show plot (default ``False``).
  contourlevels : int, optional
    number of coutour levels to use in coutourplot (default ``15``).
  log_contour : bool, optional
    plot contour levels using logarithmic distances between lines.
  show_lat : bool, optional
    show labels for latitude(requires `show_grid`) (default ``True``).
  show_lon : bool, optional
    show labels for longitude(requires `show_grid`) (default ``False``).
  show_grid : bool, optional
    show gridlines(graticules) (default ``True``). 
  
  Returns 
  -------
  tuple 
    `matplotlib.figure.Figure`_ and `mpl_toolkits.basemap.Basemap
    <http://matplotlib.org/basemap/api/basemap_api.html#mpl_toolkits.basemap.Basemap>`_ 
    object for plot.

  Notes
  -----
  See http://matplotlib.org/basemap/api/basemap_api.html for full set
  of possible keyword arguments. In particular the projection can be
  set with ``projection``, which is by default set to 'moll'
  (Mollweide projection) in this funciton. In addition, depending on 
  the value of `ptype`, the following values are used as default:

  `scatter`:

    See mpl_toolkits.basemap.scatter
    default values:

        - linewidths : 0.0 
        - vmin : min(param)
        - vmax : max(param)
  
  `colormesh`:
  
    See mpl_toolkits.basemap.pcolormesh
    Note that as `colormesh requires 2D arrays; providing 
    ``latlon=True`` allows latitude and longitude to be converted to a
    2d mesh properly from two 1D arrays.
    default values:

        - shading : flat
        - alpha : 0.8
  
  `contour`:

    See mpl_toolkits.basemap.pcolormesh.contour
    Note that as `colormesh requires 2D arrays; providing 
    ``latlon=True`` allows latitude and longitude to be converted to a
    2d mesh properly from two 1D arrays.
    default values :
  
        - animated : True
  
  """
  mapkw={ 'llcrnrlon':None, 'llcrnrlat':None, 'urcrnrlon':None, 
          'urcrnrlat':None, 'llcrnrx':None, 'llcrnry':None, 'urcrnrx':None,
          'urcrnry':None, 'width':None, 'height':None, 'projection':'moll',
          'resolution':'c', 'area_thresh':None, 'rsphere':6370997.0,
          'ellps':None, 'lat_ts':None, 'lat_1':None, 'lat_2':None, 
          'lat_0':None, 'lon_0':0, 'lon_1':None, 'lon_2':None, 'o_lon_p':None,
          'o_lat_p':None, 'k_0':None, 'no_rot':False, 'suppress_ticks':True, 
          'satellite_height':35786000, 'boundinglat':None, 'fix_aspect':True, 
          'anchor':'C', 'celestial':False, 'round':False, 'epsg':None, 
          'ax':None}
  
  for kw in list(kwargs.keys()):
    if kw in mapkw:
      mapkw[kw]=kwargs.pop(kw)
  # to set default values different from standard default values
  s_kwargs={
    'cmap':cmap,'linewidths':0.0, 'vmin':np.min(param),'vmax':np.max(param)}
  contour_kwargs={'linewidths':0.5,'animated':True}
  cmesh_kwargs={'shading':'flat','cmap':s_kwargs['cmap'],'alpha':0.8}
    
  m = Basemap(**mapkw)
  if ptype in ('colormesh','contour') and len(np.asarray(lon).shape)==1:
    x,y=np.meshgrid(lon,lat)
  else:#scatter or 1D contour/colormesh
    x, y = m(lon,lat)
    if ptype in ('colormesh','contour'):
      cmesh_kwargs['latlon']=True

  fig=plt.figure(figsize=figsize)
  if show_grd:
    if (mapkw['projection'] not in ('ortho',)) and show_lat:
      m.drawparallels(np.arange(-90,90,30),labels=[1,0,0,0])
    else: 
      m.drawparallels(np.arange(-90,90,30))
    if show_lon:
      m.drawmeridians(np.arange(m.lonmin,m.lonmax+30,60),labels=[0,0,0,1])
    else:  
      m.drawmeridians(np.arange(m.lonmin,m.lonmax+30,60))
  
  if dark_map:
    m.drawmapboundary(fill_color='#333333')
    m.fillcontinents(color='#000000',lake_color='#333333',zorder=0, alpha=0.8)
  else:  
    m.drawmapboundary(fill_color='#dddddd')
    m.fillcontinents(color='#ffffff',lake_color='#dddddd',zorder=0, alpha=0.8)
  if ptype=='colormesh':
    cmesh_kwargs.update(kwargs)
    im=m.pcolormesh(x,y,param,**cmesh_kwargs)
  elif ptype=='contour':
    if not log_contour:
      clevs=np.linspace(np.min(param),np.max(param),contourlevels)
    else:
      clevs=np.logspace(np.min(param),np.max(param),contourlevels)
    contour_kwargs.update(kwargs)
    im=m.contour(x,y,param,clevs,**contour_kwargs)
  else: #assume ptype to default, ie 'scatter'
    s_kwargs.update(kwargs)
    im=m.scatter(x,y,c=param,**s_kwargs)
  

  if cbar:
    if 'vmax' in kwargs and 'vmin' in kwargs:
      import matplotlib as mpl
      norm=mpl.colors.Normalize(vmin=kwargs['vmin'], vmax=kwargs['vmax'])
      c_ax=fig.add_axes([0.95, 0.3, 0.015, 0.5])
      cb = mpl.colorbar.ColorbarBase(c_ax, cmap=cmap,norm=norm)
    else:
      cb = m.colorbar(im,"right", size="2%", pad='2%')
  
  if show:
    plt.show()
  return fig,m

def save_raw(fig_,fn='raw_img.png',shape_ratio=None,dpi=1):
  """
  Save content of figure to file without axes or padding

  Parameters
  ----------
  fig_ : list or tuple
    List with figure in first index. This corresponds to the output 
    of the plotting functions.
  fn : str, optional
    Name of output file (default ``'raw_img.png'``).
  shape_ratio : list or tuple
    width and height of image in relative units (matplotlib's "inches")
    , should be manually set to prevent padding (default ``None``).
  dpi : scalar,optional
    the resolution of the image in dots per inch.

  Examples
  --------
  Printing straight from plot function:

  >>> import swarmtoolkit as st
  >>> st.save_raw(st.plot([0,1,2],[0,2,1]))

  How to retrieve the image as a numpy.ndarray:

  >>> import matplotlib.pyplot as plt
  >>> img_as_array = plt.imread('raw_img.png',interpolation='nearest')

  Save plotted image normally in matplotlib:

  >>> import matplotlib.pyplot as plt 
  >>> plt.savefig('myfilename.png')

  Alternatively `figure` or `axes` object can be used (eg. ``fig.savefig``)

  """
  f = fig_[0]
  f.frameon=False
  if shape_ratio:
    f.set_size_inches(*shape_ratio)
  f.set_tight_layout(True)
  ax = f.gca()
  ax.set_axis_off()

  plt.savefig(fn,dpi=dpi)
  aux.logger.info("Image '{}' created".format(fn))

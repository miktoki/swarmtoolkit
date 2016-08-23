Complementary packages
======================

.. below are some packages that can complement `swarmtoolkit`

Converting between geographic and magnetic coordinates `aacgmv2 <https://github.com/cmeeren/aacgmv2>`_
------------------------------------------------------------------------------------------------------

`aacgmv2` is a *pip*-installable wrapper to the  `AACGM-v2 C library <https://engineering.dartmouth.edu/superdarn/aacgm.html>`_ (Altitude adjusted corrected geomagnetic) with computation required for conversion to the *magnetic local time*. From their `github repository <https://github.com/cmeeren/aacgmv2>`_ (taken 15.07.2016):

Convert between AACGM and geographic coordinates:

::

    >>> from aacgmv2 import convert
    >>> from datetime import date
    >>> # geo to AACGM, single numbers
    >>> mlat, mlon = convert(60, 15, 300, date(2013, 11, 3))
    >>> mlat
    array(57.47207691280528)
    >>> mlon
    array(93.62138045643167)
    >>> # AACGM to geo, mix arrays/numbers
    >>> glat, glon = convert([90, -90], 0, 0, date(2013, 11, 3), a2g=True)
    >>> glat
    array([ 82.96656071, -74.33854592])
    >>> glon
    array([ -84.66516034,  125.84014944])

Convert between AACGM and MLT:

::
    >>> from aacgmv2 import convert_mlt
    >>> from datetime import datetime
    >>> # MLT to AACGM
    >>> mlon = convert_mlt([0, 12], datetime(2013, 11, 3, 18, 0), m2a=True)
    >>> mlon
    array([ 159.10097421,  339.10097421])

Where the 1st and 2nd arguments of `convert` are the latitude and longitudes, the 3rd argument  is the altitude, and the date will default to the current time. Switching between conversion and the inverse conversion is done using the `a2g` parameter which is `False` by default. latitude, longitude and altitude can be arrays or scalars.

`convert_mlt` takes the magnetic logitude(magnetic local time) and a datetime object and returns the magnetic local time(magnetic longitude) given that `m2a` is set to `False`(`True`). Magnetic longitude and magnetic local time can be arrays or scalars.

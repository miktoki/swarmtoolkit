.. _swarmtoolkit_doc:

.. automodule:: swarmtoolkit
    :members:
    
Keyword arguments hierachy:
---------------------------

Several key functions in `swarmtoolkit` pass on keyword arguments to functions they call under the hood. Below is an overview to help keep track of this:

    `getCDFparams`_  -> `getCDFparamlist`_ (if no parameter is provided)

    `getCDFparams`_  -> `getCDFlist`_  -> `dl_ftp`_ (if ``use_ftp=True`` or ``user`` is provided)
 
    `getCDFparams`_  -> `extract_parameter`_  -> `concatenate_values`_

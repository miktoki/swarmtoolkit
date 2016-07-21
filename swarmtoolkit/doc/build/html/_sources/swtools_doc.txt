.. _swtools_doc:

.. automodule:: swtools
    :members:
    
Keyword arguments hierachy:
---------------------------

Several key functions in `swtools` pass on keyword arguments to functions they call under the hood. Below is an overview to help keep track of this:

    `getCDFparams`_  -> `getCDFlist`_  -> `dl_ftp`_
 
    `getCDFparams`_  -> `extract_parameter`_  -> `concatenate_values`_


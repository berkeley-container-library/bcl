.. CatCutifier documentation master file, created by
   sphinx-quickstart on Wed Apr 24 15:19:01 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BCL's documentation!
=======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

:ref:`genindex`

Communication
-------------

.. doxygenstruct:: BCL::GlobalPtr
  :members:

.. doxygenfunction:: BCL::alloc

.. doxygenfunction:: BCL::dealloc

.. doxygenfunction:: BCL::reinterpret_pointer_cast(const GlobalPtr<U>&)

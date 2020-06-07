.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_tutorials_tutorial_08_backend.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_tutorials_tutorial_08_backend.py:


Back-end Support
================

**Author**: Yi-Hsiang Lai (seanlatias@github)

HeteroCL provides multiple back-end supports. Currently, we support both CPU
and FPGA flows. We will be extending to other back ends including ASICs and
PIMs (processing in memory). To set to different back ends, simply set the
``target`` of ``hcl.build`` API. In this tutorial, we will demonstrate how
to target different back ends in HeteroCL. The same program and schedule will
be used throughout the entire tutorial.

.. code-block:: default

    import heterocl as hcl
    import numpy as np

    A = hcl.placeholder((10, 10), "A")
    def kernel(A):
        return hcl.compute((8, 8), lambda y, x: A[y][x] + A[y+2][x+2], "B")
    s = hcl.create_scheme(A, kernel)
    s.downsize(kernel.B, hcl.UInt(4))
    s = hcl.create_schedule_from_scheme(s)
    s.partition(A)
    s[kernel.B].pipeline(kernel.B.axis[1])






CPU
---
CPU is the default back end of a HeteroCL program. If you want to be more
specific, set the ``target`` to be ``llvm``. Note the some customization
primitives are ignored by the CPU back end. For instance, ``partition`` and
``pipeline`` have no effect. Instead, we can use ``parallel``.


.. code-block:: default

    f = hcl.build(s) # equivalent to hcl.build(s, target="llvm")







We can execute the returned function as we demonstrated in other tutorials.


.. code-block:: default

    hcl_A = hcl.asarray(np.random.randint(0, 10, A.shape))
    hcl_B = hcl.asarray(np.zeros((8, 8)), dtype=hcl.UInt(4))
    f(hcl_A, hcl_B)







FPGA
----
For FPGA, we provide several back ends.

Vivado HLS C++ Code Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To generate Vivado HLS code, simply set the target to ``vhls``. Note that
the returned function is a **code** instead of an executable.


.. code-block:: default

    f = hcl.build(s, target="vhls")
    print(f)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    #include <ap_int.h>
    #include <ap_fixed.h>
    #include <math.h>

    void default_function(ap_int<32> A[10][10], ap_uint<4> B[8][8]) {
    #pragma HLS array_partition variable=A complete dim=0
      for (ap_int<32> y = 0; y < 8; ++y) {
        for (ap_int<32> x = 0; x < 8; ++x) {
        #pragma HLS pipeline
          B[y][x] = ((ap_uint<4>)(((ap_int<33>)A[y][x]) + ((ap_int<33>)A[(y + 2)][(x + 2)])));
        }
      }
    }


Vivado HLS C++ Code Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
HeteroCL provides users with the ability to simulation the generated HLS
code directly from the Python interface. To use this feature, you need to
have the Vivado HLS header files in your ``g++`` include path. If this is
the case, then we can set target to ``vhls_csim``, which returns an
**executable**. We can then run it the same as what we do for the CPU back
end.

.. note::

   The Vivado HLS program will not be triggered during the simulation.
   We only need the header files to be in the path.


.. code-block:: default

    import subprocess
    import sys
    proc = subprocess.Popen(
            "g++ -E -Wp,-v -xc++ /dev/null",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            )
    stdout, stderr = proc.communicate()
    if "Vivado_HLS" in str(stderr):
        f = hcl.build(s, target="vhls_csim")
        f(hcl_A, hcl_B)







Intel HLS C++ Code Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
HeteroCL can also generate Intel HLS code. However, due to certain
limitation, some directives cannot be generated. To generate the code, set
the target to ``ihls``.


.. code-block:: default

    f = hcl.build(s, target="ihls")
    print(f)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    #include <HLS/hls.h>
    #include <HLS/ac_int.h>
    #include <HLS/ac_fixed.h>
    #include <HLS/ac_fixed_math.h>
    #include <math.h>

    component void default_function(ac_int<32, true> A[10][10], ac_int<4, false> B[8][8]) {
    #pragma HLS array_partition variable=A complete dim=0
      for (ac_int<32, true> y = 0; y < 8; ++y) {
        #pragma ii 1
        for (ac_int<32, true> x = 0; x < 8; ++x) {
          B[y][x] = ((ac_int<4, false>)(((ac_int<33, true>)A[y][x]) + ((ac_int<33, true>)A[(y + 2)][(x + 2)])));
        }
      }
    }


Merlin C Code Generation
~~~~~~~~~~~~~~~~~~~~~~~~
HeteroCL can generate C code that can be used along with
`Merlin C compiler <https://www.falconcomputing.com/merlin-fpga-compiler/>`_.
The generated Merlin C code has special support for several customization
primitives. For example, the ``unroll`` primitive implies a fine-grained
parallelism, which unroll all sub-loops. The ``parallel`` primitive implies
a coarse-grained parallelism that generates a PE array. Finally, the
``pipeline`` primitive implies a coarse-grained pipeline operation.


.. code-block:: default

    f = hcl.build(s, target="merlinc")
    print(f)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    #include <string.h>
    #include <math.h>
    #include <assert.h>
    #pragma ACCEL kernel
    void default_function(int* A, unsigned char* B) {
      for (int y = 0; y < 8; ++y) {
    #pragma ACCEL pipeline
        for (int x = 0; x < 8; ++x) {
          B[(x + (y * 8))] = ((unsigned char)(((long)A[(x + (y * 10))]) + ((long)A[((x + (y * 10)) + 22)])));
        }
      }
    }


SODA Stencil Code Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
HeteroCL incorporates the SODA framework for efficient stencil architecture
generation. For more details, please refer to
:ref:`sphx_glr_tutorials_tutorial_09_stencil.py`.


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.363 seconds)


.. _sphx_glr_download_tutorials_tutorial_08_backend.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: tutorial_08_backend.py <tutorial_08_backend.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: tutorial_08_backend.ipynb <tutorial_08_backend.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_

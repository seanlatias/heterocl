.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_tutorials_tutorial_01_get_started.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_tutorials_tutorial_01_get_started.py:


Getting Started
===============

**Author**: Yi-Hsiang Lai (seanlatias@github)

In this tutorial, we demonstrate the basic usage of HeteroCL.

Import HeteroCL
---------------
We usually use ``hcl`` as the acronym of HeteroCL.

.. code-block:: default


    import heterocl as hcl







Initialize the Environment
--------------------------
We need to initialize the environment for each HeteroCL application. We can
do this by calling the API ``hcl.init()``. We can also set the default data
type for every computation via this API. The default data type is **32-bit**
integers.

.. note::

   For more information on the data types, please see
   :ref:`sphx_glr_tutorials_tutorial_05_dtype.py`.


.. code-block:: default


    hcl.init()







Algorithm Definition
--------------------
After we initialize, we define the algorithm by using a Python function
definition, where the arguments are the input tensors. The function can
optionally return tensors as outputs. In this example, the two inputs are a
scalar `a` and a tensor `A`, and the output is also a tensor `B`. The main
difference between a scalar and a tensor is that *a scalar cannot be updated*.

Within the algorithm definition, we use HeteroCL APIs to describe the
operations. In this example, we use a tensor-based declarative-style
operation ``hcl.compute``. We also show the equivalent  Python code.

.. note::

   For more information on the APIs, please see
   :ref:`sphx_glr_tutorials_tutorial_03_api.py`


.. code-block:: default


    def simple_compute(a, A):

        B = hcl.compute(A.shape, lambda x, y: A[x, y] + a, "B")
        """
        The above API is equivalent to the following Python code.

        for x in range(0, 10):
            for y in range(0, 10):
                B[x, y] = A[x, y] + a
        """

        return B







Inputs/Outputs Definition
-------------------------
One of the advantages of such *modularized algorithm definition* is that we
can reuse the defined function with different input settings. We use
``hcl.placeholder`` to set the inputs, where we specify the shape, name,
and data type. The shape must be specified and should be in the form of a
**tuple**. If it is empty (i.e., `()`), the returned object is a *scalar*.
Otherwise, the returned object is a *tensor*. The rest two fields are
optional. In this example, we define a scalar input `a` and a
two-dimensional tensor input `A`.

.. note::

   For more information on the interfaces, please see
   :obj:`heterocl.placeholder`


.. code-block:: default


    a = hcl.placeholder((), "a")
    A = hcl.placeholder((10, 10), "A")







Apply Hardware Customization
----------------------------
Usually, our next step is apply various hardware customization techniques to
the application. In this tutorial, we skip this step which will be discussed
in the later tutorials. However, we still need to build a default schedule
by using ``hcl.create_schedule`` whose inputs are a list of inputs and
the Python function that defines the algorithm.


.. code-block:: default


    s = hcl.create_schedule([a, A], simple_compute)







Inspect the Intermediate Representation (IR)
--------------------------------------------
A HeteroCL program will be lowered to an IR before backend code generation.
HeteroCL provides an API for users to inspect the lowered IR. This could be
helpful for debugging.


.. code-block:: default


    print(hcl.lower(s))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    produce B {
      // attr [0] extern_scope = 0
      for (x, 0, 10) {
        for (y, 0, 10) {
          B[(y + (x*10))] = int32((int33(A[(y + (x*10))]) + int33(a)))
        }
      }
    }


Create the Executable
---------------------
The next step is to build the executable by using ``hcl.build``. You can
define the target of the executable, where the default target is `llvm`.
Namely, the executable will be run on CPU. The input for this API is the
schedule we just created.


.. code-block:: default


    f = hcl.build(s)







Prepare the Inputs/Outputs for the Executable
---------------------------------------------
To run the generated executable, we can feed it with Numpy arrays by using
``hcl.asarray``. This API transforms a Numpy array to a HeteroCL container
that is used as inputs/outputs to the executable. In this tutorial, we
randomly generate the values for our input tensor `A`. Note that since we
return a new tensor at the end of our algorithm, we also need to prepare
an input array for tensor `B`.


.. code-block:: default


    import numpy as np

    hcl_a = 10
    np_A = np.random.randint(100, size = A.shape)
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np.zeros(A.shape))







Run the Executable
------------------
With the prepared inputs/outputs, we can finally feed them to our executable.


.. code-block:: default


    f(hcl_a, hcl_A, hcl_B)







View the Results
----------------
To view the results, we can transform the HeteroCL tensors back to Numpy
arrays by using ``asnumpy()``.


.. code-block:: default


    np_A = hcl_A.asnumpy()
    np_B = hcl_B.asnumpy()

    print(hcl_a)
    print(np_A)
    print(np_B)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    10
    [[77 86  7 30 58 82  2 24 32 92]
     [37  4 86 26 79  2 35 26 46 63]
     [11 62 43 83 29 99 14  4 78 92]
     [46 61 44 66 33  6 62 86 63 68]
     [21 56 86 32 44 51 19 15 31  7]
     [62  9 91 75 84 54 36 23 42  0]
     [33 57 30 38 56 74 79 60  0  7]
     [41 35 64 28 31  8 67 43 67 58]
     [ 6 82 63 84 14 26  8 42 69 18]
     [48 51 35  3 64 47 23 50 42 18]]
    [[ 87  96  17  40  68  92  12  34  42 102]
     [ 47  14  96  36  89  12  45  36  56  73]
     [ 21  72  53  93  39 109  24  14  88 102]
     [ 56  71  54  76  43  16  72  96  73  78]
     [ 31  66  96  42  54  61  29  25  41  17]
     [ 72  19 101  85  94  64  46  33  52  10]
     [ 43  67  40  48  66  84  89  70  10  17]
     [ 51  45  74  38  41  18  77  53  77  68]
     [ 16  92  73  94  24  36  18  52  79  28]
     [ 58  61  45  13  74  57  33  60  52  28]]


Let's run some test


.. code-block:: default


    for i in range(10):
        for j in range(10):
            assert np_B[i][j] == np_A[i][j] + 10







.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.105 seconds)


.. _sphx_glr_download_tutorials_tutorial_01_get_started.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: tutorial_01_get_started.py <tutorial_01_get_started.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: tutorial_01_get_started.ipynb <tutorial_01_get_started.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_

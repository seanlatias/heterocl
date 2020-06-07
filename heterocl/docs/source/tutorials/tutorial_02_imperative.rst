.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_tutorials_tutorial_02_imperative.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_tutorials_tutorial_02_imperative.py:


Imperative Programming
======================

**Author**: Yi-Hsiang Lai (seanlatias@github)

There exist many applications that cannot be described using only vectorized
code such as `hcl.compute`. Thus, we introduce imperative programming in
HeteroCL, which makes HeteroCL applications more expressive. In this tutorial,
we will implement *insertion sort* in HeteroCL.

.. code-block:: default


    import heterocl as hcl

    hcl.init()

    A = hcl.placeholder((10,), "A")







Stages in HeteroCL
------------------
In HeteroCL, when users write an application, they are actually building a
compute graph. Each node in a graph is a *stage*. Each edge is directed,
which represents the data flow between two stages. Some HeteroCL APIs
naturally form a stage, such as ``hcl.compute``. Since the imperative code
we are going to write cannot be described using a HeteroCL API, we need to
wrap it as a stage explicitly via ``hcl.Stage``. Users can specify the name
of a stage, which is optional. Note that **a HeteroCL application must have
at least one stage**.


.. code-block:: default


    def insertion_sort(A):

        # Introduce a stage.
        with hcl.Stage("S"):
            # for i in range(1, A.shape[0])
            # We can name the axis
            with hcl.for_(1, A.shape[0], name="i") as i:
                key = hcl.local(A[i], "key")
                j = hcl.local(i-1, "j")
                # while(j >= 0 && key < A[j])
                with hcl.while_(hcl.and_(j >= 0, key < A[j])):
                    A[j+1] = A[j]
                    j[0] -= 1
                A[j+1] = key[0]







Imperative DSL
--------------
To write imperative code in HeteroCL, we need to use a subset of HeteroCL
DSL, which is *imperative DSL*. HeteroCL's imperative DSL supports a subset
of Python's control flow statements, including conditional statements and
control flows. In the above code, we show how we can use ``hcl.for_`` to
write a `for` loop and ``hcl.while_`` to write a `while` loop. Moreover, we
use ``hcl.and_`` for logical expressions. Here we also introduce a new API,
which is ``hcl.local``. It is equivalent to

``hcl.compute((1,))``

Namely, it declares a tensor with exactly one element, which can be treated
as a **stateful scalar**. Following we show the execution results of the
implemented sorting algorithm.

.. note::

   Currently we support the following imperative DSLs. Logic operations:
   :obj:`heterocl.and_`, :obj:`heterocl.or_`. Control flow statements:
   :obj:`heterocl.if_`, :obj:`heterocl.else_`, :obj:`heterocl.elif_`,
   :obj:`heterocl.for_`, :obj:`heterocl.while_`, :obj:`heterocl.break_`.


.. code-block:: default


    s = hcl.create_schedule([A], insertion_sort)







We can inspect the generated IR.


.. code-block:: default

    print(hcl.lower(s))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    // attr [S] storage_scope = "global"
    allocate S[int32 * 1]
    produce S {
      // attr [0] extern_scope = 0
      for (i, 0, 9) {
        // attr [key] storage_scope = "global"
        allocate key[int32 * 1]
        produce key {
          // attr [0] extern_scope = 0
          key[0] = A[(i + 1)]
        }
        // attr [j] storage_scope = "global"
        allocate j[int32 * 1]
        produce j {
          // attr [0] extern_scope = 0
          j[0] = i
        }
        while (((0 <= j[0]) && (key[0] < A[j[0]]))) {
          A[(j[0] + 1)] = A[j[0]]
          j[0] = (j[0] + -1)
        }
        A[(j[0] + 1)] = key[0]
      }
    }


Finally, we build the executable and feed it with Numpy arrays.


.. code-block:: default

    f = hcl.build(s)

    import numpy as np

    hcl_A = hcl.asarray(np.random.randint(50, size=(10,)))

    print('Before sorting:')
    print(hcl_A)

    f(hcl_A)

    print('After sorting:')
    np_A = hcl_A.asnumpy()
    print(np_A)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Before sorting:
    [15  8 48  1 24 15 13  1 20  9]
    After sorting:
    [ 1  1  8  9 13 15 15 20 24 48]


Let's run some tests for verification.


.. code-block:: default

    for i in range(1, 10):
        assert np_A[i] >= np_A[i-1]







Bit Operations
--------------
HeteroCL also support bit operations including setting/getting a bit/slice
from a number. This is useful for integer and fixed-point operations.
Following we show some basic examples.


.. code-block:: default

    hcl.init()
    A = hcl.placeholder((10,), "A")
    def kernel(A):
        # get the LSB of A
        B = hcl.compute(A.shape, lambda x: A[x][0], "B")
        # get the lower 4-bit of A
        C = hcl.compute(A.shape, lambda x: A[x][4:0], "C")
        return B, C







Note that for the slicing operations, we follow the convention of Python,
which is **left exclusive and right inclusive**. Now we can test the results.


.. code-block:: default

    s = hcl.create_schedule(A, kernel)
    f = hcl.build(s)

    np_A = np.random.randint(0, 100, A.shape)
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np.zeros(A.shape))
    hcl_C = hcl.asarray(np.zeros(A.shape))

    f(hcl_A, hcl_B, hcl_C)

    print("Input array:")
    print(hcl_A)
    print("Least-significant bit:")
    print(hcl_B)
    print("Lower four bits:")
    print(hcl_C)

    # a simple test
    np_B = hcl_B.asnumpy()
    np_C = hcl_C.asnumpy()
    for i in range(0, 10):
        assert np_B[i] == np_A[i] % 2
        assert np_C[i] == np_A[i] % 16





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Input array:
    [74 47  2 58 81  6 24 49 24 42]
    Least-significant bit:
    [0 1 0 0 1 0 0 1 0 0]
    Lower four bits:
    [10 15  2 10  1  6  8  1  8 10]


The operations for bit/slice setting is similar. The only difference is that
we need to use imperative DSL. Following is an example.


.. code-block:: default

    hcl.init()
    A = hcl.placeholder((10,), "A")
    B = hcl.placeholder((10,), "B")
    C = hcl.placeholder((10,), "C")
    def kernel(A, B, C):
        with hcl.Stage("S"):
            with hcl.for_(0, 10) as i:
                # set the LSB of B to be the same as A
                B[i][0] = A[i][0]
                # set the lower 4-bit of C
                C[i][4:0] = A[i]

    s = hcl.create_schedule([A, B, C], kernel)
    f = hcl.build(s)
    # note that we intentionally limit the range of A
    np_A = np.random.randint(0, 16, A.shape)
    np_B = np.random.randint(0, 100, A.shape)
    np_C = np.random.randint(0, 100, A.shape)
    hcl_A = hcl.asarray(np_A)
    hcl_B = hcl.asarray(np_B)
    hcl_C = hcl.asarray(np_C)

    f(hcl_A, hcl_B, hcl_C)

    print("Input array:")
    print(hcl_A)
    print("Before setting the least-significant bit:")
    print(np_B)
    print("After:")
    print(hcl_B)
    print("Before setting the lower four bits:")
    print(np_C)
    print("After:")
    print(hcl_C)

    # let's do some checks
    np_B2 = hcl_B.asnumpy()
    np_C2 = hcl_C.asnumpy()
    for i in range(0, 10):
        assert np_B2[i] % 2 == np_A[i] % 2
        assert np_B2[i] // 2 == np_B[i] // 2
        assert np_C2[i] % 16 == np_A[i]
        assert np_C2[i] // 16 == np_C[i] // 16




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Input array:
    [14  8 14  9  9  7  1 11 10 12]
    Before setting the least-significant bit:
    [27 29 64 38 36 78 24 40  9  1]
    After:
    [26 28 64 39 37 79 25 41  8  0]
    Before setting the lower four bits:
    [19 87 32 64 62 90 71  3 14 20]
    After:
    [30 88 46 73 57 87 65 11 10 28]



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.146 seconds)


.. _sphx_glr_download_tutorials_tutorial_02_imperative.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: tutorial_02_imperative.py <tutorial_02_imperative.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: tutorial_02_imperative.ipynb <tutorial_02_imperative.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_

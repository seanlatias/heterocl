.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_tutorials_tutorial_04_compute.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_tutorials_tutorial_04_compute.py:


Compute Customization
=====================

**Author**: Yi-Hsiang Lai (seanlatias@github)

In this tutorial, we will demonstrate how to apply hardware customization to a
HeteroCL application. More specifically, we will focus on compute
customization, including loop transformation and parallelism. We will
introduce several compute customization primitives.

.. code-block:: default


    import heterocl as hcl







Hardware Customization
----------------------
Hardware customization is important in hardware applications. HeteroCL
provides a clean abstraction that can capture different types of hardware
customization. Typical hardware customization includes compute customization,
data type customization, and memory customization. In this tutorial, we will
focus on compute customization. We can categorize compute customization into
two types: loop transformation and parallelism. We will introduce them
respectively. This is also where ``hcl.create_schedule`` comes in. We will
use a single two-stage computation to demonstrate some of the customization
primitives.


.. code-block:: default


    hcl.init()

    A = hcl.placeholder((10, 100), "A")

    def two_stage(A):
        B = hcl.compute(A.shape, lambda x, y: A[x, y] + 1, "B")
        C = hcl.compute(A.shape, lambda x, y: B[x, y] + 1, "C")
        return C

    s = hcl.create_schedule([A], two_stage)
    s_B = two_stage.B







Note that we can get the stage by accessing the properties of the function
that defines the algorithm `two_stage`. To access the stage in this way, you
**need to name the stages**.

This is the generated IR without applying any hardware customization.


.. code-block:: default


    print(hcl.lower(s))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    // attr [B] storage_scope = "global"
    allocate B[int32 * 10 * 100]
    produce B {
      // attr [0] extern_scope = 0
      for (x, 0, 10) {
        for (y, 0, 100) {
          B[(y + (x*100))] = (A[(y + (x*100))] + 1)
        }
      }
    }
    produce C {
      // attr [0] extern_scope = 0
      for (x, 0, 10) {
        for (y, 0, 100) {
          C[(y + (x*100))] = (B[(y + (x*100))] + 1)
        }
      }
    }


We can take a look at the dataflow graph to visualize the relation between
stages.


.. code-block:: default

    try:
        s.dataflow_graph(plot=True)
    except:
        pass




.. image:: /tutorials/images/sphx_glr_tutorial_04_compute_001.png
    :class: sphx-glr-single-img




Loop Transformation
-------------------
Applying loop transformations to our application can potentially increase
the parallelism inside our program. HeteroCL provides several loop
transformation primitives.

``reorder``
~~~~~~~~~~~
The first primitive we introduce here is loop reordering. With this primitive,
we can redefine the order of a loop nest. For example,


.. code-block:: default


    s[s_B].reorder(s_B.axis[1], s_B.axis[0])







To apply a compute customization primitive, we need to use the schedule
we created. We can also access the axis of a stage by its index. In this
example, `s_B.axis[0]` refers to axis `x`. Similarly, `s_B.axis[1]` refers
to axis `y`. We can take a look at the generated IR.


.. code-block:: default


    print(hcl.lower(s))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    // attr [B] storage_scope = "global"
    allocate B[int32 * 10 * 100]
    produce B {
      // attr [0] extern_scope = 0
      for (y, 0, 100) {
        for (x, 0, 10) {
          B[(y + (x*100))] = (A[(y + (x*100))] + 1)
        }
      }
    }
    produce C {
      // attr [0] extern_scope = 0
      for (x, 0, 10) {
        for (y, 0, 100) {
          C[(y + (x*100))] = (B[(y + (x*100))] + 1)
        }
      }
    }


We can see that axis `x` and axis `y` are indeed reordered.

``split``
~~~~~~~~~
This primitive allows users a to split an axis with a given factor. Namely,
a loop will be split into two sub-loops. For example,


.. code-block:: default


    s = hcl.create_schedule([A], two_stage)
    s_B = two_stage.B
    x_out, x_in = s[s_B].split(s_B.axis[0], 5)







Here we recreate a new schedule so that we will not confuse it with the
previous schedule. We can see that, with the ``hcl.split`` primitive, we get
two new axes `x_out` and `x_in`. To make it clear, let's take a look at the
generated IR.


.. code-block:: default


    print(hcl.lower(s))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    // attr [B] storage_scope = "global"
    allocate B[int32 * 10 * 100]
    produce B {
      // attr [0] extern_scope = 0
      for (x.outer, 0, 2) {
        for (x.inner, 0, 5) {
          for (y, 0, 100) {
            B[(y + (((x.outer*5) + x.inner)*100))] = (A[(y + (((x.outer*5) + x.inner)*100))] + 1)
          }
        }
      }
    }
    produce C {
      // attr [0] extern_scope = 0
      for (x, 0, 10) {
        for (y, 0, 100) {
          C[(y + (x*100))] = (B[(y + (x*100))] + 1)
        }
      }
    }


The returned variable `x_out` corresponds to the axis `x.outer` in the IR.
Since we split the axis with a factor 5, now the outer loop only iterates
two times with the inner loop iterating from 0 to 5. We can further combine
the `reorder` primitive we just introduced.


.. code-block:: default


    s[s_B].reorder(s_B.axis[1], x_out, x_in)

    print(hcl.lower(s))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    // attr [B] storage_scope = "global"
    allocate B[int32 * 10 * 100]
    produce B {
      // attr [0] extern_scope = 0
      for (y, 0, 100) {
        for (x.outer, 0, 2) {
          for (x.inner, 0, 5) {
            B[(y + (((x.outer*5) + x.inner)*100))] = (A[(y + (((x.outer*5) + x.inner)*100))] + 1)
          }
        }
      }
    }
    produce C {
      // attr [0] extern_scope = 0
      for (x, 0, 10) {
        for (y, 0, 100) {
          C[(y + (x*100))] = (B[(y + (x*100))] + 1)
        }
      }
    }


In the generated IR, we can see that the three axes are reordered according
to what we specified.

``fuse``
~~~~~~~~
This primitives is the reversed version of ``hcl.split``. Namely, we can
fuse **two consecutive** sub-loops into a single loop.


.. code-block:: default


    s = hcl.create_schedule([A], two_stage)
    s_B = two_stage.B
    x_y = s[s_B].fuse(s_B.axis[0], s_B.axis[1])

    print(hcl.lower(s))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    // attr [B] storage_scope = "global"
    allocate B[int32 * 10 * 100]
    produce B {
      // attr [0] extern_scope = 0
      for (x.y.fused, 0, 1000) {
        B[x.y.fused] = (A[x.y.fused] + 1)
      }
    }
    produce C {
      // attr [0] extern_scope = 0
      for (x, 0, 10) {
        for (y, 0, 100) {
          C[(y + (x*100))] = (B[(y + (x*100))] + 1)
        }
      }
    }


Similar to the previous example, we recreate a new schedule. Here we fuse
the two axes `x` and `y` into a single axis `x_y`, which corresponds to
`x.y.fused` in the generated IR. Now the loop iterates from 0 to 1000, as
expected.

``compute_at``
~~~~~~~~~~~~~~
Previously, we focus on the loop transformation within one stage. However,
we can also perform loop transformations across multi-stages. This primitive
allows users to merge the loops from two stages. The idea behind it is to
compute a stage within another stage so that we can reuse some partial
results.


.. code-block:: default


    s = hcl.create_schedule([A], two_stage)
    s_B = two_stage.B
    s_C = two_stage.C
    s[s_B].compute_at(s[s_C], s_C.axis[0])







In this example, we specify stage B to be computed within stage C at the
first axis `x`. Originally, we first completely compute stage B and then
stage C. However, in this scenario, after we finish the computation of
stage B axis `y`, we do not continue on computing the next `x`. Instead,
we go on to compute stage C axis `y`. It would be easier to understand with
the generated IR.


.. code-block:: default


    print(hcl.lower(s))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    produce C {
      // attr [0] extern_scope = 0
      for (x, 0, 10) {
        // attr [B] storage_scope = "global"
        allocate B[int32 * 1 * 100]
        produce B {
          // attr [0] extern_scope = 0
          for (y, 0, 100) {
            B[y] = (A[(y + (x*100))] + 1)
          }
        }
        for (y, 0, 100) {
          C[(y + (x*100))] = (B[y] + 1)
        }
      }
    }


We can observe from the IR that now both stages share the same outer loop
`x`. Moreover, we only need to allocate the memory for partial results.

Parallelism
-----------
In addition to loop transformations, we can also explore the parallelism
within an applications. In this category, normally we just annotate the
loop and the backend code generator will handle the rest. Thus, we do not
explain each parallelism primitive one by one. The primitives we support
include ``unroll``, ``parallel``, and ``pipeline``.

Combine All Together
--------------------
Finally, we can combine different compute customization primitives together.


.. code-block:: default


    s = hcl.create_schedule([A], two_stage)
    s_B = two_stage.B
    s_C = two_stage.C

    s[s_B].reorder(s_B.axis[1], s_B.axis[0])
    s[s_C].reorder(s_C.axis[1], s_C.axis[0])
    s[s_B].compute_at(s[s_C], s_C.axis[0])
    s[s_C].parallel(s_C.axis[1])
    s[s_C].pipeline(s_C.axis[0])

    print(hcl.lower(s))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    produce C {
      // attr [0] extern_scope = 0
      parallel (y, 0, 100) {
        pipelined "initiation_interval"=1 (x, 0, 10) {
          // attr [B] storage_scope = "global"
          allocate B[int32 * 1 * 1]
          produce B {
            // attr [0] extern_scope = 0
            B[0] = (A[(y + (x*100))] + 1)
          }
          C[(y + (x*100))] = (B[0] + 1)
        }
      }
    }


Apply to Imperative DSL
-----------------------
HeteroCL also lets users to apply these primitives to imperative DSLs. In
other words, all the loops written with ``hcl.for_`` can be applied. To do
that, we also need to name those axes.


.. code-block:: default


    hcl.init()

    A = hcl.placeholder((10,))

    def custom_imperative(A):
        with hcl.Stage("S"):
            with hcl.for_(0, 10, name="i") as i:
                A[i] = i - 10

    s = hcl.create_schedule([A], custom_imperative)
    s_S = custom_imperative.S
    i_out, i_in = s[s_S].split(s_S.i, 2)

    print(hcl.lower(s))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    // attr [S] storage_scope = "global"
    allocate S[int32 * 1]
    produce S {
      // attr [0] extern_scope = 0
      for (i.outer, 0, 5) {
        for (i.inner, 0, 2) {
          placeholder8[((i.outer*2) + i.inner)] = (((i.outer*2) + i.inner) + -10)
        }
      }
    }


We can also access the imperative axes with their showing up order.


.. code-block:: default


    assert(s_S.i == s_S.axis[0])







.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.311 seconds)


.. _sphx_glr_download_tutorials_tutorial_04_compute.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: tutorial_04_compute.py <tutorial_04_compute.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: tutorial_04_compute.ipynb <tutorial_04_compute.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_

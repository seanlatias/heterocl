.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_samples_kmeans_kmeans_main.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_samples_kmeans_kmeans_main.py:


HeteroCL Tutorial : K-means Clustering Algorithm
================================================

**Author**: Yi-Hsiang Lai (seanlatias@github), Ziyan Feng

This is the K-means clustering algorithm written in Heterocl.

.. code-block:: default

    import numpy as np
    import heterocl as hcl
    import time
    import random






Define the number of the clustering means as K, the number of points as N,
the number of dimensions as dim, and the number of iterations as niter


.. code-block:: default

    K = 16
    N = 320
    dim = 32
    niter = 200

    hcl.init()







Main Algorithm
==============


.. code-block:: default

    def top(target=None):
        points = hcl.placeholder((N, dim))
        means = hcl.placeholder((K, dim))

        def kmeans(points, means):
            def loop_kernel(labels):
                # assign cluster
                with hcl.for_(0, N, name="N") as n:
                    min_dist = hcl.local(100000)
                    with hcl.for_(0, K) as k:
                        dist = hcl.local(0)
                        with hcl.for_(0, dim) as d:
                            dist_ = points[n, d]-means[k, d]
                            dist[0] += dist_ * dist_
                        with hcl.if_(dist[0] < min_dist[0]):
                            min_dist[0] = dist[0]
                            labels[n] = k
                # update mean
                num_k = hcl.compute((K,), lambda x: 0)
                sum_k = hcl.compute((K, dim), lambda x, y: 0)
                def calc_sum(n):
                    num_k[labels[n]] += 1
                    with hcl.for_(0, dim) as d:
                        sum_k[labels[n], d] += points[n, d]
                hcl.mutate((N,), lambda n: calc_sum(n), "calc_sum")
                hcl.update(means,
                        lambda k, d: sum_k[k, d]//num_k[k], "update_mean")

            labels = hcl.compute((N,), lambda x: 0)
            hcl.mutate((niter,), lambda _: loop_kernel(labels), "main_loop")
            return labels

        # create schedule and apply compute customization
        s = hcl.create_schedule([points, means], kmeans)
        main_loop = kmeans.main_loop
        update_mean = main_loop.update_mean
        s[main_loop].pipeline(main_loop.N)
        s[main_loop.calc_sum].unroll(main_loop.calc_sum.axis[0])
        fused = s[update_mean].fuse(update_mean.axis[0], update_mean.axis[1])
        s[update_mean].unroll(fused)
        return hcl.build(s, target=target)

    f = top()

    points_np = np.random.randint(100, size=(N, dim))
    labels_np = np.zeros(N)
    means_np = points_np[random.sample(range(N), K), :]

    hcl_points = hcl.asarray(points_np, dtype=hcl.Int())
    hcl_means = hcl.asarray(means_np, dtype=hcl.Int())
    hcl_labels = hcl.asarray(labels_np)

    start = time.time()
    f(hcl_points, hcl_means, hcl_labels)
    total_time = time.time() - start
    print("Kernel time (s): {:.2f}".format(total_time))

    print("All points:")
    print(hcl_points)
    print("Final cluster:")
    print(hcl_labels)
    print("The means:")
    print(hcl_means)

    from kmeans_golden import kmeans_golden
    kmeans_golden(niter, K, N, dim, np.concatenate((points_np,
        np.expand_dims(labels_np, axis=1)), axis=1), means_np)
    assert np.allclose(hcl_means.asnumpy(), means_np)




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Kernel time (s): 0.06
    All points:
    [[55 99 33 ... 23 65 11]
     [99 20 95 ... 94 85 51]
     [31 59 50 ...  2 89 13]
     ...
     [99 58 50 ... 45 50 30]
     [ 8 15 35 ... 44 95 28]
     [90 87 29 ... 18 32  1]]
    Final cluster:
    [ 1  6  6 12  3  8  1 11 12  9 14  5 15  5 15  0 10  0 13  4  9  3  6  3
      9  6  3  3  1 14 10  1  0 11  6 12  3  0  9  6 11  5 10  2 13  2 12  9
      1  2  6 12  4 15 11 13 10  2 11  1  7  3 14 15  4  0 12  7 11  2 15  4
      5  5  1 10 14  4  7 13  3 15 15  6 12  0  3  1  4  6 11 15  6  4  2 10
     15  0  2  3  6 12  0  2 15  7 15  0  4  2 15  0  8 12 14 10 15 12 10  9
     15  0 12  9  4  6 15  8  0  6  2 11  1  6  5  6 13  6 12  8 11 10 13  7
      0 11  5  4  7 10  7  3  8 11  3  6  2  1 10  5 11  1  9 15  7  0 11  0
      9  3  5  0  6 11  2  8 14  7 15 12 12 12 11  4  1 13  1  1  7  0  8 13
     13  8  5  6  0  4 12  7  7  4  2 13  0 11  5 14  1  7  4 14  9  1  0 11
     15  0  3 12  2  4  6  6  4 10  6  0  7  4  2  8 15  6  8  5  8  4  8 15
      3  9  1 14  3  7 11 11 12  6 12  1  4 13  9 15  1 10 12 10  7 13  1 13
     11  4 15  1 14  3 15  2  5  0 13  3 12  5  7 15 10  1  6 15 15 13  6 10
     12 11  3 11  6  4  0  0  8  0  1 13  6 14  0  7 14 15  5  7 11 12 12 11
     11  0 13  6  3  1 10  5]
    The means:
    [[25 47 45 39 58 65 42 30 40 37 70 44 42 37 55 49 58 37 37 54 28 54 48 60
      44 59 68 37 46 35 38 41]
     [76 42 65 32 62 58 58 53 65 59 47 53 58 27 45 42 38 46 45 51 51 53 49 41
      41 41 48 50 65 33 55 37]
     [62 44 40 61 27 37 51 51 48 24 26 62 20 52 39 67 68 25 37 37 46 54 53 35
      43 39 36 57 50 52 35 32]
     [67 36 58 62 36 47 54 64 40 61 48 38 39 34 65 65 50 26 62 38 34 32 28 48
      45 78 30 62 51 51 42 61]
     [54 24 22 46 59 63 46 40 77 51 41 47 30 49 19 66 44 45 56 53 34 44 27 54
      45 32 63 43 30 43 43 45]
     [53 53 35 23 48 33 25 50 67 63 49 69 63 68 52 68 56 46 63 33 28 49 73 57
      61 37 59 52 28 66 44 34]
     [44 49 61 44 47 33 63 49 44 32 46 55 47 67 46 40 52 54 69 62 36 30 64 56
      39 47 57 40 59 43 77 54]
     [39 61 64 23 63 75 62 61 63 47 36 58 46 34 46 58 56 60 37 42 64 63 54 53
      74 70 23 28 58 40 32 39]
     [30 51 37 32 31 24 65 58 75 54 72 54 35 51 63 34 68 56 58 62 61 62 45 40
      51 48 39 61 60 51 21 33]
     [26 39 33 45 39 35 39 59 14 52 55 28 40 65 33 24 70 49 56 46 51 30 23 62
      52 63 44 76 22 41 70 76]
     [53 75 44 63 69 44 70 54 37 74 68 45 53 42 41 32 29 63 34 51 66 59 43 61
      71 37 69 68 60 58 49 34]
     [48 55 42 67 44 69 50 36 47 62 36 60 57 62 62 37 61 54 55 25 50 49 29 34
      78 38 55 43 45 51 54 44]
     [60 39 52 65 23 47 38 48 42 53 65 45 32 52 47 48 23 62 57 44 75 24 64 64
      53 62 48 36 32 70 50 44]
     [61 35 68 35 42 51 47 68 41 41 55 62 66 81 37 43 49 65 45 50 40 76 61 45
      37 75 61 64 53 40 37 68]
     [53 45 50 54 39 53 22 49 45 66 51 61 46 23 43 63 25 69 75 33 35 48 72 20
      48 49 35 60 37 29 33 38]
     [37 44 80 40 63 27 37 59 43 31 56 44 51 46 62 50 36 38 31 50 67 57 50 33
      55 29 47 45 39 46 64 67]]



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  46.713 seconds)


.. _sphx_glr_download_samples_kmeans_kmeans_main.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: kmeans_main.py <kmeans_main.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: kmeans_main.ipynb <kmeans_main.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_

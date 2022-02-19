# Quasiseparable matrices in JAX

A project to extend the [celerite](celerite) Gaussian Process method by building
on the literature studying "quasiseparable plus diagonal" matrices. An intro can
be found in [Matrix Computations and Semiseparable Matrices: Linear
Systems](text) (I got it from the NY public library), and so far the algorithms
are mostly based on [Eidelman & Gohberg (1999)](new-class) and [Foreman-Mackey
et al. (2017)](celerite). An interesting reference to consider for potentially
more efficient algorithms is [Pernet & Storjohann (2018)](efficient).

[celerite]: https://arxiv.org/abs/1703.09710
[text]: https://muse.jhu.edu/book/16537
[new-class]: https://link.springer.com/article/10.1007%2FBF01300581
[efficient]: https://arxiv.org/abs/1701.00396

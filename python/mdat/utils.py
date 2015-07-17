"""Utility functions """

import numpy as np

def form_D(n):
    """
    Form a first-difference matrix

    Parameters
    ----------
    n : int
        The number of entries in the vector to be differenced.

    Returns
    -------
    D : (n-1, n) array
       With -1 on the diagonal and 1 on the supradigonal. When applied to a
       vector, produces the first difference vector of the data.
    """
    D = np.zeros((n-1, n))
    for i in range(n-1):
        D[i, i] = -1
        D[i, i+1] = 1
    return D


def form_DxInv(x, order):
    """

    Parameters
    ----------
    x : 1D array
       Sampling points on the relevant x axis (e.g. age, points along the
       tracks, etc.). Assumed to be ordered
    order : int
       The order of differentiation

    Returns
    -------
    D :
       Diagonal matrix that has 1/spacing between order-spaced points
    """
    if not np.all(np.sort(x) == x):
        msg = "Please order the input x, before passing to DxInv"
        raise ValueError(msg)

    m = x.shape[0] - order
    D = np.zeros((m, m))
    for i in range(m):
        D[i, i] = 1/(x[i+order] - x[i])
    return D

#
# ## Averages observations with the same age, and returns a new X vector, and weights corresponding to number of terms averaged
#
# function combine(ages, X)
#     (n,p) = size(X)
#     unique_ages = unique(ages)
#     weights = Float32[]
#     new_X = Array(Float32,0,p)
#     for age in unique_ages
#         push!(weights, sum(age.==ages))
#         new_X = vcat(new_X, mean(X[find(ages.==age),:],1))
#     end
#     return {"X"=>new_X, "w"=>weights, "age"=>unique_ages}
# end
#

def combine_by_x(x, Y):
    """
    Combine multiple observations with the same sampling values.

    Parameters
    ----------
    Y (n, p) array
       Where n is the number of subjects and p is the number
       of observations tracts.

    Returns
    -------
    average_y :
    weights :
    unique_x :
    """
    unique_x = np.unique(x)
    weights = np.zeros(unique_x.shape[0])
    # We'll reduce Y down to less rows:
    new_Y = np.zeros((unique_x.shape[0], Y.shape[1]))
    for i, u in enumerate(unique_x):
        idx = np.isclose(x, u)
        new_Y[i] = np.mean(Y[idx], -1)
        weights[i] = np.sum(idx)

    return new_Y, weights, unique_x


# function fitter(X, D, D0, lam1, lam2)
#     (n, p) = size(X)
#     b = Variable(p,n)
#     problem = minimize(sum_squares(b - X') + lam1*norm(D*b,1) + lam2*norm(D0*(b'),1))
#     out = solve!(problem, SCSSolver(max_iters = 10000, normalize = 0))
#     return b.value
# end
#
# ## Fits a 2-d smoother. D determines smoothness along columns, D0 determines smoothness wrt a covariate (age in our case)
# ## Also takes in a weight vector (w), used because we have averaged observations with the same age
#
# function fitter_group(X, w, D, D0, lam1, lam2)
#     (n, p) = size(X)
#     b = Variable(p,n)
#     problem = minimize(quad_form(b' - X, diagm(vec(w))) + lam1*norm(D*b,1) + lam2*norm(D0*(b'),1))
#     out = solve!(problem, SCSSolver(max_iters = 10000, normalize = 0))
#     return b.value
# end
#
# ## Fits a 1-d smoother + linear model in age. D determines smoothness along columns
# ## Also takes in a weight vector (w), used because we have averaged observations with the same age
#
#
# function fitter_linear(X, age, w, D, lam1, lam2)
#     (n, p) = size(X)
#     b = Variable(1,p)
#     beta = Variable(1,p)
#     problem = minimize(quad_form(ones(n,1) * b + age * beta - X, diagm(vec(w))) + lam1*norm(D*b',1) + lam2*norm((beta),1))
#     out = solve!(problem, SCSSolver(max_iters = 100000, normalize = 0))
#     return {"b"=>b.value, "beta"=>beta.value}
# end
#

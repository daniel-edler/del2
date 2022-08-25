#!/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################################
##
## Copyright (C) 2000-2021 The Octave Project Developers (Octave version)
## Copyright (C) 2021 Daniel Edler (Python version)
##
## This file was part of Octave and was ported to python. If porting code
## to another language means keeping the copyright then this file is licensed
## under the GNU General Public License, see <https://www.gnu.org/licenses/>.
## I am almost certain that this is the case.
##
########################################################################
import copy
import numpy as np

## GNU Octave-like `shiftdim` function
## NOTE: clean room reimplementation
def shiftdim(X, N):
    ## Remove all unnecessary "1" dimensions "on the right"
    ## Such that:
    ##    shape(X) = (3,1,1,1) => shape(X') = (3,1)
    ##    shape(X) = (1,1,1,1) => shape(X') = (1,1)
    ##    shape(X) = (3,2,1,1) => shape(X') = (3,2)
    if 1 in X.shape:
        if len(X.shape) > 2 and X.shape[-1] == 1:
            X = X.reshape(X.shape[:-1])
            X = shiftdim(X, N)

    if N == 0:
        return X
    elif N < 0:
        if len(X.shape) == 2 and X.shape[1] == 1:
            # We have a column vector
            X = X.reshape(X.shape[::-1])

            if N == -1:
                return X
            else:
                N += 1

        ## N times the "1" then the shape of the input
        return X.reshape(*(1,)*abs(N), *X.shape)

    elif N > 0:
        # TODO Maybe not working. But won't be used in this case anyway
        return np.moveaxis(X, np.arange(X.ndim), np.arange(X.ndim)-(N%len(X)))

## GNU Octave-like `repmat`
## NOTE: clean room reimplementation
def repmat(A, M):
    m = np.ones(M)
    if A.ndim < m.ndim:
        ndimdiff = m.ndim - A.ndim
        a = A.reshape(*A.shape, *(1,)*ndimdiff)
    else:
        a = A

    return np.kron(m, a)

## GNU Octave-like `del2`
## NOTE: ported from Octave
def del2(M, *varargin, useOctavePrefactor=True):
    """
    Calculate the discrete Laplace

    For a 2-dimensional matrix `M` this is defined as (when `useOctavePrefactor` is true)
    $$L = {1 \over 4} \left( {d^2 \over dx^2} M(x,y) + {d^2 \over dy^2} M(x,y) \right)$$

          1    / d^2            d^2         \
    L  = --- * | ---  M(x,y) +  ---  M(x,y) |
          4    \ dx^2           dy^2        /

    The 1/4 prefactor is generalized to 1/(2d) where d>1 is the dimensionality of the system.
    For a one dimensional system it is still 1/4.

    Using useOctavePrefactor=False sets the prefactor to 1.

    For N-dimensional arrays the sum in parentheses is expanded to include
    second derivatives over the additional higher dimensions.

    The spacing between evaluation points may be defined by `h`, which is a
    scalar defining the equidistant spacing in all dimensions.  Alternatively,
    the spacing in each dimension may be defined separately by `dx`,
    `dy`, etc.  A scalar spacing argument defines equidistant spacing,
    whereas a vector argument can be used to specify variable spacing.  The
    length of the spacing vectors must match the respective dimension of
    `M`.  The default spacing value is 1.

    Dimensions with fewer than 3 data points are skipped.  Boundary points are
    calculated from the linear extrapolation of interior points.
    """
    M = np.asarray(M)
    nd = M.ndim
    sz = M.shape
    dx = [None] * nd
    if len(varargin) == 0:
        if nd == 1:
            dx[0] = np.ones(sz[0])
        else:
            for i in range(nd):
                dx[i] = np.ones((sz[i], 1))
    elif len(varargin) == 1 and np.isscalar(varargin[0]):
        h = varargin[0]
        if nd == 1:
            dx[0] = h * np.ones(sz[0])
        else:
            for i in range(nd):
                dx[i] = h * np.ones((sz[i], 1))
    elif len(varargin) <= nd:
        ndx = len(varargin)
        varargin = list(varargin) + [1] * (nd - ndx) # Fill missing dims with 1.
        ## Reverse dx[0] and dx[1] as the X-dim is the 2nd dim of a meshgrid array
        varargin[0], varargin[1] = varargin[1], varargin[0]
        for i in range(nd):
            arg = varargin[i]
            if np.isscalar(arg):
                dx[i] = arg * np.ones((sz[i], 1))
            elif len(arg.shape) == 1 or (len(arg.shape) == 2 and arg.shape[1] == 1):
                if arg.size != sz[i]:
                    print("ERROR: del2: number of elements in spacing vector {}"
                          "does not match dimension {} of M".format(i, i))
                dx[i] = np.array([np.diff(varargin[i])]).T
            else:
                print("ERROR: del2: spacing element {} must be a scalar or vector".format(i))
    else:
        print("ERROR: del2: Please read the doc")
        print(del2.__doc__)

    idx = [ slice(sz[i]) for i in range(nd) ]

    L = np.zeros(sz)
    for i in range(nd):
        if sz[i] >= 3:
            DD = np.zeros(sz)
            idx1 = copy.deepcopy(idx)
            idx2 = copy.deepcopy(idx)
            idx3 = copy.deepcopy(idx)

            ## interior points
            idx1[i] = slice(0, sz[i] - 2)
            idx2[i] = slice(1, sz[i] - 1)
            idx3[i] = slice(2, sz[i])
            # Using a list instead of tuple for indexing raises a FutureWarning
            iidx1, iidx2, iidx3 = tuple(idx1), tuple(idx2), tuple(idx3)

            szi = np.copy(sz) # Deep copy
            szi[i] = 1        # Does not change sz[i] because of the deep copy

            if nd == 1:
                h1 = dx[i][0:-2]
                h2 = dx[i][1:-1]
            else:
                h1 = repmat(shiftdim(dx[i][0:-2], -i), szi)
                h2 = repmat(shiftdim(dx[i][1:-1], -i), szi)

            DD[iidx2] = ((M[iidx1]-M[iidx2])/h1 + (M[iidx3]-M[iidx2])/h2) / (h1 + h2)

            ## left and right boundary
            if sz[i] == 3:
                DD[iidx1] = DD[iidx3] = DD[iidx2]
            else:
                ## "left"
                idx1[i], idx2[i], idx3[i] = 0, 1, 2
                bidx1, bidx2, bidx3 = tuple(idx1), tuple(idx2), tuple(idx3)
                DD[bidx1] = (dx[i][0] + dx[i][1])/dx[i][1] * DD[bidx2] - dx[i][0]/dx[i][1] * DD[bidx3]

                ## "right"
                idx1[i], idx2[i], idx3[i] = -1, -2, -3
                bidx1, bidx2, bidx3 = tuple(idx1), tuple(idx2), tuple(idx3)
                DD[bidx1] = (dx[i][-2] + dx[i][-3])/dx[i][-3] * DD[bidx2] - dx[i][-2]/dx[i][-3] * DD[bidx3]

            L += DD

    if useOctavePrefactor:
        if nd == 1:
            return L/2
        else:
            return L/nd
    else:
        return 2*L

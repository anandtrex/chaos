from numpy.random import permutation
from scipy import rand, randn, ones
from scipy.sparse import csr_matrix

def _rand_sparse(m, n, density):
    # check parameters here
    if density > 1.0 or density < 0.0:
        raise ValueError('density should be between 0 and 1')
    # More checks?
    # Here I use the algorithm suggested by David to avoid ending
    # up with less than m*n*density nonzero elements (with the algorithm
    # provided by Nathan there is a nonzero probability of having duplicate
    # rwo/col pairs).
    nnz = max( min( int(m*n*density), m*n), 0)
    rand_seq = permutation(m*n)[:nnz]
    row  = rand_seq / n
    col  = rand_seq % n
    data = ones(nnz, dtype='int8')
    # duplicate (i,j) entries will be summed together
    return csr_matrix( (data,(row,col)), shape=(m,n) )

def sprand(m, n, density):
    """Build a sparse uniformly distributed random matrix

       Parameters
       ----------

       m, n     : dimensions of the result (rows, columns)
       density  : fraction of nonzero entries.


       Example
       -------

       >>> from scipy.sparse import sprand
       >>> print sprand(2, 3, 0.5).todense()
       matrix[[ 0.5724829   0.          0.92891214]
             [ 0.          0.07712993  0.        ]]

    """
    A = _rand_sparse(m, n, density)
    A.data = rand(A.nnz)
    return A

def sprandn(m, n, density):
    """Build a sparse normally distributed random matrix

       Parameters
       ----------

       m, n     : dimensions of the result (rows, columns)
       density  : fraction of nonzero entries.


       Example
       -------

       >>> from scipy.sparse import sprandn
       >>> print sprandn(2, 4, 0.5).todense()
       matrix([[-0.84041995,  0.        ,  0.        , -0.22398594],
               [-0.664707  ,  0.        ,  0.        , -0.06084135]])


    """
    A = _rand_sparse(m, n, density)
    A.data = randn(A.nnz)
    return A

if __name__ == '__main__':
    print sprand(2, 3, 0.5).todense()
    print sprandn(2, 5, 0.2).todense()
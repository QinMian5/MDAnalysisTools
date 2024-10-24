# Author: Mian Qin
# Date Created: 2/2/24
import numpy as np
from scipy.optimize import minimize, newton
from autograd import value_and_grad, grad, hessian
import autograd.numpy as agnp


class ConvergenceError(Exception):
    """Exception raised for errors in the convergence of an iterative method.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="The iterative method did not converge."):
        self.message = message
        super().__init__(self.message)


def self_consistency(f, x0, max_iter=10000, tol=1e-10, normalize=True, iprint=-1):
    """
        Perform a self-consistency iteration to find a fixed point of a function.

        :param f: The function for which to find a fixed point.
        :param x0: The initial guess for the fixed point.
        :param max_iter: Maximum number of iterations to perform (default is 10000).
        :param tol: Tolerance for the convergence criterion.
                    The iteration stops when the change in x is less than this value.
        :param normalize: Boolean flag to determine if the function's output should be normalized
                          by subtracting its mean at each iteration.
        :param iprint: Controls the printing of the convergence process. If >=0, prints a message upon convergence.
                       If >0, also prints the iteration number and the difference at intervals specified by `iprint`.
        :return: The fixed point found, if the method converges within the given number of iterations and tolerance.
        """
    x = x0
    for i in range(max_iter):
        x_new = f(x)

        # In case of shifting
        if normalize:
            x_new = x_new - x_new.mean()

        diff = np.linalg.norm(x_new - x)
        if diff <= tol:
            if iprint >= 0:
                print(f"Self-consistency method converged at iteration {i}.")
            return x_new
        if iprint > 0:
            if i % iprint == 0:
                print(f"Iteration {i}, diff: {diff:.4}.")
    else:
        raise ConvergenceError(f"Self-consistency method failed to converge after {max_iter} iterations.")


def LBFGSB(f, x0, args=(), iprint=-1):
    """
    Minimize a given function using the L-BFGS-B algorithm, a quasi-Newton method for optimization.

    :param f: The objective function to be minimized.
    :param x0: The initial guess for the variables to be optimized.
    :param args: Extra arguments passed to f
    :param iprint: Controls the verbosity of the optimization process. A value of -1 suppresses output,
                   while higher values increase the verbosity of the output.
    :return: The optimized variable values if the method converges.
    """
    result = minimize(value_and_grad(f), x0, args=args, method='L-BFGS-B', jac=True, options={"iprint": iprint})
    if result.success:
        if iprint >= 0:
            print("L-BFGS-B method converged.")
        return result.x
    else:
        raise ConvergenceError("L-BFGS-B method failed to converge.")


def newton_raphson(f, x0, args=(), iprint=-1,):
    result = minimize(value_and_grad(f), x0, args=args, method="Newton-CG", jac=True, hess=hessian(f))
    if result.success:
        if iprint >= 0:
            print("Newton-CG method converged.")
        return result.x
    else:
        raise ConvergenceError("Newton-CG method failed to converge.")


def alogsumexp(a, b=None, axis=None, keepdims=False):
    """
    Performs logsumexp using autograd.numpy
    np.log(np.sum(a*np.exp(b)))

    Args:
        a(np.ndarray): The matrix/vector to be exponentiated  (shape (N,...))
        b(np.ndarray): The number at which to multiply exp(a) (shape (N,)) (default None)
        axis(int): the axis at which to sum over (defaul None)
        keepdims(bool): whether to keep the result as the same shape (default False)

    Return:
        a matrix that is the logsumexp result of a & b
    """
    if b is not None:
        if agnp.any(b == 0):
            a = a + 0.  # promote to at least float
            a[b == 0] = -agnp.inf

    # find maximum of a along the axis provided
    a_max = agnp.amax(a, axis=axis, keepdims=True)

    if b is not None:
        b = agnp.asarray(b)
        tmp = b * agnp.exp(a - a_max)
    else:
        tmp = agnp.exp(a - a_max)

    # suppress warnings about log of zero
    with agnp.errstate(divide='ignore'):
        s = agnp.sum(tmp, axis=axis, keepdims=keepdims)

    out = agnp.log(s)

    if not keepdims:
        a_max = agnp.squeeze(a_max, axis=axis)

    out += a_max

    return out


def main():
    ...


if __name__ == "__main__":
    main()

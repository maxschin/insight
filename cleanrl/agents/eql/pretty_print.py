"""
Generate a mathematical expression of the symbolic regression network (AKA EQL network) using SymPy. This expression
can be used to pretty-print the expression (including human-readable text, LaTeX, etc.). SymPy also allows algebraic
manipulation of the expression.
The main function is network(...)
There are several filtering functions to simplify expressions, although these are not always needed if the weight matrix
is already pruned.
"""

import sympy as sym
from concurrent.futures import ProcessPoolExecutor
from . import functions


def apply_activation(W, funcs, n_double=0):
    """Given an (n, m) matrix W and (m) vector of funcs, apply funcs to W.

    Arguments:
        W:  (n, m) matrix
        funcs: list of activation functions (SymPy functions)
        n_double:   Number of activation functions that take in 2 inputs

    Returns:
        SymPy matrix with 1 column that represents the output of applying the activation functions.
    """
    W = sym.Matrix(W)
    if n_double == 0:
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W[i, j] = funcs[j](W[i, j])
    else:
        W_new = W.copy()
        out_size = len(funcs)
        for i in range(W.shape[0]):
            in_j = 0
            out_j = 0
            while out_j < out_size - n_double:
                W_new[i, out_j] = funcs[out_j](W[i, in_j])
                in_j += 1
                out_j += 1
            while out_j < out_size:
                W_new[i, out_j] = funcs[out_j](W[i, in_j], W[i, in_j+1])
                in_j += 2
                out_j += 1
        for i in range(n_double):
            W_new.col_del(-1)
        W = W_new
    return W


def sym_pp(W_list, funcs, var_names, threshold=0.01, n_double=0):
    """Pretty print the hidden layers (not the last layer) of the symbolic regression network

    Arguments:
        W_list: list of weight matrices for the hidden layers
        funcs:  list of lambda functions using sympy. has the same size as W_list[i][j, :]
        var_names: list of strings for names of variables
        threshold: threshold for filtering expression. set to 0 for no filtering.
        n_double:   Number of activation functions that take in 2 inputs

    Returns:
        Simplified sympy expression.
    """
    vars = []
    for var in var_names:
        if isinstance(var, str):
            vars.append(sym.Symbol(var))
        else:
            vars.append(var)
    expr = sym.Matrix(vars).T
    # W_list = np.asarray(W_list)
    for W in W_list:
        W = filter_mat(sym.Matrix(W), threshold=threshold)
        expr = expr * W
        expr = apply_activation(expr, funcs, n_double=n_double)
    # expr = expr * W_list[-1]
    return expr


def last_pp(eq, W):
    """Pretty print the last layer."""
    return eq * filter_mat(sym.Matrix(W))


def network_deprecated(weights, funcs, var_names, threshold=0.01):
    """Pretty print the entire symbolic regression network.

    Arguments:
        weights: list of weight matrices for the entire network
        funcs:  list of lambda functions using sympy. has the same size as W_list[i][j, :]
        var_names: list of strings for names of variables
        threshold: threshold for filtering expression. set to 0 for no filtering.

    Returns:
        Simplified sympy expression."""
    n_double = functions.count_double(funcs)
    funcs = [func.sp for func in funcs]

    expr = sym_pp(weights[:-1], funcs, var_names, threshold=threshold, n_double=n_double)
    expr = last_pp(expr, weights[-1])
    expr = expr
    return expr


def filter_mat(mat, threshold=0.01):
    """Remove elements of a matrix below a threshold."""
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if abs(mat[i, j]) < threshold:
                mat[i, j] = 0
    return mat


def filter_expr(expr, threshold=0.01):
    """Remove additive terms with coefficient below threshold
    TODO: Make more robust. This does not work in all cases."""
    expr_new = sym.Integer(0)
    for arg in expr.args:
        if arg.is_constant() and abs(arg) > threshold:   # hack way to check if it's a number
            expr_new = expr_new + arg
        elif not arg.is_constant() and abs(arg.args[0]) > threshold:
            expr_new = expr_new + arg
    return expr_new


def filter_expr2(expr, threshold=0.01):
    """Sets all constants under threshold to 0
    TODO: Test"""
    for a in sym.preorder_traversal(expr):
        if isinstance(a, sym.Float) and a < threshold:
            expr = expr.subs(a, 0)
    return expr

def round_expr(expr, accuracy=0.01):
    """
    Rounds all numeric coefficients and constants in the expression 
    to the nearest multiple of the specified accuracy.

    Parameters:
        expr (sympy expression): The expression to be rounded.
        accuracy (float): The rounding increment (e.g. 0.01 to round to two decimal places).

    Returns:
        sympy expression: A new expression with all numeric constants rounded.
    """
    # Build a mapping from each numeric atom to its rounded value.
    mapping = {}
    for num in expr.atoms(sym.Float):
        # Round to the nearest multiple of 'accuracy'
        rounded_value = accuracy * round(float(num) / accuracy)
        mapping[num] = sym.Float(rounded_value)
    
    # Replace the numbers in the expression with their rounded versions.
    return expr.xreplace(mapping)

def network(weights, funcs, var_names, threshold=0.01, accuracy=0.01):
    """
    Pretty print a symbolic regression network with one symbolic layer.
    
    This function assumes that the network consists of:
      (0) A SymbolicLayer with a Linear transform having weight W1 (shape: [n_hidden, in_dim])
          and bias b1 (shape: [n_hidden]). This layer then applies activation functions in the
          following order:
              - The first n_single outputs are f(x) = f_sp(g) where g = b1 + W1 * x.
              - The remaining n_double outputs are computed in pairs: for each j, 
                  h[n_single + j] = f_sp( g[n_single+2*j], g[n_single+2*j+1] )
          (Here, n_hidden = n_single + 2*n_double.)
      (1) A final Linear layer that computes: out = b2 + W2 * h,
          where W2 has shape [n_out, n_funcs] and b2 has shape [n_out].
    
    The weights list is assumed to be [W1, b1, W2, b2].
    
    Args:
        weights: list of numpy arrays from get_weights() with shapes:
                 [ (n_hidden, in_dim), (n_hidden,), (n_out, n_funcs), (n_out,) ]
        funcs: list of activation function objects (each with a .sp method) of length n_funcs.
               Their order must match that used in the SymbolicLayer forward pass.
        var_names: list of strings, names of the input variables.
        threshold: constants with absolute value below this are set to zero.
        accuracy: rounding accuracy for coefficients.
    
    Returns:
        A list of sympy expressions, one per output neuron.
    """

    # Unpack the weights
    W1 = sym.Matrix(weights[0])  # shape (n_hidden, in_dim)
    b1 = sym.Matrix(weights[1])  # shape (n_hidden,) -> treat as column vector
    W2 = sym.Matrix(weights[2])  # shape (n_out, n_funcs)
    b2 = sym.Matrix(weights[3])  # shape (n_out,)

    # filter mat to speed up processing
    W1 = filter_mat(W1, threshold=threshold)
    b1 = filter_mat(b1, threshold=threshold)
    W2 = filter_mat(W2, threshold=threshold)
    b2 = filter_mat(b2, threshold=threshold)

    # Ensure biases are column vectors.
    if b1.shape[1] != 1:
        b1 = b1.reshape(b1.rows, 1)
    if b2.shape[1] != 1:
        b2 = b2.reshape(b2.rows, 1)

    # Create sympy symbols for the input variables.
    # For example, if var_names = ['x1', 'x2', ..., 'x24']
    x = sym.symbols(var_names)

    in_dim = len(var_names)
    n_hidden = W1.rows  # This is n_hidden = n_single + 2*n_double.
    # Compute the pre-activation outputs (g) of the symbolic layer.
    # For each hidden unit i, g[i] = b1[i] + sum_{k} W1[i,k] * x[k]
    g = []
    for i in range(n_hidden):
        expr = b1[i, 0]
        for k in range(in_dim):
            expr += W1[i, k] * x[k]
        g.append(expr)

    # Determine how many activation functions there are.
    n_funcs = len(funcs)
    # Count how many functions are double-input:
    n_double = functions.count_double(funcs)
    n_single = n_funcs - n_double

    # Apply the activation functions to form the output h of the symbolic layer.
    h = []
    # First, for the single-input activations:
    for i in range(n_single):
        f_sp = funcs[i].sp  # the sympy version of the function
        h.append(f_sp(g[i]))
    # Next, for the double-input activations, use pairs from g.
    for j in range(n_double):
        f_sp = funcs[n_single + j].sp
        idx1 = n_single + 2 * j
        idx2 = n_single + 2 * j + 1
        h.append(f_sp(g[idx1], g[idx2]))

    # Now, h is a list of length n_funcs (should match W2.cols).
    # Compute the final outputs: for each output neuron m,
    #   out[m] = b2[m] + sum_{j} W2[m,j] * h[j]
    outputs = []
    n_out = W2.rows
    for m in range(n_out):
        expr = b2[m, 0]
        for j in range(W2.cols):
            expr += W2[m, j] * h[j]
        # Simplify, expand, filter out very small terms, and round the coefficients.
        expr_simpl = sym.simplify(expr)
        expr_expanded = sym.expand(expr_simpl)
        expr_filtered = filter_expr2(expr_expanded, threshold)
        expr_rounded = round_expr(expr_filtered, accuracy)
        outputs.append(expr_rounded)
    return outputs

def parallel_simplify(expr, threshold, accuracy):
    """Helper function to simplify, expand, filter, and round an expression."""
    expr_simpl = sym.simplify(expr)
    expr_expanded = sym.expand(expr_simpl)
    expr_filtered = filter_expr2(expr_expanded, threshold)
    expr_rounded = round_expr(expr_filtered, accuracy)
    return expr_rounded

def simplify_helper(expr, threshold, accuracy):
    return parallel_simplify(expr, threshold, accuracy)


def network_optimized(weights, funcs, var_names, threshold=0.01, accuracy=0.01):
    """
    Optimized version of the network function that speeds up processing by vectorizing
    the pre-activation computation and parallelizing the final simplification step.
    
    Arguments:
        weights: list of numpy arrays for [W1, b1, W2, b2]
        funcs: list of activation function objects (each with a .sp attribute)
        var_names: list of strings, names of the input variables.
        threshold: constants with absolute value below this are set to zero.
        accuracy: rounding accuracy for coefficients.
    
    Returns:
        A list of optimized sympy expressions (one per output neuron).
    """
    # Unpack weights and convert to sympy matrices
    W1 = sym.Matrix(weights[0])
    b1 = sym.Matrix(weights[1])
    W2 = sym.Matrix(weights[2])
    b2 = sym.Matrix(weights[3])
    
    # Filter small values early to speed up later computation
    W1 = filter_mat(W1, threshold=threshold)
    b1 = filter_mat(b1, threshold=threshold)
    W2 = filter_mat(W2, threshold=threshold)
    b2 = filter_mat(b2, threshold=threshold)
    
    # Ensure biases are column vectors.
    if b1.shape[1] != 1:
        b1 = b1.reshape(b1.rows, 1)
    if b2.shape[1] != 1:
        b2 = b2.reshape(b2.rows, 1)
    
    # Create sympy symbols and convert to a column vector.
    x = sym.symbols(var_names)
    x_vec = sym.Matrix(x).reshape(len(var_names), 1)
    
    # Vectorized computation of pre-activation outputs: g = b1 + W1*x
    g_vec = b1 + W1 * x_vec  # g_vec is a column vector of length n_hidden
    # Convert g_vec to a list for applying individual activation functions
    g_list = list(g_vec)
    
    n_funcs = len(funcs)
    n_double = functions.count_double(funcs)
    n_single = n_funcs - n_double
    
    # Compute activation outputs for single-input functions
    h = [funcs[i].sp(g_list[i]) for i in range(n_single)]
    # Compute activation outputs for double-input functions in pairs
    for j in range(n_double):
        f_sp = funcs[n_single + j].sp
        idx1 = n_single + 2 * j
        idx2 = n_single + 2 * j + 1
        h.append(f_sp(g_list[idx1], g_list[idx2]))
    
    # Form a column matrix H from h and compute final outputs using matrix multiplication:
    # outputs = b2 + W2 * H
    H = sym.Matrix(h)
    output_matrix = b2 + W2 * H
    outputs = [output_matrix[i, 0] for i in range(output_matrix.rows)]
    
    # Use multiple processes to simplify each output expression in parallel.
    with ProcessPoolExecutor() as executor:
            # Map the helper function across outputs
            outputs_optimized = list(executor.map(simplify_helper, outputs, [threshold]*len(outputs), [accuracy]*len(outputs)))
    
    return outputs_optimized

def extract_equations(agent, variable_names, output_names, accuracy=0.001, threshold=0.01, print_out=True, use_multiprocessing=True):
    # get expressions
    eql_actor_weights = agent.eql_actor.get_weights()
    activation_funcs = agent.activation_funcs
    network_func = network if not use_multiprocessing else network_optimized
    expra = network_func(
            eql_actor_weights,
            activation_funcs,
            variable_names,
            accuracy=accuracy,
            threshold=threshold
    )
    outputs = []
    for expr, action_name in zip(expra, output_names):
        output = f"logits_{action_name} = {str(expr)}"
        if print_out:
            print(output)

    return outputs, expra



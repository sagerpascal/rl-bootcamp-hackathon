"""

This file copied from the OpenAI Baseline and slightly adjusted (for MPI support, see results -> ppo2)

Source: https://github.com/openai/baselines

"""

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def sync_from_root(variables, comm=None):
    """
    Send the root node's parameters to every worker.
    Arguments:
      variables: all parameter variables including optimizer's
    """
    if comm is None: comm = MPI.COMM_WORLD
    values = comm.bcast([var.numpy() for var in variables])
    for (var, val) in zip(variables, values):
        var.assign(val)

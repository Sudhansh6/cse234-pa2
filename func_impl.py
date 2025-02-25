import numpy as np
from mpi4py import MPI


def get_info(
    comm,
    rank: int,
    mp_size: int,
    dp_size: int,
    fc_layer: str,
    in_dim: int,
    out_dim: int,
):
    """
    Prepare necessary information for later communications in forward and backward passes.

    Parameters
    ----------
    comm : Communicator
        The global MPI communicator.
    rank : int
        The global rank of the process.
    mp_size : int
        Model Parallel size.
    dp_size : int
        Data Parallel size.
    fc_layer : str
        Identifier for the fully-connected layer. It must be one of:
        'fc_q', 'fc_k', 'fc_v', or 'fc_o'.
        - For 'fc_q', 'fc_k', and 'fc_v', the partitioning is along the output dimension.
        - For 'fc_o', the partitioning is along the input dimension.
    in_dim : int
        Original input feature dimension.
    out_dim : int
        Original output feature dimension.

    Returns
    -------
    mp_idx : int
        Model parallel index (position within a data parallel replica).
    dp_idx : int
        Data parallel index (which replica this process belongs to).
    mp_comm : Communicator
        The model parallel communicator (all processes in one data parallel replica).
    dp_comm : Communicator
        The data parallel communicator (all processes holding the same weight shard).
    part_in_dim : int
        The partitioned input dimension for the FC layer.
    part_out_dim : int
        The partitioned output dimension for the FC layer.
    """
    color = rank // mp_size  
    mp_comm = comm.Split(color=color, key=rank)  
    dp_comm = comm.Split(color=rank % mp_size, key=rank)  

    mp_idx = rank % mp_size
    dp_idx = rank // mp_size

    if fc_layer == "fc_o":
        part_in_dim = in_dim // mp_size
        part_out_dim = out_dim  
    else:
        part_in_dim = in_dim
        part_out_dim = out_dim // mp_size 

    return mp_idx, dp_idx, mp_comm, dp_comm, part_in_dim, part_out_dim


def naive_collect_forward_input(
    x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """
    Collects the fc_o layer's forward inputs from all model-parallel nodes.

    Each node holds a piece of the full input with shape:
      (batch_size, seq_length, part_in_dim)
    After gathering, the full input should have shape:
      (batch_size, seq_length, part_in_dim * mp_size)
    """
    batch_size, sequence_length, part_in_dimension = x.shape
    x = np.ascontiguousarray(x)
    full_in_dimension = part_in_dimension * mp_size
    collected_x = np.zeros((batch_size, sequence_length, full_in_dimension), dtype=x.dtype)

    receive_buffers = []
    for i in range(mp_size):
        if i == mp_comm.Get_rank():
            receive_buffer = x
        else:
            receive_buffer = np.empty((batch_size, sequence_length, part_in_dimension), dtype=x.dtype)

        receive_buffer = np.ascontiguousarray(receive_buffer)
        receive_buffers.append(receive_buffer)
     
    for i in range(mp_size):
        if i == mp_comm.Get_rank():
            continue

        if mp_comm.Get_rank() < i:
            mp_comm.Send(x, dest=i)
            mp_comm.Recv(receive_buffers[i], source=i)
        else:
            mp_comm.Recv(receive_buffers[i], source=i)
            mp_comm.Send(x, dest=i)

    for i in range(mp_size):
        start = i * part_in_dimension
        end = start + part_in_dimension
        collected_x[:, :, start:end] = receive_buffers[i]
        
    return collected_x


def naive_collect_forward_output(
    out: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """
    Collects the fc_o layer's forward outputs from all model-parallel nodes.

    Each node holds a piece of the full output with shape:
      (batch_size, seq_length, part_out_dim)
    After gathering, the full output should have shape:
      (batch_size, seq_length, part_out_dim * mp_size)
    """
    batch_size, sequence_length, part_out_dimension = out.shape
    
    out = np.ascontiguousarray(out)

    full_out_dimension = part_out_dimension * mp_size

    collected_out = np.empty((batch_size, sequence_length, full_out_dimension), dtype=out.dtype)

    rank = mp_comm.Get_rank()

    send_buffer = out.copy()

    receive_buffer = np.empty([mp_size, batch_size, sequence_length, part_out_dimension], dtype=out.dtype)

    mp_comm.Allgather(send_buffer, receive_buffer)

    for i in range(mp_size):
        start = i * part_out_dimension
        end =start + part_out_dimension
        collected_out[:, :, start: end] = receive_buffer[i]

    
    return collected_out


def naive_collect_backward_output(
    output_grad: np.ndarray,
    mp_group_idx: int,
    mp_size: int,
):
    """
    Collect the fc output layer's output gradient for the local MP node.
    
    In our setup, the full output_grad is a 3-D tensor of shape 
        (batch_size, seq_length, out_dim),
    and the fully connected layer's weight is partitioned along out_dim.
    Therefore, we split output_grad along axis=2 into mp_size parts and
    return the part corresponding to mp_group_idx.
    """
    batch_size, seq_length, out_dim = output_grad.shape
    part_out_dim = out_dim // mp_size  

    collected_output_grad = output_grad[:, :, mp_group_idx * part_out_dim: (mp_group_idx + 1) * part_out_dim]

    return collected_output_grad



def naive_collect_backward_x(grad_x: np.ndarray, mp_comm, mp_size: int):
    """
    Perform Reduce-Scatter to sum and distribute grad_x across model parallel nodes.

    Parameters
    ----------
    grad_x : np.ndarray
        Local gradient with shape (batch_size, seq_length, in_dim).
    mp_comm : MPI communicator
        The communicator for model parallel nodes.
    mp_size : int
        Number of model parallel nodes.

    Returns
    -------
    collected_grad_x : np.ndarray
        The reduced and distributed grad_x with shape
        (batch_size, seq_length, in_dim // mp_size).
    """

    batch_size, seq_length, in_dim = grad_x.shape
    part_in_dim = in_dim // mp_size  

    grad_x = np.ascontiguousarray(grad_x)

    global_grad_x = np.empty_like(grad_x)
    mp_comm.Allreduce(sendbuf=grad_x, recvbuf=global_grad_x, op=MPI.SUM)

    collected_grad_x = np.empty((batch_size, seq_length, part_in_dim), dtype=grad_x.dtype)
    scatter_buffer = np.split(global_grad_x, mp_size, axis=2)

    mp_comm.Scatter(sendbuf=np.ascontiguousarray(scatter_buffer), recvbuf=collected_grad_x, root=0)

    return collected_grad_x

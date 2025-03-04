from mpi4py import MPI
import numpy as np

class Communicator(object):
    def __init__(self, comm: MPI.Comm):
        self.comm = comm
        self.total_bytes_transferred = 0

    def Get_size(self):
        return self.comm.Get_size()

    def Get_rank(self):
        return self.comm.Get_rank()

    def Barrier(self):
        return self.comm.Barrier()

    def Allreduce(self, src_array, dest_array, op=MPI.SUM):
        assert src_array.size == dest_array.size
        src_array_byte = src_array.itemsize * src_array.size
        self.total_bytes_transferred += src_array_byte * 2 * (self.comm.Get_size() - 1)
        self.comm.Allreduce(src_array, dest_array, op)

    def Allgather(self, src_array, dest_array):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Allgather(src_array, dest_array)

    def Reduce_scatter(self, src_array, dest_array, op=MPI.SUM):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Reduce_scatter_block(src_array, dest_array, op)

    def Split(self, key, color):
        return __class__(self.comm.Split(key=key, color=color))

    def Alltoall(self, src_array, dest_array):
        nprocs = self.comm.Get_size()

        # Ensure that the arrays can be evenly partitioned among processes.
        assert src_array.size % nprocs == 0, (
            "src_array size must be divisible by the number of processes"
        )
        assert dest_array.size % nprocs == 0, (
            "dest_array size must be divisible by the number of processes"
        )

        # Calculate the number of bytes in one segment.
        send_seg_bytes = src_array.itemsize * (src_array.size // nprocs)
        recv_seg_bytes = dest_array.itemsize * (dest_array.size // nprocs)

        # Each process sends one segment to every other process (nprocs - 1)
        # and receives one segment from each.
        self.total_bytes_transferred += send_seg_bytes * (nprocs - 1)
        self.total_bytes_transferred += recv_seg_bytes * (nprocs - 1)

        self.comm.Alltoall(src_array, dest_array)

    def myAllreduce(self, src_array, dest_array, op=MPI.SUM):
        """
        A manual implementation of all-reduce using a reduce-to-root
        followed by a broadcast.
        
        Each non-root process sends its data to process 0, which applies the
        reduction operator (by default, summation). Then process 0 sends the
        reduced result back to all processes.
        
        The transfer cost is computed as:
          - For non-root processes: one send and one receive.
          - For the root process: (n-1) receives and (n-1) sends.
        """
        #TODO: Your code here
        assert src_array.size == dest_array.size

        rank = self.comm.Get_rank()
        size = self.comm.Get_size()

        if rank == 0:
            # For some reason np.copy does not work but np.copyto works!
            # dest_array = np.copy(src_array) 
            # Update: We ideally want to update dest_array in place. 
            # # Using src_array.copy() creates a new object essentially losing the reference to the memory given in the argument
            # The same applied to the minimum and maxoimum operations below
            np.copyto(dest_array, src_array)

            for i in range(1, size):
                recv_array = np.empty_like(src_array)  
                self.comm.Recv(recv_array, source=i, tag=0)
                # Update: This is not required according to a Piazza post, and removing these 
                # total_bytes operations can save some more time. 
                self.total_bytes_transferred += src_array.nbytes

                if op == MPI.SUM:
                    dest_array += recv_array
                elif op == MPI.MIN:
                    # Again, the commented line does not work but out parameter works!
                    # dest_array = np.minimum(dest_array, recv_array)
                    np.minimum(dest_array, recv_array, out=dest_array)

                elif op == MPI.MAX:
                    np.maximum(dest_array, recv_array, out=dest_array)

            for i in range(1, size):
                self.comm.Send(dest_array, dest=i, tag = 0)
                self.total_bytes_transferred += dest_array.nbytes

        else:
            # Send the array rank i (i != 0) has
            self.comm.Send(src_array, dest=0, tag=0)
            self.total_bytes_transferred += src_array.nbytes
            # Receive the result from rank 0 that is the root
            self.comm.Recv(dest_array, source=0, tag=0)
            self.total_bytes_transferred += dest_array.nbytes


    def myAlltoall(self, src_array, dest_array):
        """
        A manual implementation of all-to-all where each process sends a
        distinct segment of its source array to every other process.
        
        It is assumed that the total length of src_array (and dest_array)
        is evenly divisible by the number of processes.
        
        The algorithm loops over the ranks:
          - For the local segment (when destination == self), a direct copy is done.
          - For all other segments, the process exchanges the corresponding
            portion of its src_array with the other process via Sendrecv.
            
        The total data transferred is updated for each pairwise exchange.
        """
        #TODO: Your code here
        assert src_array.size == dest_array.size

        size = self.comm.Get_size()
        rank = self.comm.Get_rank()
        
        # Calculate segment size
        segment_size = len(src_array) // size
        
        # Requests for non-blocking communication
        requests = []
        
        # Initiate all non-blocking sends and receives
        for i in range(size):
            if i != rank:  # Skip self
                
                # Calculate offsets
                send_offset = i * segment_size
                recv_offset = i * segment_size
                
                # Non-blocking send and receive
                req_send = self.comm.Isend(
                    src_array[send_offset:send_offset+segment_size], 
                    dest=i, tag=rank
                )
                req_recv = self.comm.Irecv(
                    dest_array[recv_offset:recv_offset+segment_size], 
                    source=i, tag=i
                )
                
                requests.append(req_send)
                requests.append(req_recv)
        
        # Handle local copy while communication simultaneously
        local_offset = rank * segment_size
        dest_array[local_offset:local_offset+segment_size] = \
            src_array[local_offset:local_offset+segment_size].copy()
        
        # Wait for the communication to complete
        MPI.Request.Waitall(requests)
        
        self.total_bytes_transferred += segment_size * (size - 1) * 2  # Both send and receive
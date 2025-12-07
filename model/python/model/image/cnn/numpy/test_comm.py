"""
Test script for custom communication library.
Run with: ./launch_ddp.sh 4 localhost 29500
(but change the python command in launch_ddp.sh to test_comm.py)
"""
import numpy as np
from simple_comm import init_communicator


def test_broadcast():
    """Test broadcast operation."""
    comm = init_communicator()
    rank = comm.get_rank()
    world_size = comm.get_world_size()
    
    # Create data
    if rank == 0:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        print(f"[Rank {rank}] Broadcasting: {data}")
    else:
        data = np.zeros(5)
    
    # Broadcast from rank 0
    comm.broadcast(data, root=0)
    
    print(f"[Rank {rank}] After broadcast: {data}")
    
    # Verify
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.allclose(data, expected), f"Rank {rank} broadcast failed!"
    
    comm.barrier()
    if rank == 0:
        print("✓ Broadcast test passed!\n")


def test_allreduce():
    """Test all-reduce operation."""
    comm = init_communicator()
    rank = comm.get_rank()
    world_size = comm.get_world_size()
    
    # Each rank has different data
    send_data = np.array([rank, rank * 2, rank * 3], dtype=np.float64)
    recv_data = np.zeros_like(send_data)
    
    print(f"[Rank {rank}] Before allreduce: {send_data}")
    
    # All-reduce sum
    comm.allreduce(send_data, recv_data, op='sum')
    
    print(f"[Rank {rank}] After allreduce: {recv_data}")
    
    # Verify: sum should be [0+1+2+3, 0+2+4+6, 0+3+6+9] for 4 processes
    expected_sum = sum(range(world_size))
    expected = np.array([expected_sum, expected_sum * 2, expected_sum * 3])
    assert np.allclose(recv_data, expected), f"Rank {rank} allreduce failed!"
    
    comm.barrier()
    if rank == 0:
        print("✓ All-reduce test passed!\n")


def test_allreduce_ring():
    """Test ring all-reduce operation."""
    comm = init_communicator()
    rank = comm.get_rank()
    world_size = comm.get_world_size()
    
    # Larger data for ring algorithm
    send_data = np.ones(100, dtype=np.float64) * rank
    recv_data = np.zeros_like(send_data)
    
    print(f"[Rank {rank}] Before ring allreduce: sum={send_data.sum()}")
    
    # Ring all-reduce sum
    comm.allreduce_ring(send_data, recv_data, op='sum')
    
    print(f"[Rank {rank}] After ring allreduce: sum={recv_data.sum()}")
    
    # Verify
    expected_sum = sum(range(world_size))
    expected = np.ones(100) * expected_sum
    assert np.allclose(recv_data, expected), f"Rank {rank} ring allreduce failed!"
    
    comm.barrier()
    if rank == 0:
        print("✓ Ring all-reduce test passed!\n")


def test_barrier():
    """Test barrier synchronization."""
    comm = init_communicator()
    rank = comm.get_rank()
    
    import time
    
    print(f"[Rank {rank}] Before barrier at {time.time():.2f}")
    
    # Simulate different arrival times
    time.sleep(rank * 0.1)
    
    # Barrier - all processes wait here
    comm.barrier()
    
    print(f"[Rank {rank}] After barrier at {time.time():.2f}")
    
    if rank == 0:
        print("✓ Barrier test passed!\n")


def test_send_recv():
    """Test point-to-point communication."""
    comm = init_communicator()
    rank = comm.get_rank()
    world_size = comm.get_world_size()
    
    if world_size < 2:
        if rank == 0:
            print("⊘ Send/recv test skipped (need at least 2 processes)\n")
        return
    
    if rank == 0:
        # Rank 0 sends to rank 1
        data = np.array([10.0, 20.0, 30.0])
        print(f"[Rank {rank}] Sending to rank 1: {data}")
        comm.send(data, dest=1)
        
        # Receive from rank 1
        received = comm.recv(source=1)
        print(f"[Rank {rank}] Received from rank 1: {received}")
        
    elif rank == 1:
        # Rank 1 receives from rank 0
        received = comm.recv(source=0)
        print(f"[Rank {rank}] Received from rank 0: {received}")
        
        # Send back to rank 0
        data = np.array([100.0, 200.0, 300.0])
        print(f"[Rank {rank}] Sending to rank 0: {data}")
        comm.send(data, dest=0)
    
    comm.barrier()
    if rank == 0:
        print("✓ Send/recv test passed!\n")


def main():
    """Run all tests."""
    comm = init_communicator()
    rank = comm.get_rank()
    world_size = comm.get_world_size()
    
    if rank == 0:
        print("=" * 60)
        print(f"Testing Custom Communication Library")
        print(f"World size: {world_size}")
        print("=" * 60)
        print()
    
    comm.barrier()
    
    # Run tests
    test_broadcast()
    test_allreduce()
    test_allreduce_ring()
    test_barrier()
    test_send_recv()
    
    # Cleanup
    comm.cleanup()
    
    if rank == 0:
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)


if __name__ == "__main__":
    main()

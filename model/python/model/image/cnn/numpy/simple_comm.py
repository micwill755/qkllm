"""
Simple communication library from scratch using sockets.
Implements basic collective operations: Broadcast, Allreduce, Barrier.
"""
import socket
import struct
import numpy as np
import pickle
import time
import os


class Communicator:
    """Simple communicator using TCP sockets."""
    
    def __init__(self, rank, world_size, master_addr='localhost', master_port=29500):
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        
        # Socket connections to other processes
        self.sockets = {}
        self.server_socket = None
        
        # Initialize connections
        self._setup_connections()
    
    def _setup_connections(self):
        """Set up peer-to-peer connections between all processes."""
        if self.rank == 0:
            # Rank 0 acts as rendezvous server
            self._setup_master()
        else:
            # Other ranks connect to master
            time.sleep(0.5)  # Give master time to start
            self._connect_to_master()
        
        # Exchange connection info and establish peer connections
        self._establish_peer_connections()
    
    def _setup_master(self):
        """Rank 0 sets up server socket."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.master_addr, self.master_port))
        self.server_socket.listen(self.world_size)
        print(f"[Rank {self.rank}] Master listening on {self.master_addr}:{self.master_port}")
    
    def _connect_to_master(self):
        """Non-master ranks connect to rank 0."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.master_addr, self.master_port))
        self.sockets[0] = sock
        
        # Send our rank to master
        self._send_data(sock, {'rank': self.rank})
        print(f"[Rank {self.rank}] Connected to master")
    
    def _establish_peer_connections(self):
        """Establish connections between all peers."""
        if self.rank == 0:
            # Master collects info from all ranks
            peer_info = {0: (self.master_addr, self.master_port + 100)}
            
            for i in range(1, self.world_size):
                client_sock, addr = self.server_socket.accept()
                data = self._recv_data(client_sock)
                peer_rank = data['rank']
                self.sockets[peer_rank] = client_sock
                
                # Each rank listens on master_port + 100 + rank
                peer_info[peer_rank] = (addr[0], self.master_port + 100 + peer_rank)
            
            # Broadcast peer info to all ranks
            for rank in range(1, self.world_size):
                self._send_data(self.sockets[rank], peer_info)
            
            print(f"[Rank {self.rank}] All peers connected")
        else:
            # Receive peer info from master
            peer_info = self._recv_data(self.sockets[0])
            
            # Set up our own server for peer connections
            my_port = self.master_port + 100 + self.rank
            peer_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            peer_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            peer_server.bind(('', my_port))
            peer_server.listen(self.world_size)
            
            # Connect to lower-ranked peers
            for peer_rank in range(self.rank):
                if peer_rank != 0:  # Already connected to rank 0
                    addr, port = peer_info[peer_rank]
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect((addr, port))
                    self.sockets[peer_rank] = sock
            
            # Accept connections from higher-ranked peers
            for peer_rank in range(self.rank + 1, self.world_size):
                client_sock, _ = peer_server.accept()
                self.sockets[peer_rank] = client_sock
            
            peer_server.close()
            print(f"[Rank {self.rank}] Peer connections established")
    
    def _send_data(self, sock, data):
        """Send data over socket."""
        serialized = pickle.dumps(data)
        size = len(serialized)
        
        # Send size first (4 bytes)
        sock.sendall(struct.pack('!I', size))
        # Send data
        sock.sendall(serialized)
    
    def _recv_data(self, sock):
        """Receive data from socket."""
        # Receive size (4 bytes)
        size_data = self._recv_all(sock, 4)
        size = struct.unpack('!I', size_data)[0]
        
        # Receive data
        data = self._recv_all(sock, size)
        return pickle.loads(data)
    
    def _recv_all(self, sock, size):
        """Receive exactly size bytes from socket."""
        data = b''
        while len(data) < size:
            chunk = sock.recv(size - len(data))
            if not chunk:
                raise ConnectionError("Socket connection broken")
            data += chunk
        return data
    
    def send(self, data, dest):
        """Send data to destination rank."""
        self._send_data(self.sockets[dest], data)
    
    def recv(self, source):
        """Receive data from source rank."""
        return self._recv_data(self.sockets[source])
    
    def barrier(self):
        """Synchronization barrier - all processes wait here."""
        if self.rank == 0:
            # Master waits for all others to arrive
            for rank in range(1, self.world_size):
                self._recv_data(self.sockets[rank])
            
            # Signal all to continue
            for rank in range(1, self.world_size):
                self._send_data(self.sockets[rank], {'continue': True})
        else:
            # Send arrival signal to master
            self._send_data(self.sockets[0], {'arrived': True})
            
            # Wait for continue signal
            self._recv_data(self.sockets[0])
    
    def broadcast(self, data, root=0):
        """
        Broadcast data from root to all processes.
        
        Args:
            data: numpy array to broadcast (modified in-place on non-root)
            root: rank that has the data
        """
        if self.rank == root:
            # Root sends to all others
            for rank in range(self.world_size):
                if rank != root:
                    self._send_data(self.sockets[rank], data)
        else:
            # Receive from root
            received = self._recv_data(self.sockets[root])
            np.copyto(data, received)
    
    def allreduce(self, send_data, recv_data, op='sum'):
        """
        All-reduce operation using ring algorithm.
        
        Args:
            send_data: numpy array with local data
            recv_data: numpy array to store result
            op: operation ('sum', 'max', 'min', 'prod')
        """
        # Simple implementation: gather to rank 0, reduce, broadcast
        # (Ring all-reduce is more complex but more efficient)
        
        if self.rank == 0:
            # Rank 0 collects from all
            result = send_data.copy()
            
            for rank in range(1, self.world_size):
                other_data = self._recv_data(self.sockets[rank])
                
                if op == 'sum':
                    result += other_data
                elif op == 'max':
                    result = np.maximum(result, other_data)
                elif op == 'min':
                    result = np.minimum(result, other_data)
                elif op == 'prod':
                    result *= other_data
            
            # Broadcast result to all
            np.copyto(recv_data, result)
            for rank in range(1, self.world_size):
                self._send_data(self.sockets[rank], result)
        else:
            # Send to rank 0
            self._send_data(self.sockets[0], send_data)
            
            # Receive result from rank 0
            result = self._recv_data(self.sockets[0])
            np.copyto(recv_data, result)
    
    def allreduce_ring(self, send_data, recv_data, op='sum'):
        """
        Ring all-reduce - more efficient than gather-reduce-broadcast.
        
        Divides data into chunks and uses ring topology for communication.
        """
        n = len(send_data)
        chunk_size = (n + self.world_size - 1) // self.world_size
        
        # Initialize recv_data with send_data
        np.copyto(recv_data, send_data)
        
        # Ring reduce-scatter phase
        for step in range(self.world_size - 1):
            send_rank = (self.rank - step) % self.world_size
            recv_rank = (self.rank - step - 1) % self.world_size
            
            send_chunk_idx = send_rank
            recv_chunk_idx = recv_rank
            
            send_start = send_chunk_idx * chunk_size
            send_end = min(send_start + chunk_size, n)
            recv_start = recv_chunk_idx * chunk_size
            recv_end = min(recv_start + chunk_size, n)
            
            # Send and receive simultaneously
            next_rank = (self.rank + 1) % self.world_size
            prev_rank = (self.rank - 1 + self.world_size) % self.world_size
            
            # Send chunk to next rank
            send_chunk = recv_data[send_start:send_end].copy()
            self._send_data(self.sockets[next_rank], send_chunk)
            
            # Receive chunk from previous rank
            recv_chunk = self._recv_data(self.sockets[prev_rank])
            
            # Reduce received chunk
            if op == 'sum':
                recv_data[recv_start:recv_end] += recv_chunk
            elif op == 'max':
                recv_data[recv_start:recv_end] = np.maximum(
                    recv_data[recv_start:recv_end], recv_chunk
                )
        
        # Ring all-gather phase
        for step in range(self.world_size - 1):
            send_rank = (self.rank - step + 1) % self.world_size
            recv_rank = (self.rank - step) % self.world_size
            
            send_chunk_idx = send_rank
            recv_chunk_idx = recv_rank
            
            send_start = send_chunk_idx * chunk_size
            send_end = min(send_start + chunk_size, n)
            recv_start = recv_chunk_idx * chunk_size
            recv_end = min(recv_start + chunk_size, n)
            
            next_rank = (self.rank + 1) % self.world_size
            prev_rank = (self.rank - 1 + self.world_size) % self.world_size
            
            # Send chunk to next rank
            send_chunk = recv_data[send_start:send_end].copy()
            self._send_data(self.sockets[next_rank], send_chunk)
            
            # Receive chunk from previous rank
            recv_chunk = self._recv_data(self.sockets[prev_rank])
            recv_data[recv_start:recv_end] = recv_chunk
    
    def get_rank(self):
        """Get rank of this process."""
        return self.rank
    
    def get_world_size(self):
        """Get total number of processes."""
        return self.world_size
    
    def cleanup(self):
        """Close all socket connections."""
        for sock in self.sockets.values():
            sock.close()
        if self.server_socket:
            self.server_socket.close()


def init_communicator():
    """Initialize communicator from environment variables."""
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = int(os.environ.get('MASTER_PORT', 29500))
    
    return Communicator(rank, world_size, master_addr, master_port)

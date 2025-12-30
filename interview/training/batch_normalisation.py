"""NumPy implementation of PyTorch-style 2D Batch Normalization for NCHW inputs.

y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta

The task:
1. Implement the constructor (__init__) with basic state initialization
2. Implement the forward() method for training mode
3. Implement the forward() method for evaluation mode
4. Ensure numerical stability and handle edge cases
"""

import numpy as np

class BatchNorm:
    """
    NumPy implementation of PyTorch-style Batch Normalization for NCHW inputs.
    """

    def __init__(self, num_channels: int, momentum: float = 0.1, eps: float = 1e-5):
        """
        Initialize BatchNorm module.

        Initialize the module with:
        - Scalar hyperparameters: momentum, eps, num_channels
        - Learnable affine parameters: gamma, beta
        - Running statistics: running_mean, running_var
        - Mode: training by default

        Args:
            num_channels (int): Number of channels (C).
            momentum (float): Momentum for running statistics.
            eps (float): Small value for numerical stability.
        """
        self.num_channels = num_channels
        self.momentum = momentum
        self.eps = eps

        # Learnable parameters
        self.gamma = np.ones((num_channels,), dtype=np.float32) # (C,)
        self.beta  = np.zeros((num_channels,), dtype=np.float32) # (C,)

        # Running statistics
        self.running_mean = np.zeros((num_channels,), dtype=np.float32) # (C,)
        self.running_var  = np.ones((num_channels,), dtype=np.float32) # (C,)

        # Mode: training by default
        self.training = True

    def train(self):
        """Set module to training mode."""
        self.training = True

    def eval(self):
        """Set module to evaluation mode."""
        self.training = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for NCHW input.
        
        y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
        training: use batch stats (mean + variance)
        eval: use running stats (mean + variance)

        Args:
            x (np.ndarray): Input tensor of shape (N, C, H, W).

        Returns:
            np.ndarray: Batch-normalized tensor of same shape and dtype as input.
        """
        mean = None
        var = None
        
        if self.training:
            mean = np.mean(x, axis=(0, 2, 3)) # shape (C,)
            var = np.var(x, axis=(0, 2, 3)) # shape (C,)
            
            self.running_mean[:] = (1.0 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var[:] = (1.0 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
            
        mean = mean.reshape(1, -1, 1, 1)
        var = var.reshape(1, -1, 1, 1)
        gamma = self.gamma.reshape(1, -1, 1, 1)
        beta = self.beta.reshape(1, -1, 1, 1)
        
        # y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
        x_norm = (x.astype(np.float32) - mean) / np.sqrt(var + self.eps) * gamma + beta
        return x_norm
        



































################################################################################
# Tests
################################################################################

B=print
def run_tests():
    F=False;C=1.;D=0;G=19;E=1e-05;B('Running progressive BatchNorm tests...\n')
    def A(fn,name):
        C=':';A=name;nonlocal D
        try:fn();B(f"PASS: {A.split(C)[0].strip()}");D+=1
        except AssertionError:B(f"FAIL: {A.split(C)[0].strip()}")
        except Exception as E:B(f"ERROR: {A.split(C)[0].strip()}")
    def H():A=BatchNorm(3);assert A.gamma.shape==(3,);assert np.all(A.gamma==1);assert A.beta.shape==(3,);assert np.all(A.beta==0)
    A(H,'Step 1.1')
    def I():A=BatchNorm(4);assert A.running_mean.shape==(4,);assert np.all(A.running_mean==0)
    A(I,'Step 1.2')
    def J():A=BatchNorm(4);assert A.running_var.shape==(4,);assert np.all(A.running_var==1)
    A(J,'Step 1.3')
    def K():A=BatchNorm(2,momentum=.2,eps=.0001);assert np.isclose(A.momentum,.2);assert np.isclose(A.eps,.0001)
    A(K,'Step 1.4')
    def L():A=BatchNorm(1);assert A.training is True;A.eval();assert A.training is F;A.train();assert A.training is True
    A(L,'Step 1.5')
    def M():A=BatchNorm(2);B=np.random.randn(4,2,5,5).astype(np.float32);A.train();C=A.forward(B);assert C.shape==B.shape
    A(M,'Step 2.1')
    def N():
        A=BatchNorm(2);D=np.random.randn(8,2,4,4).astype(np.float32);A.train();B=A.forward(D)
        for C in range(2):F=B[:,C,:,:].mean();G=B[:,C,:,:].std();assert np.allclose(F,0,atol=E);assert np.allclose(G,1,atol=E)
    A(N,'Step 2.2')
    def O():A=BatchNorm(1);D=np.array([[[[1,2],[3,4]]]],dtype=np.float32);A.gamma=np.array([2.],dtype=np.float32);A.beta=np.array([C],dtype=np.float32);A.train();B=A.forward(D);assert np.allclose(B.mean(),C,atol=.01);assert np.allclose(B.std(),2.,atol=.01)
    A(O,'Step 2.3')
    def P():A=BatchNorm(1,momentum=.5);A.train();B=np.zeros((4,1,3,3),dtype=np.float32);C=np.ones((4,1,3,3),dtype=np.float32);A.forward(B);D=A.running_mean.copy();A.forward(C);assert D<A.running_mean
    A(P,'Step 2.4')
    def Q():A=BatchNorm(1);B=id(A.running_mean);C=id(A.running_var);A.train();D=np.random.randn(2,1,2,2).astype(np.float32);A.forward(D);assert id(A.running_mean)==B;assert id(A.running_var)==C
    A(Q,'Step 2.5')
    def R():A=BatchNorm(1);A.train();A.forward(np.zeros((4,1,3,3),dtype=np.float32));B=A.running_mean.copy();A.forward(np.ones((4,1,3,3),dtype=np.float32)*10);C=A.running_mean.copy();assert C>B
    A(R,'Step 2.6')
    def S():
        E='Step 2.7: wrong ndim input rejected';B=BatchNorm(3);C=[np.array(np.random.randn(),dtype=np.float32),np.random.randn(3).astype(np.float32),np.random.randn(3,4).astype(np.float32),np.random.randn(3,4,5).astype(np.float32),np.random.randn(2,3,4,5,6).astype(np.float32)]
        for A in C:
            try:B.forward(A);assert F
            except ValueError:continue
        A=np.random.randn(2,3,4,5).astype(np.float32);D=B.forward(A);assert D.shape==A.shape
    A(S,'Step 2.7')
    def T():A=BatchNorm(1);A.running_mean[:]=5.;A.running_var[:]=4.;A.eval();B=np.ones((2,1,3,3),dtype=np.float32)*100;D=A.forward(B);E=(1e2-A.running_mean)/np.sqrt(A.running_var+A.eps);F=E.reshape(1,-1,1,1)*np.ones_like(B);assert np.allclose(D,F,atol=1e-06);C=np.ones_like(B)*200;G=A.forward(C);H=(2e2-A.running_mean)/np.sqrt(A.running_var+A.eps);I=H.reshape(1,-1,1,1)*np.ones_like(C);assert np.allclose(G,I,atol=1e-06)
    A(T,'Step 3.1')
    def U():B=BatchNorm(2);A=np.random.randn(2,2,4,4).astype(np.float32);B.eval();C=B.forward(A);assert C.shape==A.shape;assert C.dtype==A.dtype
    A(U,'Step 3.2')
    def V():A=BatchNorm(1);A.running_mean[:]=2.;A.running_var[:]=9.;A.eval();B=np.ones((1,1,2,2),dtype=np.float32)*11;C=A.forward(B);D=9/np.sqrt(9+A.eps);assert np.allclose(C.mean(),D,atol=.001)
    A(V,'Step 3.3')
    def W():A=BatchNorm(1,momentum=C);A.train();B=np.full((4,1,3,3),1.23456789,dtype=np.float32);A.forward(B);A.eval();D=np.ones((4,1,3,3),dtype=np.float32);E=A.forward(D);F=(C-1.23456789)/np.sqrt(A.eps);G=abs(E.mean()-F);assert G<.0001
    A(W,'Step 4.1')
    def X():A=BatchNorm(1);A.beta=np.array([3.],dtype=np.float32);B=np.ones((2,1,2,2),dtype=np.float32);A.train();C=A.forward(B);assert np.allclose(C.mean(),3.,atol=.01)
    A(X,'Step 4.2')
    def Y():B=BatchNorm(1);C=np.array([[[[-1,-2],[-3,-4]]]],dtype=np.float32);B.train();A=B.forward(C);assert not np.isnan(A).any();assert np.allclose(A.mean(),0,atol=.001);assert np.allclose(A.std(),1,atol=.001)
    A(Y,'Step 4.3')
    def Z():A=BatchNorm(1);C=np.array([[[[1,2],[3,4]]]],dtype=np.float32);A.train();B=A.forward(C);assert np.allclose(B.mean(),0,atol=.001);assert np.allclose(B.std(),1,atol=.001)
    A(Z,'Step 4.4');B(f"\nTest suite finished: {D}/{G} tests passed.")

if __name__ == "__main__":
    run_tests()
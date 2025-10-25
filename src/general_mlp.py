import numpy as np
class General_MLP:
    def __init__(self, layer_sizes, activations, loss):
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.loss = loss
        self.L = len(layer_sizes)-1
        self.total_epochs = 0
        self.cache_A = []
        self.cache_Z = []
        self.grad_w = []
        self.grad_b = []
        self.losses = []
        self.weights = []
        self.biases = []

        for i in range(self.L):
            self.weights.append(np.random.randn(layer_sizes[i+1], layer_sizes[i]))
            self.biases.append(np.zeros((layer_sizes[i+1],1)))
        
    def forward(self, x):
        self.cache_A = [x]
        self.cache_Z = [x]

        for i in range(self.L):
            z = self.weights[i] @ x + self.biases[i]
            a = self.activations[0].f(z) if i < self.L-1 else self.activations[1].f(z)
            x = a
            self.cache_A.append(a)
            self.cache_Z.append(z)
        
    def backprop(self, y):
        batch_size = y.shape[1]
        self.grad_w = [None] * self.L  
        self.grad_b = [None] * self.L

        dL_dA = self.loss.df(y, self.cache_A[-1])  # (n_out * n_samples)
        for i in range(1, self.L+1):
            dA_dZ = self.activations[0].df(self.cache_Z[-i]) if i > 1 else self.activations[0].df(self.cache_Z[-i])  # (n_out * n_samples)
            dL_dZ = dL_dA * dA_dZ   # (n_out * n_samples)
            dZ_dW = self.cache_A[-i-1]   # (n_out_prev * n_samples)
            dL_dW = (dL_dZ @ dZ_dW.T) / batch_size   # (n_out * n_out_prev)
            dL_dB = np.sum(dL_dZ, axis=1, keepdims=True) / batch_size   # (n_out * 1)

            self.grad_w[-i] = dL_dW
            self.grad_b[-i] = dL_dB
            dL_dA = self.weights[-i].T @ dL_dZ # this is the dL/dA_i      (n_out_prev * n_samples)
    
    def update(self, lr):
        for i in range(self.L):
            self.weights[i] -= lr * self.grad_w[i]
            self.biases[i] -= lr * self.grad_b[i]
    
    def train(self, x, y, epoch, lr, batch_size=1, print_every = 100):
        total_batches = int(np.ceil(y.shape[1] / batch_size))
        for e in range(epoch):
            losses_batch = []
            for b in range(total_batches):
                curr_batch_start =  b * batch_size
                curr_batch_end = curr_batch_start + batch_size if curr_batch_start + batch_size < y.shape[1]  else y.shape[1]
                
                x_batch = x[ : , curr_batch_start : curr_batch_end]
                y_batch = y[ : , curr_batch_start : curr_batch_end]
                self.forward(x_batch)
                self.backprop(y_batch)
                self.update(lr)

                losses_batch.append(self.loss.f(y_batch, self.cache_A[-1]))

            self.losses.append(np.mean(losses_batch))
            if (e+1) % print_every == 0 :
                print(self.losses[self.total_epochs+e])
            
        self.total_epochs += epoch

    def predict(self, x):
        self.forward(x)
        return self.cache_A[-1]
        
    def save_model(self, path):
        np.savez(
            path,
            layers=np.array(self.layer_sizes),
            **{f"W{i}": w for i, w in enumerate(self.weights)},
            **{f"B{i}": b for i, b in enumerate(self.biases)},
        )
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        data = np.load(path)
        self.layer_sizes = data["layers"].tolist()
        self.weights = [data[f"W{i}"] for i in range(len(self.layer_sizes) - 1)]
        self.biases  = [data[f"B{i}"] for i in range(len(self.layer_sizes) - 1)]
        print(f"Model loaded from {path} ")
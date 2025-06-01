import pickle
import matplotlib.pyplot as plt
import numpy as np
import math

# Live plotting 
class LivePlotter:
    def __init__(self):
        self.points = []
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss (MSE)')

    def add_point(self, x, y):
        self.points.append((x, y))
        self.ax.clear()
        x_vals, y_vals = zip(*self.points)
        self.ax.set_xlim(0, max(x_vals) + 1)
        self.ax.set_ylim(0, max(y_vals) * 1.1)
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss (MSE)')
        self.ax.axhline(0, color='red', linestyle='--')
        self.ax.plot(x_vals, y_vals, marker='o', label='Loss')
        if len(self.points) > 1:
            # simple finite‑difference derivative of loss curve
            dy = [(y_vals[i+1] - y_vals[i]) / (x_vals[i+1] - x_vals[i])
                  for i in range(len(y_vals) - 1)]
            self.ax.plot(x_vals[1:], dy, marker='x', color='green', label='dLoss/dEpoch')
        self.ax.legend()
        plt.draw()
        plt.pause(0.001)

    def show(self):
        plt.ioff()
        plt.show()

# Save / load functions
def save_model(network, filename='brain_model_adam_loops.pkl'):
    model_data = {
        'weights': network.w_ih,
        'biases': network.b_h,
        'weights_hidden_output': network.w_ho,
        'biases_output': network.b_o,
    }
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    print(f'Model saved to {filename}')


def load_model(network, filename='brain_model_adam_loops.pkl'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    network.w_ih = data['weights']
    network.b_h = data['biases']
    network.w_ho = data['weights_hidden_output']
    network.b_o = data['biases_output']
    print(f'Model loaded from {filename}')

# Activation functions
def relu(x: float) -> float:
    return x if x > 0 else 0.0


def relu_derivative(x: float) -> float:
    return 1.0 if x > 0 else 0.0


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(x: float) -> float:
    s = sigmoid(x)
    return s * (1.0 - s)


# Two‑layer MLP with manual‑loop Adam optimiser
class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 learning_rate: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        self.in_size = input_size
        self.hid_size = hidden_size
        self.out_size = output_size
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

        # Parameter tensors (NumPy just for storage convenience)
        self.w_ih = np.random.randn(input_size, hidden_size)
        self.b_h = np.random.randn(hidden_size)
        self.w_ho = np.random.randn(hidden_size, output_size)
        self.b_o = np.random.randn(output_size)

        # First (m) and second (v) moment estimates, same shapes as params
        self.m_w_ih = np.zeros_like(self.w_ih)
        self.v_w_ih = np.zeros_like(self.w_ih)
        self.m_b_h = np.zeros_like(self.b_h)
        self.v_b_h = np.zeros_like(self.b_h)
        self.m_w_ho = np.zeros_like(self.w_ho)
        self.v_w_ho = np.zeros_like(self.w_ho)
        self.m_b_o = np.zeros_like(self.b_o)
        self.v_b_o = np.zeros_like(self.b_o)

    # --------------------------- forward pass (single sample) ---------------------------
    def forward(self, x_sample):
        if isinstance(x_sample, np.ndarray):
            self.x = x_sample.tolist()
        else:
            self.x = list(x_sample)

        self.h_in = [0.0] * self.hid_size
        for j in range(self.hid_size):
            total = self.b_h[j]
            for i in range(self.in_size):
                total += self.x[i] * self.w_ih[i][j]
            self.h_in[j] = total

        self.h_out = [relu(h) for h in self.h_in]

        self.o_in = [0.0] * self.out_size
        for k in range(self.out_size):
            total = self.b_o[k]
            for j in range(self.hid_size):
                total += self.h_out[j] * self.w_ho[j][k]
            self.o_in[k] = total

        self.y_hat = [sigmoid(o) for o in self.o_in]
        return self.y_hat

    # ---------------- loss --------------
    def mse(self, target):
        error = 0.0
        for k in range(self.out_size):
            diff = target[k] - self.y_hat[k]
            error += diff * diff
        return error / self.out_size

    # ------------- backward pass + Adam -------
    def backpropagate(self, target):
        #deltas for output layer
        delta_out = [0.0] * self.out_size
        for k in range(self.out_size):
            error = target[k] - self.y_hat[k]
            delta_out[k] = error * sigmoid_derivative(self.o_in[k])

        #deltas for hidden layer
        delta_hid = [0.0] * self.hid_size
        for j in range(self.hid_size):
            err = 0.0
            for k in range(self.out_size):
                err += delta_out[k] * self.w_ho[j][k]
            delta_hid[j] = err * relu_derivative(self.h_in[j])

        #gradients (input to hidden)
        grad_w_ih = np.zeros_like(self.w_ih)
        grad_b_h = np.zeros_like(self.b_h)
        for i in range(self.in_size):
            for j in range(self.hid_size):
                grad_w_ih[i][j] = self.x[i] * delta_hid[j]
        for j in range(self.hid_size):
            grad_b_h[j] = delta_hid[j]

        #gradients (hidden to output)
        grad_w_ho = np.zeros_like(self.w_ho)
        grad_b_o = np.zeros_like(self.b_o)
        for j in range(self.hid_size):
            for k in range(self.out_size):
                grad_w_ho[j][k] = self.h_out[j] * delta_out[k]
        for k in range(self.out_size):
            grad_b_o[k] = delta_out[k]

        self.t += 1
        lr_t = self.lr * (math.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t))

        #helper for param arrays
        def adam_step(param, grad, m, v):
            m *= self.beta1
            m += (1 - self.beta1) * grad
            v *= self.beta2
            v += (1 - self.beta2) * (grad ** 2)
            param += lr_t * m / (np.sqrt(v) + self.epsilon)
            return param, m, v

        #input to hidden weights
        for i in range(self.in_size):
            for j in range(self.hid_size):
                g = grad_w_ih[i][j]
                self.m_w_ih[i][j] = self.beta1 * self.m_w_ih[i][j] + (1 - self.beta1) * g
                self.v_w_ih[i][j] = self.beta2 * self.v_w_ih[i][j] + (1 - self.beta2) * (g * g)
                m_hat = self.m_w_ih[i][j] / (1 - self.beta1 ** self.t)
                v_hat = self.v_w_ih[i][j] / (1 - self.beta2 ** self.t)
                self.w_ih[i][j] += lr_t * m_hat / (math.sqrt(v_hat) + self.epsilon)

        #hidden biases
        for j in range(self.hid_size):
            g = grad_b_h[j]
            self.m_b_h[j] = self.beta1 * self.m_b_h[j] + (1 - self.beta1) * g
            self.v_b_h[j] = self.beta2 * self.v_b_h[j] + (1 - self.beta2) * (g * g)
            m_hat = self.m_b_h[j] / (1 - self.beta1 ** self.t)
            v_hat = self.v_b_h[j] / (1 - self.beta2 ** self.t)
            self.b_h[j] += lr_t * m_hat / (math.sqrt(v_hat) + self.epsilon)

        #hidden to output weights
        for j in range(self.hid_size):
            for k in range(self.out_size):
                g = grad_w_ho[j][k]
                self.m_w_ho[j][k] = self.beta1 * self.m_w_ho[j][k] + (1 - self.beta1) * g
                self.v_w_ho[j][k] = self.beta2 * self.v_w_ho[j][k] + (1 - self.beta2) * (g * g)
                m_hat = self.m_w_ho[j][k] / (1 - self.beta1 ** self.t)
                v_hat = self.v_w_ho[j][k] / (1 - self.beta2 ** self.t)
                self.w_ho[j][k] += lr_t * m_hat / (math.sqrt(v_hat) + self.epsilon)

        #output biases
        for k in range(self.out_size):
            g = grad_b_o[k]
            self.m_b_o[k] = self.beta1 * self.m_b_o[k] + (1 - self.beta1) * g
            self.v_b_o[k] = self.beta2 * self.v_b_o[k] + (1 - self.beta2) * (g * g)
            m_hat = self.m_b_o[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v_b_o[k] / (1 - self.beta2 ** self.t)
            self.b_o[k] += lr_t * m_hat / (math.sqrt(v_hat) + self.epsilon)


#Training loop
if __name__ == '__main__':
    #Dummy data
    X = np.array([[0.1, 0.2, 0.3],
                  [0.4, 0.5, 0.6],
                  [0.7, 0.8, 0.9]])
    Y = np.array([[0, 1],
                  [0, 1],
                  [1, 0]])

    net = NeuralNetwork(input_size=3, hidden_size=5, output_size=2, learning_rate=1e-3)
    plotter = LivePlotter()

    EPOCHS = 2000
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for idx in range(len(X)):
            y_hat = net.forward(X[idx])
            total_loss += net.mse(Y[idx])
            net.backpropagate(Y[idx])

        avg_loss = total_loss / len(X)
        print(f'Epoch {epoch} | Loss: {avg_loss:.6f}')
        plotter.add_point(epoch, avg_loss)

    print('Training complete!')
    save_model(net)
    plotter.show()

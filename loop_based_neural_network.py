import pickle
import matplotlib.pyplot as plt
import numpy as np
import math

class LivePlotter:
    def __init__(self):
        self.points = []
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.x_label = 'Epoch'
        self.y_label = 'Loss'

    def add_point(self, x, y):
        """Add a (x, y) point to the live plot."""
        self.points.append((x, y))
        self.ax.clear()

        if not self.points:
            return

        x_values, y_values = zip(*self.points)
        self.ax.set_xlim(min(x_values) - 1, max(x_values) + 1)
        self.ax.set_ylim(min(y_values) - 1, max(y_values) + 1)

        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        self.ax.axhline(0, color='red', linestyle='--')
        self.ax.plot(*zip(*self.points), marker='o')
        plt.draw()
        plt.pause(0.001)

    def show(self):
        """Block and keep the final plot on screen."""
        plt.ioff()
        plt.show()


def save_model(network, filename='brain_model.pkl'):
    model_data = {
        'weights': network.weights_input_hidden,
        'biases': network.bias_hidden,
        'weights_hidden_output': network.weights_hidden_output,
        'biases_output': network.bias_output,
    }
    with open(filename, 'wb') as file:
        pickle.dump(model_data, file)
    print(f'Model saved to {filename}')


def load_model(network, filename='brain_model.pkl'):
    with open(filename, 'rb') as file:
        model_data = pickle.load(file)
    network.weights_input_hidden = model_data['weights']
    network.bias_hidden = model_data['biases']
    network.weights_hidden_output = model_data['weights_hidden_output']
    network.bias_output = model_data['biases_output']
    print(f'Model loaded from {filename}')


#-------------Activation functions and their derivatives-------------
def ReLU(x):
    return x if x > 0 else 0

def ReLU_derivative(x):
    return 1 if x > 0 else 0

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=1e-3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Randomly initialise weights & biases (NumPy only for RNG / storage)
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.random.rand(hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.random.rand(output_size)

    # --------------- forward pass ------------------
    def forward(self, inputs):
        if isinstance(inputs, np.ndarray):
            self.inputs = inputs.tolist()
        else:
            self.inputs = list(inputs)

        #Hidden layer
        self.hidden_input = [0.0] * self.hidden_size
        for j in range(self.hidden_size):
            total = self.bias_hidden[j]
            for i in range(self.input_size):
                total += self.inputs[i] * self.weights_input_hidden[i][j]
            self.hidden_input[j] = total

        self.hidden_output = [ReLU(h) for h in self.hidden_input]

        #Output layer
        self.output_input = [0.0] * self.output_size
        for k in range(self.output_size):
            total = self.bias_output[k]
            for j in range(self.hidden_size):
                total += self.hidden_output[j] * self.weights_hidden_output[j][k]
            self.output_input[k] = total

        self.final_output = [sigmoid(o) for o in self.output_input]
        return self.final_output

    # -------------------- mean‑squared error --------------------
    def mean_squared_error(self, target):
        error = 0.0
        for k in range(self.output_size):
            diff = target[k] - self.final_output[k]
            error += diff * diff
        self.MSE = error / self.output_size
        return self.MSE

    # ------------ back‑prop ------------
    def backpropagation(self, target):
        #Output layer delta
        error_output = [target[k] - self.final_output[k] for k in range(self.output_size)]
        delta_output = [error_output[k] * ReLU_derivative(self.output_input[k])
                        for k in range(self.output_size)]

        #Hidden layer delta
        error_hidden = [0.0] * self.hidden_size
        for j in range(self.hidden_size):
            for k in range(self.output_size):
                error_hidden[j] += delta_output[k] * self.weights_hidden_output[j][k]

        delta_hidden = [error_hidden[j] * sigmoid_derivative(self.hidden_input[j])
                        for j in range(self.hidden_size)]

        #Update weights and biases (input to hidden)
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.weights_input_hidden[i][j] += (
                    self.learning_rate * self.inputs[i] * delta_hidden[j]
                )

        for j in range(self.hidden_size):
            self.bias_hidden[j] += self.learning_rate * delta_hidden[j]

        #Update weights and biases (hidden to output)
        for j in range(self.hidden_size):
            for k in range(self.output_size):
                self.weights_hidden_output[j][k] += (
                    self.learning_rate * self.hidden_output[j] * delta_output[k]
                )

        for k in range(self.output_size):
            self.bias_output[k] += self.learning_rate * delta_output[k]


#--------------------------Training loop-----------------
if __name__ == '__main__':
    #Dummy data
    x = np.array([[0.1, 0.2, 0.3],
                  [0.4, 0.5, 0.6],
                  [0.7, 0.8, 0.9]])

    y = np.array([[0, 1],
                  [0, 1],
                  [1, 0]])

    plotter = LivePlotter()
    network = NeuralNetwork(input_size=3, hidden_size=5, output_size=2, learning_rate=1e-3)

    EPOCHS = 200
    for epoch in range(EPOCHS):
        total_loss = 0.0

        for sample_idx in range(len(x)):
            outputs = network.forward(x[sample_idx])
            total_loss += network.mean_squared_error(y[sample_idx])
            network.backpropagation(y[sample_idx])

        average_loss = total_loss / len(x)
        print(f'Epoch {epoch + 1}, Average Loss: {average_loss}')
        plotter.add_point(epoch, average_loss)

    print('Model training complete!')
    save_model(network, 'brain_model2.pkl')

// 				from_perceptrons_to_DNNs_
/*
Simple Perceptron in C
A perceptron is a basic artificial neuron that takes multiple inputs, applies weights, sums them, and passes the result through an activation function (usually a step function).

Simple Implementation

This minimal perceptron uses a fixed learning rate and binary step activation.

How It Works
Takes two inputs and two weights.
Computes weighted sum + bias.
Passes result through step function (1 if positive, 0 otherwise).



#include <stdio.h>

// Activation function: Step function
int step_function(float x) {
	return (x >= 0) ? 1 : 0;
}

// Perceptron function
int perceptron(float inputs[], float weights[], float bias, int num_inputs) {
	float sum = bias;
	for (int i = 0; i < num_inputs; i++) {
		sum += inputs[i] * weights[i];
	}
	return step_function(sum);
}

int main() {
	float inputs[] = {0, 1};   // Example input
	float weights[] = {0.5, -0.5}; // Example weights
	float bias = 0.1;
	
	int output = perceptron(inputs, weights, bias, 2);
	printf("Perceptron Output: %d\n", output);
	
	return 0;
}
	
*//*

Advanced Perceptron with Learning & Multiple Training Samples

This version: ✅ Supports training with multiple samples
✅ Uses dynamic memory allocation for weights
✅ Implements learning rule (weight update)

*//*

#include <stdio.h>
#include <stdlib.h>

// Step activation function
int step_function(float x) {
	return (x >= 0) ? 1 : 0;
}

// Perceptron function
int perceptron(float inputs[], float weights[], float bias, int num_inputs) {
	float sum = bias;
	for (int i = 0; i < num_inputs; i++) {
		sum += inputs[i] * weights[i];
	}
	return step_function(sum);
}

// Perceptron Training
void train_perceptron(float **training_inputs, int labels[], float *weights, float *bias, int num_samples, int num_inputs, float learning_rate, int epochs) {
	for (int epoch = 0; epoch < epochs; epoch++) {
		int total_error = 0;
		for (int i = 0; i < num_samples; i++) {
			int output = perceptron(training_inputs[i], weights, *bias, num_inputs);
			int error = labels[i] - output;
			total_error += abs(error);
			
			// Update weights and bias using Perceptron Learning Rule
			for (int j = 0; j < num_inputs; j++) {
				weights[j] += learning_rate * error * training_inputs[i][j];
			}
			*bias += learning_rate * error;
		}
		if (total_error == 0) break;  // Stop if all are classified correctly
	}
}

int main() {
	int num_samples = 4;
	int num_inputs = 2;
	
	// Training dataset (Logical AND function)
	float training_data[4][2] = {
		{0, 0}, {0, 1}, {1, 0}, {1, 1}
	};
	int labels[4] = {0, 0, 0, 1};  // Expected outputs for AND function
	
	// Convert to dynamic memory (for larger datasets)
	float **training_inputs = malloc(num_samples * sizeof(float *));
	for (int i = 0; i < num_samples; i++) {
		training_inputs[i] = malloc(num_inputs * sizeof(float));
		for (int j = 0; j < num_inputs; j++) {
			training_inputs[i][j] = training_data[i][j];
		}
	}
	
	// Initialize weights and bias randomly
	float weights[2] = {0.0, 0.0};
	float bias = 0.0;
	float learning_rate = 0.1;
	int epochs = 10;
	
	// Train the perceptron
	train_perceptron(training_inputs, labels, weights, &bias, num_samples, num_inputs, learning_rate, epochs);
	
	// Test the perceptron
	printf("Testing the trained perceptron:\n");
	for (int i = 0; i < num_samples; i++) {
		int output = perceptron(training_inputs[i], weights, bias, num_inputs);
		printf("Input: (%d, %d) => Output: %d\n", (int)training_inputs[i][0], (int)training_inputs[i][1], output);
	}
	
	// Free memory
	for (int i = 0; i < num_samples; i++) {
		free(training_inputs[i]);
	}
	free(training_inputs);
	
	return 0;
}

*//*

Multi-Layer Perceptron (MLP) in C (With Backpropagation)
Now, let’s extend the perceptron into a Multi-Layer Perceptron (MLP) with Backpropagation for training.

🚀 Features of This Implementation:

✅ Supports one hidden layer
✅ Uses Sigmoid activation function
✅ Implements Backpropagation for learning
✅ Uses dynamic memory allocation for scalability


1️⃣ Understanding the MLP Structure
An MLP consists of:

Input Layer (Receives input values)
Hidden Layer (Extracts features using activation functions)
Output Layer (Classifies or regresses outputs)
We'll implement an MLP with one hidden layer, using the Sigmoid function for activation.

2️⃣ Activation Functions
We'll use Sigmoid for smooth gradient updates:

σ(x)=1/(1+e^−)

Derivative:

σ(x)=σ(x)×(1−σ(x))

*//*

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Sigmoid activation function
double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

// Derivative of the Sigmoid function
double sigmoid_derivative(double x) {
	return x * (1.0 - x);  // Uses already computed sigmoid output
}

*//*

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_NODES 2
#define HIDDEN_NODES 2
#define OUTPUT_NODES 1
#define LEARNING_RATE 0.5
#define EPOCHS 10000

// Sigmoid activation function
double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

// Derivative of the Sigmoid function
double sigmoid_derivative(double x) {
	return x * (1.0 - x);  // Uses already computed sigmoid output
}

// Forward propagation function
void forward_propagation(double inputs[], double hidden_weights[][HIDDEN_NODES], double hidden_bias[],
	double output_weights[], double output_bias, double hidden_layer[], double *output) {
		// Compute hidden layer activations
		for (int i = 0; i < HIDDEN_NODES; i++) {
			hidden_layer[i] = hidden_bias[i];
			for (int j = 0; j < INPUT_NODES; j++) {
				hidden_layer[i] += inputs[j] * hidden_weights[j][i];
			}
			hidden_layer[i] = sigmoid(hidden_layer[i]);
		}
		
		// Compute output layer activation
		*output = output_bias;
		for (int i = 0; i < HIDDEN_NODES; i++) {
			*output += hidden_layer[i] * output_weights[i];
		}
		*output = sigmoid(*output);
	}

// Backpropagation function (Training step)
void backpropagation(double inputs[], double target, double hidden_weights[][HIDDEN_NODES], double hidden_bias[],
	double output_weights[], double *output_bias, double hidden_layer[], double output) {
		// Compute output error
		double output_error = target - output;
		double output_delta = output_error * sigmoid_derivative(output);
		
		// Compute hidden layer error
		double hidden_deltas[HIDDEN_NODES];
		for (int i = 0; i < HIDDEN_NODES; i++) {
			hidden_deltas[i] = output_delta * output_weights[i] * sigmoid_derivative(hidden_layer[i]);
		}
		
		// Update output weights and bias
		for (int i = 0; i < HIDDEN_NODES; i++) {
			output_weights[i] += LEARNING_RATE * output_delta * hidden_layer[i];
		}
		*output_bias += LEARNING_RATE * output_delta;
		
		// Update hidden weights and biases
		for (int i = 0; i < HIDDEN_NODES; i++) {
			for (int j = 0; j < INPUT_NODES; j++) {
				hidden_weights[j][i] += LEARNING_RATE * hidden_deltas[i] * inputs[j];
			}
			hidden_bias[i] += LEARNING_RATE * hidden_deltas[i];
		}
	}

int main() {
	// Training data: XOR Problem
	double training_inputs[4][INPUT_NODES] = {
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1}
	};
	double training_outputs[4] = {0, 1, 1, 0}; // XOR Truth Table
	
	// Initialize weights randomly
	double hidden_weights[INPUT_NODES][HIDDEN_NODES] = {{0.5, -0.7}, {0.3, 0.8}};
	double output_weights[HIDDEN_NODES] = {0.6, -0.4};
	
	// Initialize biases randomly
	double hidden_bias[HIDDEN_NODES] = {0.2, -0.3};
	double output_bias = 0.1;
	
	// Training phase
	for (int epoch = 0; epoch < EPOCHS; epoch++) {
		for (int i = 0; i < 4; i++) {
			double hidden_layer[HIDDEN_NODES];
			double output;
			
			// Forward pass
			forward_propagation(training_inputs[i], hidden_weights, hidden_bias, output_weights, output_bias, hidden_layer, &output);
			
			// Backpropagation (training)
			backpropagation(training_inputs[i], training_outputs[i], hidden_weights, hidden_bias, output_weights, &output_bias, hidden_layer, output);
		}
	}
	
	// Testing phase
	printf("Testing MLP on XOR problem:\n");
	for (int i = 0; i < 4; i++) {
		double hidden_layer[HIDDEN_NODES];
		double output;
		
		forward_propagation(training_inputs[i], hidden_weights, hidden_bias, output_weights, output_bias, hidden_layer, &output);
		printf("Input: (%.0f, %.0f) => Output: %.4f\n", training_inputs[i][0], training_inputs[i][1], output);
	}
	
	return 0;
}

*//*
Output:
Testing MLP on XOR problem:
Input: (0, 0) => Output: 0.0194
Input: (0, 1) => Output: 0.9834
Input: (1, 0) => Output: 0.9832
Input: (1, 1) => Output: 0.0174

*/
//Deep Neural Network:
/*
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_NODES 20      // Input layer size
#define HIDDEN_LAYERS 200    // Number of hidden layers
#define HIDDEN_NODES 40     // Nodes per hidden layer
#define OUTPUT_NODES 10     // Output layer size
#define LEARNING_RATE 0.5
#define EPOCHS 1000

// Activation functions
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double sigmoid_derivative(double x) { return x * (1.0 - x); }

// Neural network structure
typedef struct {
	double weights[HIDDEN_NODES][HIDDEN_NODES];  // Weights between layers
	double biases[HIDDEN_NODES];                 // Biases per layer
} Layer;

// Initialize weights and biases randomly
void init_layer(Layer *layer, int input_size) {
	for (int i = 0; i < HIDDEN_NODES; i++) {
		layer->biases[i] = ((double)rand() / RAND_MAX) - 0.5;
		for (int j = 0; j < input_size; j++) {
			layer->weights[j][i] = ((double)rand() / RAND_MAX) - 0.5;
		}
	}
}

// Forward propagation
void forward(Layer *layers, double inputs[], double outputs[][HIDDEN_NODES], int total_layers) {
	for (int l = 0; l < total_layers; l++) {
		double *prev = (l == 0) ? inputs : outputs[l - 1];  // Input for first layer, previous for others
		for (int i = 0; i < HIDDEN_NODES; i++) {
			outputs[l][i] = layers[l].biases[i];
			for (int j = 0; j < ((l == 0) ? INPUT_NODES : HIDDEN_NODES); j++) {
				outputs[l][i] += prev[j] * layers[l].weights[j][i];
			}
			outputs[l][i] = sigmoid(outputs[l][i]);
		}
	}
}

// Backpropagation (Error correction)
void backpropagate(Layer *layers, double inputs[], double target, double outputs[][HIDDEN_NODES], double output_weights[], double *output_bias) {
	// Compute output error
	double output = sigmoid(outputs[HIDDEN_LAYERS - 1][0] * output_weights[0] + *output_bias);
	double output_error = target - output;
	double output_delta = output_error * sigmoid_derivative(output);
	
	// Compute hidden layer errors
	double deltas[HIDDEN_LAYERS][HIDDEN_NODES];
	for (int l = HIDDEN_LAYERS - 1; l >= 0; l--) {
		for (int i = 0; i < HIDDEN_NODES; i++) {
			deltas[l][i] = (l == HIDDEN_LAYERS - 1 ? output_delta * output_weights[i] : 0) * sigmoid_derivative(outputs[l][i]);
		}
	}
	
	// Update output weights
	for (int i = 0; i < HIDDEN_NODES; i++) {
		output_weights[i] += LEARNING_RATE * output_delta * outputs[HIDDEN_LAYERS - 1][i];
	}
	*output_bias += LEARNING_RATE * output_delta;
	
	// Update hidden layer weights
	for (int l = 0; l < HIDDEN_LAYERS; l++) {
		double *prev = (l == 0) ? inputs : outputs[l - 1];
		for (int i = 0; i < HIDDEN_NODES; i++) {
			for (int j = 0; j < ((l == 0) ? INPUT_NODES : HIDDEN_NODES); j++) {
				layers[l].weights[j][i] += LEARNING_RATE * deltas[l][i] * prev[j];
			}
			layers[l].biases[i] += LEARNING_RATE * deltas[l][i];
		}
	}
}

int main() {
	srand(time(NULL));
	
	// Training data (XOR problem)
	double training_inputs[4][INPUT_NODES] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
	double training_outputs[4] = {0, 1, 1, 0};
	
	// Initialize hidden layers
	Layer layers[HIDDEN_LAYERS];
	for (int i = 0; i < HIDDEN_LAYERS; i++) {
		init_layer(&layers[i], (i == 0) ? INPUT_NODES : HIDDEN_NODES);
	}
	
	// Initialize output weights
	double output_weights[HIDDEN_NODES];
	for (int i = 0; i < HIDDEN_NODES; i++) {
		output_weights[i] = ((double)rand() / RAND_MAX) - 0.5;
	}
	double output_bias = ((double)rand() / RAND_MAX) - 0.5;
	
	// Training loop
	for (int epoch = 0; epoch < EPOCHS; epoch++) {
		for (int i = 0; i < 4; i++) {
			double outputs[HIDDEN_LAYERS][HIDDEN_NODES];
			
			// Forward pass
			forward(layers, training_inputs[i], outputs, HIDDEN_LAYERS);
			
			// Backpropagation
			backpropagate(layers, training_inputs[i], training_outputs[i], outputs, output_weights, &output_bias);
		}
	}
	
	// Testing trained model
	printf("Testing Deep Learning MLP on XOR problem:\n");
	for (int i = 0; i < 4; i++) {
		double outputs[HIDDEN_LAYERS][HIDDEN_NODES];
		forward(layers, training_inputs[i], outputs, HIDDEN_LAYERS);
		double output = sigmoid(outputs[HIDDEN_LAYERS - 1][0] * output_weights[0] + output_bias);
		printf("Input: (%.0f, %.0f) => Output: %.4f\n", training_inputs[i][0], training_inputs[i][1], output);
	}
	
	return 0;
}
*/
//optimized:
/*
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_NODES 2
#define HIDDEN_NODES 4
#define OUTPUT_NODES 1
#define LEARNING_RATE 0.1
#define EPOCHS 10000

// Activation functions
double relu(double x) { return (x > 0) ? x : 0; }
double relu_derivative(double x) { return (x > 0) ? 1 : 0; }
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double sigmoid_derivative(double x) { return x * (1.0 - x); }

// Neural network structure
typedef struct {
	double weights[INPUT_NODES][HIDDEN_NODES];
	double biases[HIDDEN_NODES];
} Layer;

// Xavier initialization
void init_layer(Layer *layer) {
	double limit = sqrt(2.0 / INPUT_NODES);
	for (int i = 0; i < HIDDEN_NODES; i++) {
		layer->biases[i] = ((double)rand() / RAND_MAX - 0.5) * limit;
		for (int j = 0; j < INPUT_NODES; j++) {
			layer->weights[j][i] = ((double)rand() / RAND_MAX - 0.5) * limit;
		}
	}
}

// Forward propagation
void forward(Layer *layer, double inputs[], double outputs[HIDDEN_NODES]) {
	for (int i = 0; i < HIDDEN_NODES; i++) {
		outputs[i] = layer->biases[i];
		for (int j = 0; j < INPUT_NODES; j++) {
			outputs[i] += inputs[j] * layer->weights[j][i];
		}
		outputs[i] = relu(outputs[i]);
	}
}

// Training function
void train(Layer *layer, double inputs[][INPUT_NODES], double targets[], int samples) {
	for (int epoch = 0; epoch < EPOCHS; epoch++) {
		for (int s = 0; s < samples; s++) {
			double hidden_outputs[HIDDEN_NODES];
			
			// Forward pass
			forward(layer, inputs[s], hidden_outputs);
			double output = sigmoid(hidden_outputs[0]);
			
			// Compute error
			double error = targets[s] - output;
			double delta_output = error * sigmoid_derivative(output);
			
			// Backpropagate error to hidden layer
			for (int i = 0; i < HIDDEN_NODES; i++) {
				double delta_hidden = delta_output * relu_derivative(hidden_outputs[i]);
				for (int j = 0; j < INPUT_NODES; j++) {
					layer->weights[j][i] += LEARNING_RATE * delta_hidden * inputs[s][j];
				}
				layer->biases[i] += LEARNING_RATE * delta_hidden;
			}
		}
	}
}

int main() {
	srand(time(NULL));
	
	// XOR dataset
	double training_inputs[4][INPUT_NODES] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
	double training_outputs[4] = {0, 1, 1, 0};
	
	// Initialize network
	Layer layer;
	init_layer(&layer);
	
	// Train network
	train(&layer, training_inputs, training_outputs, 4);
	
	// Test network
	printf("Testing XOR Perceptron:\n");
	for (int i = 0; i < 4; i++) {
		double hidden_outputs[HIDDEN_NODES];
		forward(&layer, training_inputs[i], hidden_outputs);
		double output = sigmoid(hidden_outputs[0]);
		printf("Input: (%.0f, %.0f) => Output: %.4f\n", training_inputs[i][0], training_inputs[i][1], output);
	}
	
	return 0;
}
*//*

chatgpt reasoning :

The user asked to improve on the previous response, which involved optimizing code with mini-batch training for training a perceptron on XOR. The code already uses mini-batches, so I'll inspect it to identify potential improvements.
For example, I should check if any enhancements or optimizations can be made to the learning rate, batch size, or activation functions. We need to review the code closely to ensure it's as efficient as possible while properly handling mini-batch updates. Let's move forward on this!
The current structure is a perceptron, but it lacks a full multi-layer perceptron (MLP) design with separate hidden and output layers. To improve, I'll add an output layer and properly connect it to the hidden layer. This entails creating a separate layer for output, adjusting the forward propagation to include both layers, and adding backpropagation that updates both the hidden and output layer weights. I'll modify the training function to reflect this new structure and ensure proper weight updates. Let's go ahead and make these adjustments!
The current code has one hidden layer, but no separate output layer, which isn't ideal for a standard MLP. To fix this, I’ll restructure the network to have two separate layers: a hidden layer and an output layer.
This will involve defining a new Layer struct that accommodates both hidden and output layers. The forward and backward passes will use standard activation and error propagation, updating weights and biases with mini-batch gradient descent. I'll also make sure the code is clear, properly structured, and includes improvements like Xavier initialization and necessary intermediate storage. Let's move forward with this.
I realize that adding mini-batch gradient updates and vectorized operations will improve performance. I'll make the training more efficient by averaging gradients over a mini-batch. Also, I'll define a Layer struct for both the hidden and output layers, storing weights and biases in arrays. The forward and backward propagation will handle ReLU for the hidden layer and Sigmoid for the output.
By using Xavier initialization for both layers, I'll optimize this for the XOR dataset with mini-batch size set to 2. This should improve the structure significantly from the previous version. Let's proceed!

Below is an improved version that implements a complete two‐layer neural network (one hidden layer plus one output layer) with mini-batch training. This version uses:

Xavier/Glorot weight initialization
ReLU activation for the hidden layer and Sigmoid for the output
A clean separation of the forward pass, backpropagation, and mini‐batch gradient update
Detailed (but concise) comments for clarity
Compile and run this C code to train the network on the XOR problem.

Summary of Improvements:
Network Structure:
Two layers: a hidden layer (with ReLU activation) and an output layer (with Sigmoid activation).
Proper weight and bias arrays for both layers.
Mini-Batch Training:
The code loops over mini-batches (here, batch size is 2) instead of updating for each sample.
Gradients for weights and biases are accumulated for all samples in the mini-batch, then averaged before updating.
Weight Initialization:
Xavier initialization is applied to both layers for more stable convergence.
Comments:
The comments are concise yet explain the key operations (forward pass, backpropagation, gradient accumulation, parameter updates).
Compile this code (for example, with gcc -o dnn dnn.c -lm) and run it. You should see the network learning the XOR function with outputs approaching 0 or 1 for the four possible inputs.

Feel free to adjust hyperparameters (like learning rate, epochs, or batch size) to further experiment with performance and convergence!

*//*

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Network hyperparameters
#define INPUT_NODES    2
#define HIDDEN_NODES   4
#define OUTPUT_NODES   1
#define LEARNING_RATE  0.1
#define EPOCHS         10000
#define BATCH_SIZE     2  // Mini-batch size

// Activation functions and derivatives
double relu(double x) {
	return (x > 0) ? x : 0;
}

double relu_derivative(double x) {
	return (x > 0) ? 1 : 0;
}

double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

// Note: Here we assume 'x' is the output of the sigmoid function.
double sigmoid_derivative(double x) {
	return x * (1.0 - x);
}

// Initialize weights with Xavier/Glorot initialization.
// For a matrix with dimensions (rows x cols) and input_size used in scaling:
void init_weights(double *weights, int rows, int cols, int input_size) {
	double limit = sqrt(2.0 / input_size);
	for (int i = 0; i < rows * cols; i++) {
		weights[i] = (((double)rand() / RAND_MAX) - 0.5) * limit;
	}
}

int main(void) {
	srand(time(NULL));
	
	// XOR training dataset: 4 samples, each with 2 inputs, and one target output.
	double training_inputs[4][INPUT_NODES] = {
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1}
	};
	double training_targets[4] = {0, 1, 1, 0};
	
	// Allocate network parameters.
	// Layer 1 (Input -> Hidden)
	double weights1[INPUT_NODES * HIDDEN_NODES];
	double biases1[HIDDEN_NODES];
	// Layer 2 (Hidden -> Output)
	double weights2[HIDDEN_NODES * OUTPUT_NODES]; // OUTPUT_NODES == 1, so essentially HIDDEN_NODES weights.
	double biases2[OUTPUT_NODES];
	
	// Initialize weights and biases.
	init_weights(weights1, INPUT_NODES, HIDDEN_NODES, INPUT_NODES);
	init_weights(weights2, HIDDEN_NODES, OUTPUT_NODES, HIDDEN_NODES);
	for (int j = 0; j < HIDDEN_NODES; j++) {
		biases1[j] = 0.0;
	}
	for (int k = 0; k < OUTPUT_NODES; k++) {
		biases2[k] = 0.0;
	}
	
	// TRAINING: Loop over epochs and process mini-batches.
	for (int epoch = 0; epoch < EPOCHS; epoch++) {
		// For each mini-batch...
		for (int batch_start = 0; batch_start < 4; batch_start += BATCH_SIZE) {
			// Accumulators for gradients.
			double grad_w1[INPUT_NODES * HIDDEN_NODES] = {0};
			double grad_b1[HIDDEN_NODES] = {0};
			double grad_w2[HIDDEN_NODES * OUTPUT_NODES] = {0}; // Essentially HIDDEN_NODES values.
			double grad_b2[OUTPUT_NODES] = {0};
			
			int batch_end = batch_start + BATCH_SIZE;
			if (batch_end > 4) batch_end = 4;
			int current_batch_size = batch_end - batch_start;
			
			// Process each sample in the mini-batch.
			for (int s = batch_start; s < batch_end; s++) {
				// --- FORWARD PASS ---
				double input[INPUT_NODES];
				double hidden[HIDDEN_NODES];
				double output[OUTPUT_NODES];
				
				// Copy sample input.
				for (int i = 0; i < INPUT_NODES; i++) {
					input[i] = training_inputs[s][i];
				}
				
				// Hidden layer: compute weighted sum + bias then apply ReLU.
				for (int j = 0; j < HIDDEN_NODES; j++) {
					double sum = biases1[j];
					for (int i = 0; i < INPUT_NODES; i++) {
						// weights1 is stored in row-major order: index = i * HIDDEN_NODES + j.
						sum += input[i] * weights1[i * HIDDEN_NODES + j];
					}
					hidden[j] = relu(sum);
				}
				
				// Output layer: compute weighted sum + bias then apply Sigmoid.
				for (int k = 0; k < OUTPUT_NODES; k++) {
					double sum = biases2[k];
					for (int j = 0; j < HIDDEN_NODES; j++) {
						// weights2 is stored in row-major order: index = j * OUTPUT_NODES + k.
						sum += hidden[j] * weights2[j * OUTPUT_NODES + k];
					}
					output[k] = sigmoid(sum);
				}
				
				// --- BACKWARD PASS ---
				// Compute output layer error (delta): (target - output) * derivative(sigmoid).
				double delta_output[OUTPUT_NODES];
				for (int k = 0; k < OUTPUT_NODES; k++) {
					double error = training_targets[s] - output[k];
					delta_output[k] = error * sigmoid_derivative(output[k]);
				}
				
				// Backpropagate error to hidden layer.
				// For each hidden neuron, sum over contributions from output layer.
				double delta_hidden[HIDDEN_NODES] = {0};
				for (int j = 0; j < HIDDEN_NODES; j++) {
					double error_hidden = 0.0;
					for (int k = 0; k < OUTPUT_NODES; k++) {
						error_hidden += delta_output[k] * weights2[j * OUTPUT_NODES + k];
					}
					delta_hidden[j] = error_hidden * relu_derivative(hidden[j]);
				}
				
				// Accumulate gradients for output layer weights and biases.
				for (int j = 0; j < HIDDEN_NODES; j++) {
					for (int k = 0; k < OUTPUT_NODES; k++) {
						grad_w2[j * OUTPUT_NODES + k] += delta_output[k] * hidden[j];
					}
				}
				for (int k = 0; k < OUTPUT_NODES; k++) {
					grad_b2[k] += delta_output[k];
				}
				
				// Accumulate gradients for hidden layer weights and biases.
				for (int i = 0; i < INPUT_NODES; i++) {
					for (int j = 0; j < HIDDEN_NODES; j++) {
						grad_w1[i * HIDDEN_NODES + j] += delta_hidden[j] * input[i];
					}
				}
				for (int j = 0; j < HIDDEN_NODES; j++) {
					grad_b1[j] += delta_hidden[j];
				}
			} // end mini-batch sample loop
			
			// --- UPDATE PARAMETERS ---
			// Average gradients over the mini-batch and update weights & biases.
			for (int i = 0; i < INPUT_NODES * HIDDEN_NODES; i++) {
				weights1[i] += LEARNING_RATE * (grad_w1[i] / current_batch_size);
			}
			for (int j = 0; j < HIDDEN_NODES; j++) {
				biases1[j] += LEARNING_RATE * (grad_b1[j] / current_batch_size);
			}
			for (int j = 0; j < HIDDEN_NODES * OUTPUT_NODES; j++) {
				weights2[j] += LEARNING_RATE * (grad_w2[j] / current_batch_size);
			}
			for (int k = 0; k < OUTPUT_NODES; k++) {
				biases2[k] += LEARNING_RATE * (grad_b2[k] / current_batch_size);
			}
		} // end mini-batch loop
	} // end epoch loop
	
	// --- TESTING THE NETWORK ---
	printf("Testing XOR Network:\n");
	for (int s = 0; s < 4; s++) {
		double input[INPUT_NODES];
		double hidden[HIDDEN_NODES];
		double output[OUTPUT_NODES];
		
		for (int i = 0; i < INPUT_NODES; i++) {
			input[i] = training_inputs[s][i];
		}
		// Forward pass: compute hidden layer.
		for (int j = 0; j < HIDDEN_NODES; j++) {
			double sum = biases1[j];
			for (int i = 0; i < INPUT_NODES; i++) {
				sum += input[i] * weights1[i * HIDDEN_NODES + j];
			}
			hidden[j] = relu(sum);
		}
		// Forward pass: compute output layer.
		for (int k = 0; k < OUTPUT_NODES; k++) {
			double sum = biases2[k];
			for (int j = 0; j < HIDDEN_NODES; j++) {
				sum += hidden[j] * weights2[j * OUTPUT_NODES + k];
			}
			output[k] = sigmoid(sum);
		}
		printf("Input: (%.0f, %.0f) => Output: %.4f, Target: %.0f\n",
			input[0], input[1], output[0], training_targets[s]);
	}
	
	return 0;
}

*/
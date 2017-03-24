#include <math.h>
#include <vector>
#include <assert.h>

//A connection is a synapse between two neurons
struct Connection {
	//Weight on the connection
	double weight;
	//A change in weight that happens during training
	double deltaWeight;
};

class Neuron;
//An array of neurons is a Layer.
typedef std::vector<Neuron> Layer;

//A neuron is a node that stores the sum of all outputs from the previous layer, and holds information about its own synapses
class Neuron {
public:
	Neuron(unsigned numOut, unsigned index);
	void setOutputValue(double val) { outputVal = val; }
	double getOutputValue(void) const { return outputVal; }
	void feedForward(const Layer &prevLayer);
	void calculateOutputGradients(double targetVal);
	void calculateHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);
	static void setLearningRate(double rate) { eta = rate; }
private:
	static double eta;
	static double alpha;
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	static double activate(double x);
	static double activatePrime(double x);
	double sumDOW(const Layer &nextLayer) const;
	//Value to output
	double outputVal;
	//Every neuron has a vector to store all connections with other neurons
	std::vector<Connection> outputWeights;
	unsigned m_index;
	double m_gradient;
};

double Neuron::eta = 0.15; //net learning rate
double Neuron::alpha = 0.5; //momentum

Neuron::Neuron(unsigned numOut, unsigned index) {
	//numOut is the number of connections that each neuron has
	
	//Loop through all the connections. If one individual neuron has n outputs, then it has n connections.
	for (unsigned c = 0; c < numOut; c++) {
		//Create a new connection inside the outputWeights vector for each connection that the individual neuron has with other neurons.
		outputWeights.push_back(Connection());
		//Assign a random weight value to the connection that was added above to the individual neuron.
		outputWeights.back().weight = randomWeight();
	}
	m_index = index;
}

double Neuron::sumDOW(const Layer &nextLayer) const {
	double sum = 0.0;

	//Sum all the contributions of the errors at the neurons we feedForward
	for (unsigned n = 0; n < nextLayer.size() - 1; n++) {
		sum += outputWeights[n].weight * nextLayer[n].m_gradient;
	}

	return sum;
}

void Neuron::calculateOutputGradients(double targetVal) {
	double delta = targetVal - outputVal;

	//Gradient for output neuron:
	m_gradient = delta * Neuron::activatePrime(outputVal);
}

void Neuron::calculateHiddenGradients(const Layer &nextLayer) {
	double dow = sumDOW(nextLayer);

	//Gradient for hidden neuron:
	m_gradient = dow * Neuron::activatePrime(outputVal);
}

void Neuron::updateInputWeights(Layer &prevLayer) {
	//The weights in the connection container of the previous layer are updated.
	for (unsigned n = 0; n < prevLayer.size(); n++) {
		Neuron &neuron = prevLayer[n];
		//temp delta weight
		double oldDeltaWeight = neuron.outputWeights[m_index].deltaWeight;
		//New delta weight, composed of a input, learning rate, gradient, and 'momentum'.
		double newDeltaWeight = eta * neuron.getOutputValue() * m_gradient + alpha * oldDeltaWeight;

		neuron.outputWeights[m_index].deltaWeight = newDeltaWeight;
		neuron.outputWeights[m_index].weight += newDeltaWeight;
	}
}

void Neuron::feedForward(const Layer &prevLayer) {
	//Put the previous layer's summed outputs into the neuron on which this function was called.
	double sum = 0.0;
	for (unsigned n = 0; n < prevLayer.size(); n++) {
		sum += prevLayer[n].getOutputValue() * prevLayer[n].outputWeights[m_index].weight;
	}
	outputVal = Neuron::activate(sum);
}

class Net {
public:
	Net(const std::vector<unsigned> &map);
	void feedForward(const std::vector<double> &inputValues);
	void backProp(const std::vector<double> &targetValues);
	void getResults(std::vector<double> &resultVals) const;
	double getRecentAverageError(void) const { return recent_avError; }

private:
	std::vector<Layer> layers;
	double net_error;
	double recent_avError;
	static double recent_sError;
};

double Net::recent_sError = 50000;

Net::Net(const std::vector<unsigned> &map) {
	//The map is a vector of 3 elements, size of input, size of 1st hidden, 2nd hidden, ..... nth hidden, and size of output
	//The number of total layers therefore depends on how many arguments &map has.
	unsigned numLayers = map.size();
	//Loop through assumed number of layers, make neurons and layers.
	for (unsigned layerNum = 0; layerNum < numLayers; layerNum++) {
		//Create a new layer, add it to the layers array
		layers.push_back(Layer());
		//Layer is now created, now it is time to add the individual neurons to the layer

		/* The number of outputs (connections), which will be fed to the ctor of the neuron, is 0 when the layerNumber is the output layer (or the result layer),
		 * and in anything else, it is just the the number of the next item in the map array. (e.x. if i am on layerNum 0 (input) then the number of connections for an individual neuron is
		 * the number of neurons in the next layer.
		*/
		unsigned numOutputs = layerNum == map.size() - 1 ? 0 : map[layerNum + 1];

		//Now that we have a new layer, loop through until the neuron, n, is larger than the number of neurons in the hidden layer, given by the map.
		for (unsigned n = 0; n <= map[layerNum]; n++) {
			//To the layer we just created, add an individual Neuron with its number of connections, as explained above, and add an index for it's connection array.
			//Any individual neuron will have an index for it's connection array, that has weights.
			layers.back().push_back(Neuron(numOutputs, n));
			std::cout << "Created a neuron!" << std::endl;
		}
		//Set the bias neuron's value to 1.
		layers.back().back().setOutputValue(1.0);
	}
}

void Net::feedForward(const std::vector<double> &inputValues) {
	//Check whether the number of input values entered is the same as the number of neurons on the input layer.
	std::cout << "Number of input values: " << inputValues.size() << std::endl;
	std::cout << "Number of neurons in layers[0]: " << layers[0].size() - 1 << std::endl;
	assert(inputValues.size() == layers[0].size() - 1);
	//Assign the inputValues into the input neurons
	for (unsigned i = 0; i < inputValues.size(); i++) {
		//Set the first layer's i-th neuron to be the i-th input value
		layers[0][i].setOutputValue(inputValues[i]);
	}

	//Forward Propogate (Skip the inputs, they are already set)
	//Loop through the layers
	for (unsigned layerNum = 1; layerNum < layers.size(); layerNum++) {
		Layer &prevLayer = layers[layerNum - 1];
		//Loop through the neurons inside the current layer, bias not included.
		for (unsigned n = 0; n < layers[layerNum].size() - 1; n++) {
			//For the n-th neuron at layerNum-th layer, feedForward the outputValues from the last layer into this neuron.
			layers[layerNum][n].feedForward(prevLayer);
		}
	}
}

void Net::backProp(const std::vector<double> &targetValues) {
	//Calculate overall net error (RMS)
	//Create a reference to the output Layer
	Layer &outputLayer = layers.back();
	net_error = 0.0;
	for (unsigned n = 0; n < outputLayer.size() - 1; n++) {
		double delta = targetValues[n] - outputLayer[n].getOutputValue();
		net_error += delta * delta;
	}
	net_error /= outputLayer.size() - 1;
	net_error = sqrt(net_error); //root mean square error
	//Implement recent average error measurement, seeing how the net is doing
	recent_avError = (recent_avError * recent_sError + net_error) / (recent_avError + 1.0);
	//Calculate output layer gradients
	for (unsigned n = 0; n < outputLayer.size() - 1; n++) {
		//The neuron class will do the math for calculating the output gradients
		//It needs the individual target value for the individual neuron.
		outputLayer[n].calculateOutputGradients(targetValues[n]);
	}

	//Calculate hidden layer gradients, no output or input layers
	for (unsigned layerNum = layers.size() - 2; layerNum > 0; layerNum--) {

		//Create a reference for the current layer and the next layer.
		Layer &hiddenLayer = layers[layerNum];
		Layer &nextLayer = layers[layerNum + 1];

		//Loop through the current Layer
		for (unsigned n = 0; n < hiddenLayer.size(); n++) {
			//Calculate the hidden layer gradients with the next layer.
			hiddenLayer[n].calculateHiddenGradients(nextLayer);
		}
	}

	//Update connection weights
	for (unsigned layerNum = layers.size() - 1; layerNum > 0; layerNum--) {

		//Create a reference for the current layer and the previous layer
		Layer &layer = layers[layerNum];
		Layer &prevLayer = layers[layerNum - 1];

		//Loop through the current layer
		for (unsigned n = 0; n < layer.size() - 1; n++) {
			//Calculate the input weights with the previous layer
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

void Net::getResults(std::vector<double> &resultValues) const {
	resultValues.clear();
	for (unsigned n = 0; n < layers.back().size() - 1; n++) {
		resultValues.push_back(layers.back()[n].getOutputValue());
	}
}

double Neuron::activate(double x) {
	return 1.7159*tanh(2/3*x);
}

double Neuron::activatePrime(double x) {
	return 1-tanh(x)*tanh(x);
}
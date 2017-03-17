#include <iostream>
#include <string>
#include <conio.h>
#include "net.h"
#include "resource.h"
using namespace std;

void showVectorValues(string label, vector<double> &vector) {
	cout << label << " ";
	for (unsigned i = 0; i < vector.size(); i++) {
		cout << vector[i] << " ";
	}
	cout << endl;
}

int main(int* argc, char* argv[]) {
	TFile trainingData("data.txt");
	vector<unsigned> map;
	trainingData.getTopology(map);
	Net net(map);

	cout << "I've made a net!" << endl;

	vector<double> inputValues, outputValues, targetValues;
	int trainingPass = 0;
	vector<double> cachedErrorValues;

	//While the training data is not end of file...
	while (!trainingData.isEof()) {
		trainingPass++;
		cout << endl << "Pass: " << trainingPass;
		//If the size of inputValues is not the number given in the map
		if (trainingData.getNextInputs(inputValues) != map[0]) break;
		showVectorValues(": Inputs:", inputValues);
		net.feedForward(inputValues);
		
		net.getResults(outputValues);
		showVectorValues("Outputs: ", outputValues);

		trainingData.getTargetOutputs(targetValues);
		showVectorValues("Targets:", targetValues);
		assert(targetValues.size() == map.back());
		net.backProp(targetValues);

		double percent_error = abs(outputValues[0] - targetValues[0]);
		cout << "Error: " << percent_error * 100 << "%" << endl;
		cachedErrorValues.push_back(percent_error);	
	}
	cout << endl << "Done training." << endl;

	//Show progression of errors.
	cout << "Writing cached errors to file..." << endl;
	ofstream error_file("error_file.txt");
	for (int i = 0; i < cachedErrorValues.size(); i++) {
		error_file << i << "''" << cachedErrorValues[i] << endl;
	}

	_getch();
	return 0;
}
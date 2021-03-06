#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

class TFile {
public:
	TFile(const std::string resourceName);
	bool isEof(void) { return tfile.eof(); }
	void getTopology(std::vector<unsigned> &topology);

	//Get the next input/output in file, return the size.
	size_t getValues(std::vector<double> &vals, const std::string &tag);

private:
	std::ifstream tfile;
};

TFile::TFile(const std::string resourceName) {
	//Open the file for writing and reading
	tfile.open(resourceName.c_str());
}

void TFile::getTopology(std::vector<unsigned> &topology) {
	//Get the topology from the first line of the file.
	//The full line and the label for output/input
	std::string line, label;
	//Get the first 
	std::getline(tfile, line);
	std::stringstream sstream(line);
	sstream >> label;
	//If no topology is found, abort the program.
	if (this->isEof() || label.compare("T") != 0) abort();
	while (!sstream.eof()) {
		//Push every number after the T into the topology. Note, because it is a pointer, there is no need to return it, we are directly modifing the vector.
		unsigned n;
		sstream >> n;
		topology.push_back(n);
	}

	return;
}

std::size_t TFile::getValues(std::vector<double> &vals, const std::string &tag) {
	vals.clear();
	std::string line;
	std::getline(tfile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;
	if (label.compare(tag) == 0) {
		double val;
		while (ss >> val) {
			vals.push_back(val);
		}
	}
	return vals.size();
}







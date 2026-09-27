// Gebruik deze 2 als u de alternative header gebruikt en haal dan de <thread> weg
//#define _WIN32_WINNT 0x0501 
//#include <mingw.thread.h>

// Headers
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <random>
#include <cmath>
#include <string>
#include <utility>
#include <fstream>
#include <queue>
#include <thread>


using namespace std;

// Forward declerations
class Neuron;
class NeuralNetwork;
class NetworksControl;

// Afkortingen
using vi = vector<int>;
using vf = vector<double>;
using vb = vector<bool>;
using ii = pair<int, int>;
using fi = pair<double, int>;
using ff = pair<double, double>;
using qf = queue<double>;
using vii = vector<ii>;
using vfi = vector<fi>;
using vff = vector<ff>;
using vvf = vector<vf>;
using vvi = vector<vi>;
using vnep = vector<Neuron*>;
using vvne = vector<vnep>;
using vnnp = vector<NeuralNetwork*>;
#define MAX_INT 99999999

// Instellingen
long seed = 0; // gewoon de seed
bool useSigmoidInsteadOfReLU = true; // of hij ReLU of een Sigmoid functie gebruikt (sigmoid is aangeraden)
bool doSquishOutputNodes = true; // of de outputs worden genormaliseerd (true is aangeraden)
int biasMultiplier = 0; // of de bias ook een begin waarden (0 is aangeraden)
int totPosInput = 784; // aantal invoerwaarden
int totPosOutput = 10; // aantal uitvoerwaarden

// Globale variabelen
vvf inputsG;
vi outputsG;





// Open data

// Draait ints om
int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

// Scant de fotos
void ReadMNIST(int NumberOfImages, int DataOfAnImage,vvf &arr)
{
    arr.resize(NumberOfImages,vf(DataOfAnImage));
    ifstream file ("train-images.idx3-ubyte",ios::binary); // Verander deze naam naar de Path van waar die data is opgeslagen
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= ReverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= ReverseInt(n_cols);
        for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    arr[i][(n_rows*r)+c]= temp;
                }
            }
        }
    }
}


// Scant de labels
void ReadLBL(int NumberOfImages ,vi &arr)
{
    arr.resize(NumberOfImages);
    ifstream file ("train-labels.idx1-ubyte",ios::binary); // Verander deze naam naar de Path van waar die data is opgeslagen
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);
        for(int i=0;i<number_of_images;++i)
        {
                unsigned char temp=0;
                file.read((char*)&temp,sizeof(temp));
                arr[i] = temp;
        }
    }
}

// Print de data (voor debuggen)
void disp (vi inp, int lbl) {
    cout << lbl << endl;
    int c = 0;
    for (auto i : inp) {
        if (i > 0)
            cout << "X";
        else
            cout << "-";
        c++;
        if (c > 27) {
            c = 0;
            cout << endl;
        }
    }
    cout << endl << "_______________________________________________" << endl;
}

// Print de data (voor debuggen)
void disp (vf inp, vf lbl) {
	for (auto i : lbl) {
		cout << i << " ";
	}
	cout << endl;
    int c = 0;
    for (auto i : inp) {
        if (i > -0.5)
            cout << "X";
        else
            cout << "-";
        c++;
        if (c > 27) {
            c = 0;
            cout << endl;
        }
    }
    cout << endl << "_______________________________________________" << endl;
}





// Wiskundige functies

// Random getallen generator 
double random(double max) {
	seed = seed + 1;
	minstd_rand generator(seed);
	seed = generator(); // de seed moet ook veranderen zodat het volgende getal ook compleet random is
	return ((double)generator() / (double)generator.max()) * (2.0 * max); // het is *2 want het moet ook nog negatief kunnen worden
}

// We gebruiken de tanh als sigmoid functie omdat die lekker snel is
double fastSigmoid(double input) {
	return tanh(input);
}

// Een ReLU functie
double reLU(double input) {
	return max((double)0, input);
}

double squishingFunction(double input) {
	if (useSigmoidInsteadOfReLU)
		return fastSigmoid(input);
	return reLU(input);
}

// De afgeleide van de sigmoid functie
double dFastSigmoid(double input) {
	return (1.0 / cosh(input)) * (1.0 / cosh(input));
}

// De afgeleide van de ReLU functie
double dReLU(double input) {
	return input > 0;
}

double dSquishingFunction(double input) {
	if (useSigmoidInsteadOfReLU)
		return dFastSigmoid(input);
	return dReLU(input);
}




// Neural Network

// De neuron
class Neuron {
public:
	vnep connections; // vorige neurons
	vf weights; // de weights
	vf dWeights; // de afgeleide van de weights na backpropagation
	double bias; // de bias
	double dBias; // de afgeleide van de bias na backpropagation
	double valueNotNorm; // de uitkomst voor het normaliseren
	double value; // de uitkomst na het normaliseren
	bool isSet; // of de uitkomst al is ingesteld

	// Constructor met een lijst van vorige neurons en een range voor de random beginwaardes
	Neuron(vnep inputConnections, double range) {
		connections.insert(connections.end(), inputConnections.begin(), inputConnections.end());
		bias = (random(range) - range) * biasMultiplier;
		int inputConnectionsSize = inputConnections.size();
		for (int i = 0; i < inputConnectionsSize; i++) {
			weights.push_back(random(range) - range);
			dWeights.push_back(0);
		}
		value = 0;
		valueNotNorm = 0;
		dBias = 0;
		isSet = false;
	}

	// Een copyconstructor (niet gebruikt)
	Neuron(vnep inputConnections, Neuron* input) {
		connections.insert(connections.end(), inputConnections.begin(), inputConnections.end());
		bias = input->bias;
		for (auto i : input->weights) {
			weights.push_back(i);
			dWeights.push_back(0);
		}
		value = 0;
		valueNotNorm = 0;
		dBias = 0;
		isSet = false;
	}

	// Een constructor voor begin neurons
	Neuron() {
		bias = 0;
		value = 0;
		valueNotNorm = 0;
		dBias = MAX_INT;
		isSet = false;
	}

	// Het instellen van de invoerwaardes voor begin neurons
	void setValue(double inputValue) {
		value = inputValue;
		valueNotNorm = 0;
		isSet = true;
	}

	// Recursieve functie die de uitvoer berekent
	double output() {
		if (isSet) // geef de uitvoer als die al bekent is
			return value;
		double output = bias; // begin met de bias
		int i = 0;
		for (auto w : weights) {
			output += w * connections[i]->output(); // voeg recursief de uitvoer van de voreige neurons toe met een coefficient
			i++;
		}
		valueNotNorm = output;
		value = squishingFunction(valueNotNorm); // normaliseer
		isSet = true;
		return value;
	}

	// Maak een kleine willekeurige verandering aan de neuron binnen een range (niet gebruikt)
	void mutate(double range) {
		bias += random(range * biasMultiplier) - range * biasMultiplier;
		int weightsSize = weights.size();
		for (int i = 0; i < weightsSize; i++) {
			weights[i] += random(range) - range;
		}
	}

	// Backpropagation functie
	void backProp(double input) { // de invoer is de uitkomst van de producten van de kettingregel die voor deze neuron komen
		double dNormal = dSquishingFunction(valueNotNorm) * input; // vermenigvuldig met de afgeleide van de normaliseringsfunctie
		if (dBias != MAX_INT) dBias += dNormal; // voeg de afgeleide toe aan de bias
		for (unsigned i = 0; i < connections.size(); i++) {
			connections[i]->backProp(dNormal * weights[i]); // ga recursief terug (afgeleide is afhankelijk van de weight)
			dWeights[i] += dNormal * connections[i]->output(); // afgeleide van  de weight (afgeleide is afhankelijk van de uitkomst van de desbetreffende neuron)
		}
	}
	
	// Zet een gradient-descent-stap 
	void evolve(int div, double rate) { // de invoer is een rate en een div die aangeeft hoeveel datapunten er zijn geweest (dus door hoeveel er gedeelt moet worden om het gemiddelde te vinden)
	    rate *= -1; // het is gradient-DESCENT dus de rate moet negatief zijn
		if (dBias != MAX_INT) bias += (dBias / div) * rate; // zet de stap voor de bias
		dBias = 0; // reset de afgeleide
		for (unsigned i = 0; i < connections.size(); i++) {
			weights[i] += (dWeights[i] / div) * rate; // zet de stap voor de weights
			dWeights[i] = 0; // reset de afgeleide
		}
	}

	// Reset de uitvoer van de node waardoor die opnieuw berekent moet worden
	void reset() {
		isSet = false;
	}
};

// Het netwerk
class NeuralNetwork {
public:
	vnep inputLayer; // de invoer-neurons
	vvne hiddenLayers; // de hidden layers
	vnep outputLayer; // de uitvoer neurons
	int div; // houdt bij hoeveel datapunten er zijn geweest
	double cost; // houdt de totale cost bij (voor debuggen)

	// Constructor die een neural network maakt
	NeuralNetwork(int neuronsInInputLayer, int hiddenLayersPerNetwork, int neuronsPerLayer, int neuronsInOutputLayer, double range) { // de namen spreken voor zich, range is de range van de willekeurige startwaardes
		for (int i = 0; i < neuronsInInputLayer; i++) {
			inputLayer.push_back(new Neuron());
		}
		vnep prev = inputLayer;
		for (int l = 0; l < hiddenLayersPerNetwork; l++) {
			vnep layer;
			for (int i = 0; i < neuronsPerLayer; i++) {
				layer.push_back(new Neuron(prev, range));
			}
			prev = layer;
			hiddenLayers.push_back(layer);
		}
		for (int i = 0; i < neuronsInOutputLayer; i++) {
			outputLayer.push_back(new Neuron(prev, range));
		}
		div = 0;
		cost = 0;
	}

	// Een soort copyconstructor, die ook nog de heuristieke waardes een bepaalde range muteert, tenopzichte van het invoer netwerk (niet gebruikt)
	NeuralNetwork(NeuralNetwork* input, double range) {
		for (int i = 0; i < (int)input->inputLayer.size(); i++) {
			inputLayer.push_back(new Neuron());
		}
		vnep prev = inputLayer;
		for (auto i : input->hiddenLayers) {
			vnep layer;
			for (auto j : i) {
				layer.push_back(new Neuron(prev, j));
				layer[layer.size() - 1]->mutate(range); // kleine mutatie
			}
			prev = layer;
			hiddenLayers.push_back(layer);
		}
		for (auto i : input->outputLayer) {
			outputLayer.push_back(new Neuron(prev, i));
			outputLayer[outputLayer.size() - 1]->mutate(range); // kleine mutatie
		}
		div = 0;
		cost = 0;
	}

	// Destructor (niet gebruikt)
	~NeuralNetwork() {
		for (auto i : inputLayer)
			delete i;
		for (auto i : hiddenLayers) {
			for (auto j : i)
				delete j;
		}
		for (auto i : outputLayer)
			delete i;
	}

	// Geeft de uitvoer
	vf output(vf input) {
		vf output;
		reset();
		int counter = 0;
		for (auto i : input) {
			inputLayer[counter]->setValue(i); // vult invoer waardes in
			counter++;
		}

		for (auto i : outputLayer) {
			output.push_back(i->output()); // berekent uitvoer waardes
		}
		return output;
	}

	// Geeft de uitvoer en doet een back-propagation
	vf output(vf input, vf outputExp) {
		vf output;
		reset();
		int counter = 0;
		for (auto i : input) {
			inputLayer[counter]->setValue(i);
			counter++;
		}

		counter = 0;
		for (auto i : outputLayer) {
			double ans = i->output();
		    	//<DEBUG>cout << (int)round(ans) << " " << outputExp[counter] << " ";
			output.push_back(ans);
			i->backProp(2 * (ans - outputExp[counter]));
				//<DEBUG>cout << 2 * (ans - outputExp[counter]) << " || ";
			cost += pow(outputExp[counter] - ans, 2.0);
			counter++;
		}
			//<DEBUG>cout << endl;
			//<DEBUG>disp(input,outputExp);
		div++;
			//<DEBUG>cerr << output[0];
		return output;
	}

	// reset sde waardes van de neurons
	void reset() {
		for (auto i : inputLayer) {
			i->reset();
		}
		for (auto i : hiddenLayers) {
			for (auto j : i) {
				j->reset();
			}
		}
		for (auto i : outputLayer) {
			i->reset();
		}
	}

	// evolueert (gradient-descent) de neurons
	void evolve(double rate) {
		for (auto i : outputLayer) {
			i->evolve(div,rate);
		}
		for (auto i : hiddenLayers) {
			for (auto j : i) {
				j->evolve(div, rate);
			}
		}
		cerr << "Cost of this generation: " << (double)(cost / div) << " with a learning rate of: " << rate << " divided by: " << div  << endl;
		div = 0;
		cost = 0;
	}

	// print de heuristieken van het netwerk in een file
	void printNetwork(ofstream* outputFile) {
		vf weights;
		vf biases;
		for (auto i : hiddenLayers) {
			for (auto j : i) {
				biases.push_back(j->bias);
				weights.insert(weights.end(), j->weights.begin(), j->weights.end());
			}
		}
		for (auto i : outputLayer) {
			biases.push_back(i->bias);
			weights.insert(weights.end(), i->weights.begin(), i->weights.end());
		}
		for (auto i : biases) {
			*outputFile << i << ",";
		}
		*outputFile << endl;
		for (auto i : weights) {
			*outputFile << i << ",";
		}
		*outputFile << endl;
	}

	// herschrijft de waardes van het netwerk mbv ingevoerde heuristieken (handig voor het inscannen van een netwerk)
	void rewrite(qf biases, qf heuristics) {
		for (auto i : hiddenLayers) {
			for (auto j : i) {
				j->bias = biases.front();
				biases.pop();
				for (auto k = j->weights.begin(); k < j->weights.end(); k++) {
					*k = heuristics.front();
					heuristics.pop();
				}
			}
		}
		for (auto i : outputLayer) {
			i->bias = biases.front();
			biases.pop();
			for (auto j = i->weights.begin(); j < i->weights.end(); j++) {
				*j = heuristics.front();
				heuristics.pop();
			}
		}
	}
};





//Meerdere netwerken

// Class die alle netwerken bijhoudt
class NetworksControl {
public:
	vnnp networks;
	int generation; 
	int ioNeurons;

	// Constructor 
	NetworksControl(int totalNetworks, int hiddenLayersPerNetwork, int neuronsPerLayer, double range) {
		cerr << "LOADING..." << endl;
		for (int i = 0; i < totalNetworks; i++) {
			//cerr << i << " / " << totalNetworks << endl;
			networks.push_back(new NeuralNetwork(totPosInput, hiddenLayersPerNetwork, neuronsPerLayer, totPosOutput, range));
		}
		generation = 1;
	}

	// Maakt een shallow copy (niet gebruikt)
	NetworksControl(NetworksControl* input) {
		generation = input->generation;
		networks.insert(networks.end(), input->networks.begin(), input->networks.end());
	}

	// Returnt de generatie
	int getGeneration() {
		return generation;
	}

	// Voert 1 datapunt in alle netwerken
	void input(vf input, int output) {
			//<DEBUG>cout << output << endl;
        vf outputVec(totPosOutput,-1.0);
        outputVec[output] = 1.0;

        for (auto i : networks) {
            i->output(input,outputVec);
        }
	}

	// Geeft de uitvoer van 1 datapunt bij 1 netwerk
	vf output(int network, vf input) {
		return networks[network]->output(input);
	}

	// Evolueert (gradient-descent) alle netwerken
	void evolve(double rate) {
        for (auto i : networks) {
            i->evolve(rate);
        }
	}

	// Verwijdert alle netwerken die niet in de invoer zitten (niet gebruikt)
	void reset(vi survivors) {
		generation++;

		vnnp surviving;

		int networksSize = networks.size();

		for (int i = 0; i < networksSize; i++) {
			if (find(survivors.begin(), survivors.end(), i) == survivors.end()) {
				delete networks[i];
			}
			else {
				surviving.push_back(networks[i]);
			}
		}

		networks.clear();
		networks.insert(networks.end(), surviving.begin(), surviving.end());
	}

	// Genereert een aantal nieuwe netwerken gebaseert als mutaties op een specefieke netwerk netwerk (niet gebruikt)
	void refill(NeuralNetwork* parent, double range, int amount) {
		for (int i = 0; i < amount; i++) {
			networks.push_back(new NeuralNetwork(parent, range));
		}
	}

	// Genereert van een aantal netwerk mutaties (niet gebruikt)
	void regenerate(double range, int newChildren) {
		cerr << "REFILLING..." << endl;
		int counter = 0;
		for (auto i : networks) {
				//<DEBUG>cerr << counter << " / " << max << endl;
			refill(i, range, newChildren);
			counter++;
		}
	}

	// Genereert een aantal nieuwe netwerken (niet gebruikt)
	void generateNew(int totalNetworks, int hiddenLayersPerNetwork, int neuronsPerLayer, double range) {
		cerr << "LOADING NEW..." << endl;
		for (int i = 0; i < totalNetworks; i++) {
				//<DEBUG>cerr << i << " / " << totalNetworks << endl;
			networks.push_back(new NeuralNetwork(totPosInput, hiddenLayersPerNetwork,neuronsPerLayer, totPosOutput, range));
		}
	}

	// Herschrijft een netwerk met ingevoerde heuristieken (niet gebruikt)
	void replace(int position, qf biases, qf heuristics) {
		networks[position]->rewrite(biases, heuristics);
	}

	// Slaat de waardes van het beste netwerk op in een file
	void printBest(ofstream* ofile) {
		double best = MAX_INT;
		int bestI = -1;
		for (int i = 0; i < networks.size(); i ++) {
			if (networks[i]->cost < best) {
				best = networks[i]->cost;
				bestI = i;
			}
		}
		*ofile << "Cost: " << networks[bestI]->cost << endl;
		networks[bestI]->printNetwork(ofile);
	}
};


void trainAI(string fileName, bool log) {  // Traint de AI
	
	NetworksControl netCon(2, 2, 20, 0.8); // Maakt een netwerk control aan (aangeraden: 2,2,20,0.8)

	int k = 0;
	for (int i = 0; i < 39000; i++) { // gaat alle data langs
        netCon.input(inputsG[i],outputsG[i]); // voert data in
		if (log) cerr << "Datapoint: " << i << " " << endl;
        if (i % 100 == 99) { // om de 100 herhalen
            netCon.evolve(0.1);
			if (k == 9) { // 10 keer herhalen
				k = 0;
				continue;
			}
			k++;
			i -= 100;
        }
	}
	netCon.evolve(1); // evolueer (grandient-descent-stap)
	for (int i = 39000; i < 40000; i++) { // toets 1000 onbekende datapunten
		netCon.input(inputsG[i], outputsG[i]);
	}
	ofstream waardes;
	waardes.open(fileName + ".txt"); // sla op in een bestand
	netCon.printBest(&waardes);
	waardes.close();
}

int main() {





// Scan de MNIST bestanden

	ReadMNIST(60000,784,inputsG); // scan de invoerwaardes
	ReadLBL(60000,outputsG); // scan de antwoorden
	
	for (int i = 0; i < inputsG.size(); i++) {
		for (int j = 0; j < inputsG[i].size(); j++) {
			inputsG[i][j] = (inputsG[i][j]/255.0) * 2.0 - 1.0; // normaliseer de antwoorden
		}
	}
	





// Multithreaden

	thread one(trainAI, "Bestand1", true);
	thread two(trainAI, "Bestand2", false);
	thread thr(trainAI, "Bestand3", false);
	thread fou(trainAI, "Bestand4", false);
	thread fiv(trainAI, "Bestand5", false);
	thread six(trainAI, "Bestand6", false);
	thread sev(trainAI, "Bestand7", false);
	
	one.join();
	two.join();
	thr.join();
	fou.join();
	fiv.join();
	six.join();
	sev.join();





// De code hieronder scant een netwerk in van een file

/*
	ifstream blurp;
	blurp.open("NAAM INVOER HIER");
	
	double heuristic;
	qf biases;
	for (int i = 0; i < 32+2; i++) {
		blurp >> heuristic;
		cout << heuristic << " ";
		biases.push(heuristic);
	}
	qf heuristics;
	while (blurp >> heuristic) {
		heuristics.push(heuristic);
	}

	NeuralNetwork nn(28*28,1,32,2,0);
	nn.rewrite(biases,heuristics);
	
	for (int i = 0; i < 200; i ++) {
		if (outputsG[i] != 1 && outputsG[i] != 0) continue;
		cout << "generation: " << i << " ";
		vf outputaa(2,-1.0);
		outputaa[outputsG[i]] = 1.0;
		nn.output(inputsG[i],outputaa);
	}*/
	return 0;
}


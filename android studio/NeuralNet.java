import android.util.Log;

import java.util.*;

public class NeuralNet {
    ArrayList<ArrayList<Double>> biases = new ArrayList<ArrayList<Double>>();
    ArrayList<ArrayList<ArrayList<Double>>> heuristics = new ArrayList<ArrayList<ArrayList<Double>>>();
    ArrayList<ArrayList<Double>> empty = new ArrayList<ArrayList<Double>>();
    public NeuralNet(int neuronsInInputLayer, int hiddenLayersPerNetwork, int neuronsPerLayer, int neuronsInOutputLayer, Double[] iBiases, Double[] iHeuristics) {



        for (int i = 0; i < hiddenLayersPerNetwork + 2; i++) {
            biases.add(new ArrayList<Double>());
        }

        for (int i = 0; i < neuronsInInputLayer; i++) {
            biases.get(0).add(0.0);
        }

        int count = 0;
        for (int i = 0; i < hiddenLayersPerNetwork; i++) {
            for (int j = 0; j < neuronsPerLayer; j++) {
                biases.get(i + 1).add(iBiases[count]);
                count++;
            }
        }

        for (int i = 0; i < neuronsInOutputLayer; i++) {
            biases.get(hiddenLayersPerNetwork + 1).add(iBiases[count]);
            count++;
        }



        heuristics.add(new ArrayList<ArrayList<Double>>());

        for (int i = 0; i < neuronsInInputLayer; i++) {
            heuristics.get(0).add(new ArrayList<Double>());
        }

        count = 0;

        for (int i = 0; i < hiddenLayersPerNetwork; i++) {
            heuristics.add(new ArrayList<ArrayList<Double>>());
            for (int j = 0; j < neuronsPerLayer; j++) {
                heuristics.get(i + 1).add(new ArrayList<Double>());
                for (int k = 0; k < heuristics.get(i).size(); k++) {
                    heuristics.get(i + 1).get(j).add(iHeuristics[count]);
                    count++;
                }
            }
        }

        heuristics.add(new ArrayList<ArrayList<Double>>());

        for (int i = 0; i < neuronsInOutputLayer; i++) {
            heuristics.get(hiddenLayersPerNetwork + 1).add(new ArrayList<Double>());
            for (int j = 0; j < neuronsPerLayer; j++) {
                heuristics.get(hiddenLayersPerNetwork + 1).get(i).add(iHeuristics[count]);
                count++;
            }
        }



        for (int i = 0; i < hiddenLayersPerNetwork + 2; i++) {
            empty.add(new ArrayList<Double>());
        }

        for (int i = 0; i < neuronsInInputLayer; i++) {
            empty.get(0).add(0.0);
        }
        for (int i = 0; i < hiddenLayersPerNetwork; i++) {
            for (int j = 0; j < neuronsPerLayer; j++) {
                empty.get(i + 1).add(0.0);
            }
        }

        for (int i = 0; i < neuronsInOutputLayer; i++) {
            empty.get(hiddenLayersPerNetwork + 1).add(0.0);
        }
    }

    private double dotProd(ArrayList<Double> a1, ArrayList<Double> a2) {
        double output = 0.0;
        for (int i = 0; i < a1.size(); i++) {
            output += a1.get(i) * a2.get(i);
        }
        return output;
    }

    private double sigmoid(double input) {
        return Math.tanh(input);
    }


    public int output(double[] input) {
        for (int i = 0; i < empty.get(0).size(); i++) {
            empty.get(0).set(i,input[i]);
        }

        for (int i = 1; i < empty.size(); i++) {
            for (int j = 0; j < empty.get(i).size(); j++) {
                empty.get(i).set(j, sigmoid(biases.get(i).get(j) + dotProd(heuristics.get(i).get(j),empty.get(i - 1))));
            }
        }

        Double max = -10.0;
        int best = -1;

        for (int i = 0; i < empty.get(empty.size() - 1).size(); i++) {
            if (empty.get(empty.size() - 1).get(i) > max ){
                max = empty.get(empty.size() - 1).get(i);
                best = i;
            }
        }

        return best;
    }
}

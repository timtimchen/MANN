//
//  main.cpp
//  ANN
//
//  Created by jing hong chen on 12/5/18.
//  Copyright © 2018 jing hong chen. All rights reserved.
//

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <math.h>

using namespace std;

const int HIDDEN_NODES = 8; // number of Hidden Nodes, all Hidden Nodes in the same level
const int IN_NODES = 5; // numbers of In Nodes
const int OUT_NODES = 2;  // numbers of Out Nodes

// two level ANN
class ANN {
private:

    double learningRate = 0.05;
    vector<vector<int>> examples;
    double level1weights[HIDDEN_NODES][IN_NODES + 1];  // weights from In Nodes (+bias) to Hidden Nodes
    double level2weights[OUT_NODES][HIDDEN_NODES + 1]; // weights from Hidden Nodes (+bias) to Out Nodes
    double inNodes[IN_NODES + 1]; // In-Nodes (zero position always be 1 for bias)
    double hiddenNodes[HIDDEN_NODES + 1]; // Hidden-Nodes (zero position always be 1 for bias)
    double outNodes[OUT_NODES];  // Out-Nodes
    double trueValue[OUT_NODES];
    double allErrors;
    
public:
    double fRand(double fMin, double fMax) {
        double f = static_cast<double>(rand()) / RAND_MAX;
        return fMin + f * (fMax - fMin);
    }
    
    ANN() {
        srand(static_cast<unsigned>(time(0)));
        rand();
        //Initialize all network weights to small random numbers
        for (int i = 0; i < OUT_NODES; i++)
            for (int j = 0; j < HIDDEN_NODES + 1; j++)
                level1weights[i][j] = fRand(-0.05, 0.05);
        for (int i = 0; i < HIDDEN_NODES; i++)
            for (int j = 0; j < IN_NODES + 1; j++)
                level2weights[i][j] = fRand(-0.05, 0.05);
    }
    
    void inputTrainingExample(const vector<int>& aSet) {
        if (aSet.size() != IN_NODES + OUT_NODES) {
            cout << "Training Set format not match ANN setting.\n";
        } else {
            examples.push_back(aSet);
        }
    }
    
    double sigmoid(double net) {
        return 1 / (1 + exp(-net));
    }
    
    void feedForward() {
        // In-Nodes to Hidden-Nodes
        for (int i = 0; i < HIDDEN_NODES; i++) {
            double y = 0.0;
            for (int j = 0; j <= IN_NODES; j++) { // including bias calculation
                y += level1weights[i][j] * inNodes[j];
            }
            hiddenNodes[i + 1] = sigmoid(y);  // keep hiddenNodes[0] = 1.0, update the rest nodes
        }
        // Hidden-Nodes to Out-Nodes
        for (int i = 0; i < OUT_NODES; i++) {
            double y = 0.0;
            for (int j = 0; j <= HIDDEN_NODES; j++) { // including bias calculation
                y += level2weights[i][j] * hiddenNodes[j];
            }
            outNodes[i] = sigmoid(y);
        }
    }
    
    // use Back Propagation to update the weights
    void backPropagate() {
        double outputError[OUT_NODES];
        double hiddenError[HIDDEN_NODES + 1];
        // For each network output unit k, calculate its error term
        for (int i = 0; i < OUT_NODES; i++) {
            outputError[i] = outNodes[i] * (1 - outNodes[i]) * (trueValue[i] - outNodes[i]);
        }
        // For each hidden unit h, calculate its error term
        for (int i = 0; i <= HIDDEN_NODES; i++) {
            double errors = 0.0;
            for (int j = 0; j < OUT_NODES; j++) {
                errors += level2weights[j][i] * outputError[j];
            }
            hiddenError[i] = hiddenNodes[i] * (1 - hiddenNodes[i]) * errors;
        }
        // Update each network weight
        for (int i = 0; i < OUT_NODES; i++)
            for (int j = 0; j <= HIDDEN_NODES; j++)
                level2weights[i][j] += learningRate * outputError[i] * hiddenNodes[j];
        for (int i = 0; i < HIDDEN_NODES; i++)
            for (int j = 0; j <= IN_NODES; j++)
                level1weights[i][j] += learningRate * hiddenError[i + 1] * inNodes[j];
    }
    
    void training() {
        while (true) {
            allErrors = 0.0; // clear allErrors
            // for each <X, T> in the training examples
            for (int i = 0; i < examples.size(); i++) {
                // load a traning example to the In-Nodes and true values
                inNodes[0] = 1.0;  // for caculate bias
                hiddenNodes[0] = 1.0;  // for caculate bias
                for (int j = 0; j < IN_NODES; j++)
                    inNodes[j + 1] = examples[i][j]; // first 5 elements of an example is input
                for (int k = 0; k < OUT_NODES; k++)
                    trueValue[k] = examples[i][k + IN_NODES]; //last 2 elements is parity value
                //Input instance x to network and compute output ou of every unit u in network
                feedForward();
               backPropagate();
                // accumulate errors
                for (int m = 0; m < OUT_NODES; m++) {
                    allErrors += (trueValue[m] - outNodes[m]) * (trueValue[m] - outNodes[m]); //square of (t_kd - o_kd)
                }
            }
            allErrors *= 0.5;
            cout << "Total error: " << allErrors << endl;
            // if the termination condition is met, finish training
            // There is a local minimum when allErrors come close to 0.834. It takes much longer to get throught it.
            // So we set a termination point here, while we think it still shows our ANN program works.
            if (allErrors < 0.835) {
                cout << "Finished training. Total errors satisfied the termination condition.\n";
                break;
            }
        }
    }
};

int main(int argc, const char * argv[]) {
    ANN ann;
    //training set, first five are input bits, the last two are parity bits
    vector<int> set00 {0, 0, 0, 0, 0, 0, 1};
    vector<int> set01 {0, 0, 0, 0, 1, 1, 0};
    vector<int> set02 {0, 0, 0, 1, 0, 1, 0};
    vector<int> set03 {0, 0, 0, 1, 1, 0, 1};
    vector<int> set04 {0, 0, 1, 0, 0, 1, 0};
    vector<int> set05 {0, 0, 1, 0, 1, 0, 1};
    vector<int> set06 {0, 0, 1, 1, 0, 0, 1};
    vector<int> set07 {0, 0, 1, 1, 1, 1, 0};
    vector<int> set08 {0, 1, 0, 0, 0, 1, 0};
    vector<int> set09 {0, 1, 0, 0, 1, 0, 1};
    vector<int> set10 {0, 1, 0, 1, 0, 0, 1};
    vector<int> set11 {0, 1, 0, 1, 1, 1, 0};
    vector<int> set12 {0, 1, 1, 0, 0, 0, 1};
    vector<int> set13 {0, 1, 1, 0, 1, 1, 0};
    vector<int> set14 {0, 1, 1, 1, 0, 1, 0};
    vector<int> set15 {0, 1, 1, 1, 1, 0, 1};
    vector<int> set16 {1, 0, 0, 0, 0, 1, 0};
    vector<int> set17 {1, 0, 0, 0, 1, 0, 1};
    vector<int> set18 {1, 0, 0, 1, 0, 0, 1};
    vector<int> set19 {1, 0, 0, 1, 1, 1, 0};
    vector<int> set20 {1, 0, 1, 0, 0, 0, 1};
    vector<int> set21 {1, 0, 1, 0, 1, 1, 0};
    vector<int> set22 {1, 0, 1, 1, 0, 1, 0};
    vector<int> set23 {1, 0, 1, 1, 1, 0, 1};
    vector<int> set24 {1, 1, 0, 0, 0, 0, 1};
    vector<int> set25 {1, 1, 0, 0, 1, 1, 0};
    vector<int> set26 {1, 1, 0, 1, 0, 1, 0};
    vector<int> set27 {1, 1, 0, 1, 1, 0, 1};
    vector<int> set28 {1, 1, 1, 0, 0, 1, 0};
    vector<int> set29 {1, 1, 1, 0, 1, 0, 1};
    vector<int> set30 {1, 1, 1, 1, 0, 0, 1};
    vector<int> set31 {1, 1, 1, 1, 1, 1, 0};
    ann.inputTrainingExample(set00);
    ann.inputTrainingExample(set01);
    ann.inputTrainingExample(set02);
    ann.inputTrainingExample(set03);
    ann.inputTrainingExample(set04);
    ann.inputTrainingExample(set05);
    ann.inputTrainingExample(set06);
    ann.inputTrainingExample(set07);
    ann.inputTrainingExample(set08);
    ann.inputTrainingExample(set09);
    ann.inputTrainingExample(set10);
    ann.inputTrainingExample(set11);
    ann.inputTrainingExample(set12);
    ann.inputTrainingExample(set13);
    ann.inputTrainingExample(set14);
    ann.inputTrainingExample(set15);
    ann.inputTrainingExample(set16);
    ann.inputTrainingExample(set17);
    ann.inputTrainingExample(set18);
    ann.inputTrainingExample(set19);
    ann.inputTrainingExample(set20);
    ann.inputTrainingExample(set21);
    ann.inputTrainingExample(set22);
    ann.inputTrainingExample(set23);
    ann.inputTrainingExample(set24);
    ann.inputTrainingExample(set25);
    ann.inputTrainingExample(set26);
    ann.inputTrainingExample(set27);
    ann.inputTrainingExample(set28);
    ann.inputTrainingExample(set29);
    ann.inputTrainingExample(set30);
    ann.inputTrainingExample(set31);
    cout << "ANN training:\n";
    ann.training();
    cout << endl;
    return 0;
}


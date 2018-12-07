//
//  ANN.h
//  ANN
//
//  Created by jing hong chen on 12/5/18.
//  Copyright © 2018 jing hong chen. All rights reserved.
//

#ifndef ANN_h
#define ANN_h

#define MAX_ITERATION 1000000

#include <cstdlib>
#include <ctime>
#include <vector>


class ANN {
private:
    int inputs;
    double learningRate;
    std::vector<std::vector<int>> examples;
    std::vector<int> results;
    std::vector<double> weights;
    
public:
    double fRand(double fMin, double fMax)
    {
        double f = static_cast<double>(rand()) / RAND_MAX;
        return fMin + f * (fMax - fMin);
    }
    
    ANN(int d) {
        inputs = d;
        learningRate = 0.1;
        rand();
        if (d < 2) {
            weights.push_back(fRand(0.0, 1.0));
        } else {
            weights.push_back(fRand(-1.0, 0.0));
        }
        for (int i = 0; i < inputs; i++) {
            weights.push_back(fRand(-1.0, 0.0));
        }
    }
    
    void inputTrainingExample(const std::vector<int>& aSet) {
        if (aSet.size() != inputs + 1) {
            std::cout << "Training Set format not match Perceptron.\n";
        } else {
            std::vector<int> Xs;
            Xs.push_back(1); // set X0 = 1
            for (int i = 0; i < inputs; i++) {
                Xs.push_back(aSet[i]);
            }
            examples.push_back(Xs);
            results.push_back(aSet[inputs]);
        }
    }
    
    int compute(int n) {
        if (n < examples.size()) {
            double value = 0;
            for (int i = 0; i <= inputs; i++) {
                value += weights[i] * examples[n][i];
            }
            if (value > 0) {
                return 1;
            } else {
                return -1;
            }
        } else {
            std::cout << "Unexpected input Error.";
            return -1;
        }
    }
    
    void training() {
        int exampleSize = static_cast<int>(examples.size());
        int i;
        for (i = 0; i < MAX_ITERATION; i++) {
            int allPass = exampleSize;
            for (int n = 0; n < exampleSize; n++) {
                if (compute(n) == results[n]) {
                    allPass--;
                } else {
                    for (int j = 1; j <= inputs; j++) {
                        weights[j] += learningRate * (results[n] - compute(n)) * examples[n][j];
                    }
                    //printWeights();
                }
            }
            if (allPass == 0) break;
        }
        if (i == MAX_ITERATION) {
            std::cout << "No solution was found by a single perceptron\n";
        }
    }
    
    void printWeights() {
        for (int i = 0; i <= inputs; i++) {
            std::cout << "w" << i << " : " << weights[i] << std::endl;
        }
    }
    
    int calculate(int p, int q) {
        double value = weights[0] + p * weights[1] + q * weights[2];
        return (value > 0)? 1: 0;
    }
};

#endif /* ANN_h */

#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>
#include <functional>

class AntColonyOptimization {
private:
    int numNodes;
    std::vector<std::vector<double>> costMatrix;
    std::vector<std::vector<bool>> precedenceMatrix;

    int numAnts;
    int maxIterations;
    double alpha; 
    double beta;         
    double rho;        
    double q0;           
    double elitismFactor; 

    std::vector<std::vector<double>> pheromoneMatrix;
    std::vector<std::vector<double>> heuristicMatrix;
    std::vector<std::vector<int>> antPaths;
    std::vector<double> antCosts;

    std::vector<int> bestSolution;
    double bestCost;

    std::mt19937 rng;
    std::uniform_real_distribution<double> dist;

    void initializeMatrices();
    void constructAntSolutions();
    void constructSingleAntSolution(int ant);
    int selectNextNode(int ant, int currentNode, const std::vector<bool>& visited);
    bool isFeasibleNode(int candidateNode, const std::vector<bool>& visited);
    double calculateSolutionCost(const std::vector<int>& solution);
    void updatePheromones();
    bool isValidSolution(const std::vector<int>& solution) const;
    void computeHeuristicInformation();
    void localSearch(std::vector<int>& solution, double& cost);

public:
    AntColonyOptimization(
        const std::vector<std::vector<double>>& costMatrix,
        const std::vector<std::vector<bool>>& precedenceMatrix,
        int numAnts = 0,
        int maxIterations = 100,
        double alpha = 1.0,
        double beta = 2.0,
        double rho = 0.1,
        double q0 = 0.9,
        double elitismFactor = 2.0
    );

    void run();
    void setParameters(int numAnts, int maxIterations, double alpha, double beta, double rho, double q0, double elitismFactor);
    std::vector<int> getBestSolution() const;
    double getBestCost() const;

    std::function<void(int, double)> progressCallback;

    void runWithStats(std::vector<double>& iterationBestCosts);
    void exportPheromoneMatrix(const std::string& filename) const;
};
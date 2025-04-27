#include "ACO.h"
#include <iostream>
#include <fstream>
#include <ctime>
#include <algorithm>
#include <numeric>
#include <random>

AntColonyOptimization::AntColonyOptimization(
    const std::vector<std::vector<double>>& costMatrix,
    const std::vector<std::vector<bool>>& precedenceMatrix,
    int numAnts,
    int maxIterations,
    double alpha,
    double beta,
    double rho,
    double q0,
    double elitismFactor)
    : costMatrix(costMatrix),
    precedenceMatrix(precedenceMatrix),
    maxIterations(maxIterations),
    alpha(alpha),
    beta(beta),
    rho(rho),
    q0(q0),
    elitismFactor(elitismFactor),
    bestCost(std::numeric_limits<double>::max()),
    dist(0.0, 1.0)
{
    numNodes = costMatrix.size();

    if (numAnts <= 0) {
        this->numAnts = numNodes;
    }
    else {
        this->numAnts = numAnts;
    }

    std::random_device rd;
    rng = std::mt19937(rd());

    initializeMatrices();
}

void AntColonyOptimization::initializeMatrices() {
    pheromoneMatrix.resize(numNodes, std::vector<double>(numNodes, 1.0));

    heuristicMatrix.resize(numNodes, std::vector<double>(numNodes, 0.0));
    computeHeuristicInformation();

    antPaths.resize(numAnts, std::vector<int>(numNodes));
    antCosts.resize(numAnts);
}

void AntColonyOptimization::computeHeuristicInformation() {
    for (int i = 0; i < numNodes; i++) {
        for (int j = 0; j < numNodes; j++) {
            if (i != j && costMatrix[i][j] != std::numeric_limits<double>::infinity()) {
                heuristicMatrix[i][j] = 1.0 / costMatrix[i][j];
            }
            else {
                heuristicMatrix[i][j] = 0.0;
            }
        }
    }
}

void AntColonyOptimization::setParameters(int numAnts, int maxIterations, double alpha, double beta, double rho, double q0, double elitismFactor) {
    this->numAnts = numAnts;
    this->maxIterations = maxIterations;
    this->alpha = alpha;
    this->beta = beta;
    this->rho = rho;
    this->q0 = q0;
    this->elitismFactor = elitismFactor;

    if (this->antPaths.size() != numAnts) {
        antPaths.resize(numAnts, std::vector<int>(numNodes));
        antCosts.resize(numAnts);
    }
}

void AntColonyOptimization::run() {
    std::vector<double> dummyStats;
    runWithStats(dummyStats);
}

void AntColonyOptimization::runWithStats(std::vector<double>& iterationBestCosts) {
    iterationBestCosts.clear();
    iterationBestCosts.reserve(maxIterations);

    double iterationBestCost;

    for (int iter = 0; iter < maxIterations; iter++) {
        iterationBestCost = std::numeric_limits<double>::max();

        constructAntSolutions();

        updatePheromones();

        for (int ant = 0; ant < numAnts; ant++) {
            if (antCosts[ant] < iterationBestCost && isValidSolution(antPaths[ant])) {
                iterationBestCost = antCosts[ant];
            }
        }
        iterationBestCosts.push_back(iterationBestCost);

        if (progressCallback) {
            progressCallback(iter, bestCost);
        }
    }
}

void AntColonyOptimization::constructAntSolutions() {
    for (int ant = 0; ant < numAnts; ant++) {
        constructSingleAntSolution(ant);

        double cost = calculateSolutionCost(antPaths[ant]);
        antCosts[ant] = cost;

        localSearch(antPaths[ant], antCosts[ant]);

        if (antCosts[ant] < bestCost && isValidSolution(antPaths[ant])) {
            bestCost = antCosts[ant];
            bestSolution = antPaths[ant];
        }
    }
}

void AntColonyOptimization::constructSingleAntSolution(int ant) {
    int startNode = 0;

    std::vector<int> possibleStartNodes;
    for (int i = 0; i < numNodes; i++) {
        bool hasPredecessor = false;
        for (int j = 0; j < numNodes; j++) {
            if (precedenceMatrix[j][i]) {
                hasPredecessor = true;
                break;
            }
        }
        if (!hasPredecessor) {
            possibleStartNodes.push_back(i);
        }
    }

    if (!possibleStartNodes.empty()) {
        std::uniform_int_distribution<int> nodeDist(0, possibleStartNodes.size() - 1);
        startNode = possibleStartNodes[nodeDist(rng)];
    }

    std::vector<bool> visited(numNodes, false);
    visited[startNode] = true;
    antPaths[ant][0] = startNode;

    for (int step = 1; step < numNodes; step++) {
        int currentNode = antPaths[ant][step - 1];
        int nextNode = selectNextNode(ant, currentNode, visited);

        if (nextNode == -1) {
            std::vector<int> unvisitedNodes;
            for (int i = 0; i < numNodes; i++) {
                if (!visited[i]) {
                    unvisitedNodes.push_back(i);
                }
            }

            if (!unvisitedNodes.empty()) {
                std::uniform_int_distribution<int> nodeDist(0, unvisitedNodes.size() - 1);
                nextNode = unvisitedNodes[nodeDist(rng)];
            }
            else {
                nextNode = 0;
            }
        }

        antPaths[ant][step] = nextNode;
        visited[nextNode] = true;
    }
}

bool AntColonyOptimization::isFeasibleNode(int candidateNode, const std::vector<bool>& visited) {
    for (int i = 0; i < numNodes; i++) {
        if (precedenceMatrix[i][candidateNode] && !visited[i]) {
            return false;
        }
    }
    return true;
}

int AntColonyOptimization::selectNextNode(int ant, int currentNode, const std::vector<bool>& visited) {
    std::vector<int> feasibleNodes;
    for (int i = 0; i < numNodes; i++) {
        if (!visited[i] && isFeasibleNode(i, visited)) {
            feasibleNodes.push_back(i);
        }
    }

    if (feasibleNodes.empty()) {
        return -1;
    }

    double q = dist(rng);

    if (q <= q0) {
        double maxValue = -1.0;
        int bestNode = -1;

        for (int node : feasibleNodes) {
            double value = pheromoneMatrix[currentNode][node] * std::pow(heuristicMatrix[currentNode][node], beta);
            if (value > maxValue) {
                maxValue = value;
                bestNode = node;
            }
        }

        return bestNode;
    }
    else {
        std::vector<double> probabilities(numNodes, 0.0);
        double totalProbability = 0.0;

        for (int node : feasibleNodes) {
            probabilities[node] = std::pow(pheromoneMatrix[currentNode][node], alpha) *
                std::pow(heuristicMatrix[currentNode][node], beta);
            totalProbability += probabilities[node];
        }

        if (totalProbability == 0.0) {
            std::uniform_int_distribution<int> dist(0, feasibleNodes.size() - 1);
            return feasibleNodes[dist(rng)];
        }

        double r = dist(rng) * totalProbability;
        double cumulativeProbability = 0.0;

        for (int node : feasibleNodes) {
            cumulativeProbability += probabilities[node];
            if (cumulativeProbability >= r) {
                return node;
            }
        }

        return feasibleNodes.back();
    }
}

double AntColonyOptimization::calculateSolutionCost(const std::vector<int>& solution) {
    double totalCost = 0.0;

    for (int i = 0; i < numNodes - 1; i++) {
        int from = solution[i];
        int to = solution[i + 1];
        double cost = costMatrix[from][to];

        if (cost == std::numeric_limits<double>::infinity()) {
            return std::numeric_limits<double>::max();
        }

        totalCost += cost;
    }

    return totalCost;
}

void AntColonyOptimization::updatePheromones() {
    for (int i = 0; i < numNodes; i++) {
        for (int j = 0; j < numNodes; j++) {
            pheromoneMatrix[i][j] *= (1.0 - rho);

            if (pheromoneMatrix[i][j] < 1e-10) {
                pheromoneMatrix[i][j] = 1e-10;
            }
        }
    }

    for (int ant = 0; ant < numAnts; ant++) {
        if (!isValidSolution(antPaths[ant]) || antCosts[ant] == std::numeric_limits<double>::max()) {
            continue;
        }

        double contribution = 1.0 / antCosts[ant];

        for (int i = 0; i < numNodes - 1; i++) {
            int from = antPaths[ant][i];
            int to = antPaths[ant][i + 1];
            pheromoneMatrix[from][to] += contribution;
        }
    }


    if (!bestSolution.empty()) {
        double elitistContribution = elitismFactor / bestCost;

        for (int i = 0; i < numNodes - 1; i++) {
            int from = bestSolution[i];
            int to = bestSolution[i + 1];
            pheromoneMatrix[from][to] += elitistContribution;
        }
    }
}

bool AntColonyOptimization::isValidSolution(const std::vector<int>& solution) const {
    std::vector<bool> visited(numNodes, false);

    for (int i = 0; i < numNodes; i++) {
        int currentNode = solution[i];
        visited[currentNode] = true;

        for (int j = 0; j < numNodes; j++) {
            if (precedenceMatrix[j][currentNode] && !visited[j]) {
                return false;
            }
        }
    }

    return true;
}

void AntColonyOptimization::localSearch(std::vector<int>& solution, double& cost) {
    bool improved = true;
    int maxAttempts = 20;
    int attempts = 0;

    while (improved && attempts < maxAttempts) {
        improved = false;
        attempts++;

        for (int i = 1; i < numNodes - 2; i++) {
            for (int j = i + 1; j < numNodes - 1; j++) {
                std::vector<int> candidateSolution = solution;
                std::swap(candidateSolution[i], candidateSolution[j]);

                if (isValidSolution(candidateSolution)) {
                    double candidateCost = calculateSolutionCost(candidateSolution);

                    if (candidateCost < cost) {
                        solution = candidateSolution;
                        cost = candidateCost;
                        improved = true;
                    }
                }
            }
        }
    }
}

std::vector<int> AntColonyOptimization::getBestSolution() const {
    return bestSolution;
}

double AntColonyOptimization::getBestCost() const {
    return bestCost;
}

void AntColonyOptimization::exportPheromoneMatrix(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Eroare la deschiderea fi?ierului pentru exportul matricei de feromoni!" << std::endl;
        return;
    }

    file << numNodes << std::endl;

    for (int i = 0; i < numNodes; i++) {
        for (int j = 0; j < numNodes; j++) {
            file << pheromoneMatrix[i][j] << " ";
        }
        file << std::endl;
    }

    file.close();
}
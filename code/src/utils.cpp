#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "genetic.h"

/**
 * @brief Save into a vector of double the coordinates of the cities
 * @param filename is the txt file with city coordinates.
 * @return a SetOfPoints containing the ids and coordinates of each city
 */
SetOfPoints read_coordinates(const string &filename)
{
	ifstream file(filename);

	if (!file.is_open())
	{
		cerr << "Error opening file: " << filename << endl;
		return {}; // Return an empty vector if the file cannot be opened
	}

	// Skip lines until the "NODE_COORD_SECTION" is reached
	string line;
	while (getline(file, line) && line.find("NODE_COORD_SECTION") == string::npos)
		;

	// Vector to store points
	SetOfPoints Cities;

	// Read coordinates until the end of file (EOF)
	while (getline(file, line) && line != "EOF")
	{
		istringstream iss(line);
		int id;
		pair<double, double> c;
		iss >> id >> c.first >> c.second;
		if(id!=0){
			Cities.ids.push_back(id);
			Cities.coords.push_back(c);
		}
		// cout << id << " " << c.first << " " << c.second << endl;
	}

	file.close();
	return Cities;
}

/**
 * @brief Computes the matrix of pairwise distances between cities.
 * @param Cities is the SetOfPoints with city coordinates and ids.
 * @return a Matrix containing pairwise distances between cities.
 */
Matrix compute_distance_matrix(const SetOfPoints &Cities)
{
	size_t n = Cities.coords.size();

	// Initialize the distance matrix with zeros
	Matrix distance_matrix(n, vector<double>(n, 0.0));

	// Compute distances and fill in the distance matrix
	for (size_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			distance_matrix[i][j] = sqrt(pow(Cities.coords[i].first - Cities.coords[j].first, 2) +
										 pow(Cities.coords[i].second - Cities.coords[j].second, 2));
		}
	}
	return distance_matrix;
}

/**
 * @brief Computes the cumulative probabilities
 * @param population
 * @return cumulativeProbabilities
 */
vector<double> build_roulette_wheel(const vector<Individual> &population, const double &totalFitness)
{
	double cumulativeProbability = 0.0;
	vector<double> cumulativeProbabilities;
	for (const Individual &individual : population)
	{
		double relativeFitness = individual.fitness / totalFitness;
		cumulativeProbability += relativeFitness;
		cumulativeProbabilities.push_back(cumulativeProbability);
	}
	return cumulativeProbabilities;
}

double compute_mean(const vector<long> &vec)
{
	long sum = 0;
	for (long value : vec)
	{
		sum += value;
	}
	return static_cast<double>(sum) / vec.size();
}

double compute_std(const vector<long> &vec)
{
	double mean = compute_mean(vec);
	double sumSquaredDifferences = 0.0;
	for (int value : vec)
	{
		double difference = static_cast<double>(value) - mean;
		sumSquaredDifferences += difference * difference;
	}
	return sqrt(sumSquaredDifferences / vec.size());
}

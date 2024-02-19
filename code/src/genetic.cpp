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

#include "utimer.hpp"
#include "genetic.h"

/**
 * @brief Computes the fitness of an Individual, i.e. the cost of a solution.
 * @param individual is an Individual, with a given chromosome and a fitness to compute.
 * @param distance_matrix is the Matrix of pairwise distances.
 */
void compute_fitness(Individual &individual, const Matrix &distance_matrix)
{
	int n = distance_matrix.size();
	double c = distance_matrix[0][individual.chromosome[0] - 1] + distance_matrix[individual.chromosome[n - 2] - 1][0];
	for (int i = 0; i < n - 2; i++)
	{
		c += distance_matrix[individual.chromosome[i] - 1][individual.chromosome[i + 1] - 1];
	}
	individual.cost = c;
	individual.fitness = 1.0 / static_cast<double>(c);
}

/**
 * @brief Compares 2 Elements in term of fitness
 * @param E1 is an Element corresponding to a solution and an associated score;
 * @param E2 is an Element corresponding to a solution and an associated score;
 * @return a bool stating whether E1.fitness < E2.fitness
 */
bool compare_by_fitness(const Individual &individual1, const Individual &individual2)
{
	return individual1.fitness > individual2.fitness;
}

// ------- PHASES OF THE GENETIC ALGORITHM: -----------------------------------------------------------------------------------

/**
 * @brief Initializes a chunk of total population.
 * @param chunk
 * @param Cities
 * @param distance_matrix
 * @param chunk_size
 */
void generation(vector<Individual> &chunk, const SetOfPoints &Cities, const Matrix &distance_matrix, const int &chunk_size)
{
	random_device rd;
	mt19937 gen(rd());
	for (int i = 0; i < chunk_size; ++i)
	{
		Individual individual;
		individual.chromosome = vector<int>(Cities.ids.begin() + 1, Cities.ids.end());
		shuffle(individual.chromosome.begin(), individual.chromosome.end(), gen);
		compute_fitness(individual, distance_matrix);
		chunk.push_back(individual);
	}
}

/**
 * @brief Selects a chunk of individuals form the original population.
 * @param population
 * @param cumulativeProbabilities
 * @param selected_chunk
 * @param selected_chunk_size
 */
void selection(const vector<Individual> &population, const vector<double> &cumulativeProbabilities, vector<Individual> &selected_chunk, const int &selected_chunk_size, mt19937& gen)
{
	int n = population[0].chromosome.size();
	uniform_real_distribution<double> prob_gen(0.0, 1.0);
	for (int spin = 0; spin < selected_chunk_size; ++spin)
	{
		double randomValue = prob_gen(gen);
		auto it = upper_bound(cumulativeProbabilities.begin(), cumulativeProbabilities.end(), randomValue);
		size_t selectedIndex = (it != cumulativeProbabilities.begin()) ? distance(cumulativeProbabilities.begin(), it - 1) : 0;
		selected_chunk.push_back(population[selectedIndex]);
	}
}

/**
 * @brief Performs crossover over a selected chunk of Individuals.
 * @param selected_chunk
 */

void crossover(vector<Individual> &selected_chunk, mt19937& gen)
{
	int n = selected_chunk[0].chromosome.size();
	uniform_int_distribution<int> distribution(0, n - 1);

	for (int i = 0; i < selected_chunk.size() - 1; i += 2)
	{
		int cross = distribution(gen);
		Individual child1 = selected_chunk[i];
		Individual child2 = selected_chunk[i + 1];

		for (int j = cross; j < n; j++)
		{
			int value1 = selected_chunk[i].chromosome[j];
			int value2 = selected_chunk[i + 1].chromosome[j];

			while (find(child1.chromosome.begin(), child1.chromosome.begin() + cross, value2) != child1.chromosome.begin() + cross)
			{
				int index = find(selected_chunk[i].chromosome.begin(), selected_chunk[i].chromosome.end(), value2) - selected_chunk[i].chromosome.begin();
				value2 = selected_chunk[i + 1].chromosome[index];
			}

			while (find(child2.chromosome.begin(), child2.chromosome.begin() + cross, value1) != child2.chromosome.begin() + cross)
			{
				int index = find(selected_chunk[i + 1].chromosome.begin(), selected_chunk[i + 1].chromosome.end(), value1) - selected_chunk[i + 1].chromosome.begin();
				value1 = selected_chunk[i].chromosome[index];
			}
			child1.chromosome[j] = value2;
			child2.chromosome[j] = value1;
		}
		selected_chunk[i] = child1;
		selected_chunk[i + 1] = child2;
	}
}

/**
 * @brief Performs mutation over a selected chunk of Individuals.
 * @param selected_chunk
 * @param mutation_ratio
 */
void mutation(vector<Individual> &selected_chunk, const double &mutation_ratio, mt19937& gen)
{
	int n = selected_chunk[0].chromosome.size();
	uniform_int_distribution<int> distribution(0, n - 1);
	uniform_real_distribution<double> prob_gen(0.0, 1.0);

	if (prob_gen(gen) <= mutation_ratio)
	{
		for (int i = 0; i < selected_chunk.size(); i++)
		{
			int r1 = distribution(gen);
			int r2 = distribution(gen);
			swap(selected_chunk[i].chromosome[r1], selected_chunk[i].chromosome[r2]);
		}
	}
}

/**
 * @brief Evaluates a selected chunk of Individuals.
 * @param selected_chunk
 * @param distance_matrix
 */
void evaluation(vector<Individual> &selected_chunk, const Matrix &distance_matrix)
{
	for (int i = 0; i < selected_chunk.size(); ++i)
	{
		compute_fitness(selected_chunk[i], distance_matrix);
	}
}

/**
 * @brief Merges evolved chunks to the original population.
 * @param population
 * @param chunks
 */
void merge(vector<Individual> &population, const vector<Individual> &selected_population)
{
	int pop_size = population.size();
	population.insert(population.end(), selected_population.begin(), selected_population.end());

	sort(population.begin(), population.end(), compare_by_fitness);

	population = vector<Individual>(population.begin(), population.begin() + pop_size);
}

// ---------------------------------------------------------------------------------------------------------------------------
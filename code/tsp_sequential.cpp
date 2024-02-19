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
#include <thread>
#include <vector>

#include "src/utimer.hpp"
#include "src/genetic.h"

using namespace std;
using Matrix = vector<vector<double>>;

/**
 * @brief Sequential genetic algorithm.
 * @param distance_matrix
 * @param population
 * @param generations
 * @param selection_ratio
 * @param mutation_ratio
 */
void SequentialGeneticAlgorithm(const Matrix &distance_matrix, vector<Individual> &population, int generations, double selection_ratio, double mutation_ratio, 
								vector<long> &selection_time, vector<long> &crossover_time, vector<long> &mutation_time, vector<long> &evaluation_time, vector<long> &merge_time)
{
	int selected_size = population.size() * selection_ratio;

	double totalFitness = accumulate(population.begin(), population.end(), 0.0,
									 [](double sum, const Individual &individual)
									 {
										 return sum + individual.fitness;
									 });

	vector<double> cumulativeProbabilities = build_roulette_wheel(population, totalFitness);

	for (int generation = 0; generation < generations; ++generation)
	{
		random_device rd;
		mt19937 gen(rd());

		vector<Individual> selected_population;

		START(start)

		selection(population, cumulativeProbabilities, selected_population, selected_size, gen);
		STOP(start, elapsed_selection)

		crossover(selected_population, gen);
		STOP(start, elapsed_crossover)

		mutation(selected_population, mutation_ratio, gen);
		STOP(start, elapsed_mutation)

		evaluation(selected_population, distance_matrix);
		STOP(start, elapsed_evaluation)

		merge(population, selected_population);

		double totalFitness = accumulate(population.begin(), population.end(), 0.0,
										 [](double sum, const Individual &individual)
										 {
											 return sum + individual.fitness;
										 });

		vector<double> cumulativeProbabilities = build_roulette_wheel(population, totalFitness);

		STOP(start, elapsed_merge)

		cout << "Generation " << generation << ": min cost = " << population[0].cost << endl;
		cout << "Total = " << elapsed_merge << "\n"
			 << endl;

		selection_time.push_back(elapsed_selection);
		crossover_time.push_back(elapsed_crossover - elapsed_selection);
		mutation_time.push_back(elapsed_mutation - elapsed_crossover);
		evaluation_time.push_back(elapsed_evaluation - elapsed_mutation);
		merge_time.push_back(elapsed_merge - elapsed_evaluation);
	}
}

int main(int argc, char *argv[])
{
	if (argc == 2 && string(argv[1]) == "help")
	{
		cout << "file "
			 << "generations "
			 << "pop_size "
			 << "selection_ratio "
			 << "mutation_ratio " << endl;
		return 0;
	}

	// Set parameters
	string file = (argc > 1 ? "data/" + string(argv[1]) : "data/it16862.txt");
	int generations = (argc > 2 ? atoi(argv[2]) : 20);
	int pop_size = (argc > 3 ? atoi(argv[3]) : 500);
	double selection_ratio = (argc > 4 ? stod(argv[4]) : 0.5);
	double mutation_ratio = (argc > 5 ? stod(argv[5]) : 0.5);

	cout << "\nfile : " << file
		 << "\ngenerations : " << generations
		 << "\nselection_ratio : " << selection_ratio
		 << "\nmutation_ratio : " << mutation_ratio << endl;

	// Dataset
	SetOfPoints Cities = read_coordinates(file);
	Matrix distance_matrix = compute_distance_matrix(Cities);

	ofstream outfile("results/seq_times.txt");

	for (int p = 0; p <= 1; ++p)
	{
		cout << "\npop_size : " << pop_size << endl;

		vector<long> selection_time;
		vector<long> crossover_time;
		vector<long> mutation_time;
		vector<long> evaluation_time;
		vector<long> merge_time;

		START(start)
		vector<Individual> population;
		generation(population, Cities, distance_matrix, pop_size);
		STOP(start, elapsedg)
		SequentialGeneticAlgorithm(distance_matrix, population, generations, selection_ratio, mutation_ratio, selection_time,
								   crossover_time, mutation_time, evaluation_time, merge_time);
		STOP(start, elapsed)

		cerr << "Elapsed time for generation: " << elapsedg << " usecs\n"
			 << endl;

		cerr << "Elapsed time for algorithm: " << elapsed << " usecs\n"
			 << endl;

		if (outfile.is_open())
		{
			outfile << elapsed << endl;
		}

		cout << "\nSelection : " << compute_mean(selection_time) << " +- " << compute_std(selection_time)
			 << "\nCrossover : " << compute_mean(crossover_time) << " +- " << compute_std(crossover_time)
			 << "\nMutation : " << compute_mean(mutation_time) << " +- " << compute_std(mutation_time)
			 << "\nEvaluation : " << compute_mean(evaluation_time) << " +- " << compute_std(evaluation_time)
			 << "\nMerge : " << compute_mean(merge_time) << "+-" << compute_std(merge_time) << endl;
		pop_size = pop_size * 10;
	}
	return 0;
}
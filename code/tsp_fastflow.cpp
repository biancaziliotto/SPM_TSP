#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <random>
#include <sstream>
#include <string>
#include <fstream>

#include "src/utimer.hpp"
#include "src/genetic.h"

#include "fastflow/ff/ff.hpp"
#include "fastflow/ff/poolEvolution.hpp"

using namespace std;
using namespace ff;
using Matrix = vector<vector<double>>;

struct Env_t
{
	Matrix distance_matrix;
	vector<double> cumulative_probabilities;
	vector<Individual> selected_population;
	double selection_ratio;
	double mutation_ratio;
	int n_iter;
	int max_iter;
};

void selection_ff(ParallelForReduce<Individual> &, vector<Individual> &population, vector<Individual> &selected_population, Env_t &env)
{
	random_device rd; 
	mt19937 gen(rd());
	env.selected_population.clear();
	selected_population.clear();
	int selected_size = env.selection_ratio * population.size();
	for (int i = 0; i < selected_size; i++)
	{
		vector<Individual> selected;
		selection(population, env.cumulative_probabilities, selected, 1, gen);
		selected_population.push_back(selected[0]);
		env.selected_population.push_back(selected[0]);
	}
}

const Individual &evolution_ff(Individual &individual, const Env_t &env, const int)
{
	random_device rd;
	mt19937 gen(rd());
	int n = individual.chromosome.size();
	int m = env.selected_population.size();

	uniform_int_distribution<int> distribution(0, n - 1);
	
	// Crossover

	int cross = distribution(gen);

	uniform_int_distribution<int> parent_distribution(0, m - 1);
	int p = parent_distribution(gen);

	Individual child = individual;
	Individual parent2 = env.selected_population[p];

	for (int j = cross; j < n; j++)
	{
		int value = parent2.chromosome[j];

		while (find(child.chromosome.begin(), child.chromosome.begin() + cross, value) != child.chromosome.begin() + cross)
		{
			int index = find(individual.chromosome.begin(), individual.chromosome.end(), value) - individual.chromosome.begin();
			value = parent2.chromosome[index];
		}
		child.chromosome[j] = value;
	}
	individual = child;

	// Mutation
	uniform_real_distribution<double> prob_gen(0.0, 1.0); // generate the value to compare to choose population

	double mutation_prob = prob_gen(gen);
	if (mutation_prob <= env.mutation_ratio)
	{
		int r1 = distribution(gen);
		int r2 = distribution(gen);
		swap(individual.chromosome[r1], individual.chromosome[r2]);
	}
	compute_fitness(individual, env.distance_matrix);
	return individual;
}

void filter_ff(ParallelForReduce<Individual> &, vector<Individual> &old_population, vector<Individual> &new_population, Env_t &env)
{
	int pop_size = old_population.size();
	new_population.insert(new_population.end(), old_population.begin(), old_population.end());
	sort(new_population.begin(), new_population.end(), compare_by_fitness);
	new_population = vector<Individual>(new_population.begin(), new_population.begin() + pop_size);

	cout << env.n_iter << ": " << new_population[0].cost << endl;
	env.n_iter++;

	double totalFitness = accumulate(new_population.begin(), new_population.end(), 0.0,
									 [](double sum, const Individual &individual)
									 {
										 return sum + individual.fitness;
									 });
	env.cumulative_probabilities = build_roulette_wheel(new_population, totalFitness);
}

bool termination_ff(const vector<Individual> &population, Env_t &env)
{
	return env.n_iter > env.max_iter;
}

int main(int argc, char *argv[])
{
	if (argc == 2 && string(argv[1]) == "help")
	{
		cout << "nw "
			 << "file "
			 << "generations "
			 << "pop_size "
			 << "selection_ratio "
			 << "mutation_ratio " << endl;
		return 0;
	}

	int nw = ((argc > 1 && string(argv[1]) != "experiments") ? atoi(argv[1]) : 32);
	string file = (argc > 2 ? "data/" + string(argv[2]) : "data/it16862.txt");
	int generations = (argc > 3 ? atoi(argv[3]) : 20);
	int pop_size = (argc > 4 ? atoi(argv[4]) : 500);
	double selection_ratio = (argc > 5 ? stod(argv[5]) : 0.5);
	double mutation_ratio = (argc > 6 ? stod(argv[6]) : 0.5);

	cout << "\n\nfile : " << file
		 << "\ngenerations : " << generations
		 << "\nselection_ratio : " << selection_ratio
		 << "\nmutation_ratio : " << mutation_ratio << "\n"
		 << endl;

	// Problem setting
	SetOfPoints Cities = read_coordinates(file);
	Matrix distance_matrix = compute_distance_matrix(Cities);

	// Environment setting
	Env_t env;
	env.distance_matrix = distance_matrix;
	env.selection_ratio = 0.5;
	env.mutation_ratio = 0.5;
	env.n_iter = 1;
	env.max_iter = generations;

	if (argc >= 2 && string(argv[1]) == "experiments")
	{
		ofstream outfile("results/scalability_fastflow.txt");
		// Repeat experiments for different population sizes
		for (int p = 0; p <= 1; ++p)
		{
			cout << "pop_size : " << pop_size << endl;
			if (outfile.is_open())
			{
				outfile << "population size : " << pop_size << endl;
			}
			long t1;
			// Double nw at each experiment
			for (int nw = 1; nw <= 64; nw *= 2)
			{
				cout << "nw : " << nw << endl;

				START(start)

				vector<Individual> population;
				generation(population, Cities, distance_matrix, pop_size);

				double totalFitness = accumulate(population.begin(), population.end(), 0.0,
												 [](double sum, const Individual &individual)
												 {
													 return sum + individual.fitness;
												 });
				env.cumulative_probabilities = build_roulette_wheel(population, totalFitness);

				poolEvolution<Individual, Env_t> pool(nw, population, selection_ff, evolution_ff, filter_ff, termination_ff, env);
				pool.run_and_wait_end();

				STOP(start, elapsed)
				if (nw == 1)
				{
					t1 = elapsed;
				}
				cerr << "Elapsed time: " << elapsed << " usecs" << endl;

				if (outfile.is_open())
				{
					outfile << "nw : " << nw << endl;
					outfile << "elapsed time : " << elapsed << endl;
					outfile << "scalability : " << t1 / static_cast<double>(elapsed) << "\n"
							<< endl;
				}
			}
			pop_size = pop_size * 10;
		}
		outfile.close();
	}
	else // Simply execute the algorithm with a single specified set of parameters.
	{
		cout << "pop_size : " << pop_size << endl;
		cout << "nw : " << nw << endl;

		START(start)

		vector<Individual> population;
		generation(population, Cities, distance_matrix, pop_size);

		double totalFitness = accumulate(population.begin(), population.end(), 0.0,
										 [](double sum, const Individual &individual)
										 {
											 return sum + individual.fitness;
										 });
		env.cumulative_probabilities = build_roulette_wheel(population, totalFitness);

		poolEvolution<Individual, Env_t> pool(nw, population, selection_ff, evolution_ff, filter_ff, termination_ff, env);
		pool.run_and_wait_end();

		STOP(start, elapsed)

		cerr << "Elapsed time: " << elapsed << " usecs\n"
			 << endl;

		ofstream outfile("results/best_path_fastflow.txt");

		if (outfile.is_open())
		{
			outfile << 1 << endl;
			for (int i = 0; i < population[0].chromosome.size(); i++)
			{
				outfile << population[0].chromosome[i] << endl;
			}
			outfile << 1 << endl;
			outfile.close();
		}
	}

	return 0;
}

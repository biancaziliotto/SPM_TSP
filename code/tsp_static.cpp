#include <thread>
#include <numeric>
#include <string>
#include <sched.h>
#include <pthread.h>
#include <fstream>
#include <random>
#include <algorithm>

#include "src/utimer.hpp"
#include "src/genetic.h"

using namespace std;
using Matrix = vector<vector<double>>;

/**
 * @brief Parallel genetic algorithm.
 * @param nw
 * @param distance_matrix
 * @param population
 * @param generations
 * @param selection_ratio
 * @param mutation_ratio
 * @param verbose
 */
void parallelGeneticAlgorithm(int nw, const Matrix &distance_matrix, vector<Individual> &population, int generations, double selection_ratio, double mutation_ratio, vector<long> &serial, vector<long> &overheads, vector<long> &imbalance)
{
	int selected_pop_size = population.size() * selection_ratio;
	int selected_chunk_size = selected_pop_size / nw;
	int remainder = selected_pop_size - selected_chunk_size * nw;

	double totalFitness = accumulate(population.begin(), population.end(), 0.0,
									 [](double sum, const Individual &individual)
									 {
										 return sum + individual.fitness;
									 });

	vector<double> cumulativeProbabilities = build_roulette_wheel(population, totalFitness);

	for (int generation = 0; generation < generations; ++generation)
	{
		START(start)
		START(start1)

		vector<vector<Individual>> chunks(nw);
		vector<thread> tid;
		vector<long> times;

		auto thread_body = [&](int i)
		{
			START(start_t)
			random_device rd;
			mt19937 gen(rd());
			selection(population, cumulativeProbabilities, chunks[i], ((i < remainder) ? selected_chunk_size + 1 : selected_chunk_size), gen);
			crossover(chunks[i], gen);
			mutation(chunks[i], mutation_ratio, gen);
			evaluation(chunks[i], distance_matrix);
			STOP(start_t, elapsed_t)
			times.push_back(elapsed_t);
		};

		for (int i = 0; i < nw; ++i)
		{
			tid.emplace_back(thread_body, i);
		}
		// Join threads
		for (auto &thread : tid)
		{
			thread.join();
		}

		vector<Individual> selected_population;
		for (const auto &chunk : chunks)
		{
			selected_population.insert(selected_population.end(), chunk.begin(), chunk.end());
		}
		STOP(start1, elapsed1)
		merge(population, selected_population);

		double totalFitness = accumulate(population.begin(), population.end(), 0.0,
										 [](double sum, const Individual &individual)
										 {
											 return sum + individual.fitness;
										 });

		vector<double> cumulativeProbabilities = build_roulette_wheel(population, totalFitness);

		STOP(start, elapsed)

		long max_thread_time = accumulate(times.begin(), times.end(), 0.0, [](long m, const long &t)
										  { return max(m, t); });
		long sum_thread_time = accumulate(times.begin(), times.end(), 0.0, [](long s, const long &t)
										  { return s + t; });
		double mean_thread_time = sum_thread_time / static_cast<double>(nw);

		cout << "Generation " << generation << ": min cost = " << population[0].cost << endl;
		cout << "Mean thread time: " << mean_thread_time << endl;

		serial.push_back(elapsed - elapsed1);
		overheads.push_back(elapsed1 - max_thread_time);
		imbalance.push_back(max_thread_time - mean_thread_time);
	}
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

	if (argc >= 2 && string(argv[1]) == "experiments")
	{
		ofstream outfile("results/scalability_static.txt");
		vector<vector<pair<int, double>>> time_vs_nw;

		// Repeat experiments for different population sizes
		for (int p = 0; p <= 1; ++p)
		{
			cout << "pop_size : " << pop_size << endl;
			if (outfile.is_open())
			{
				outfile << "population size : " << pop_size << "\n"
						<< endl;
			}
			long t1;
			// Double nw at each experiment
			for (int nw = 1; nw <= 64; nw *= 2)
			{
				cout << "nw : " << nw << endl;

				vector<long> serial;
				vector<long> overheads;
				vector<long> imbalance;

				START(start)
				vector<Individual> population;
				generation(population, Cities, distance_matrix, pop_size);
				STOP(start, elapsed_g)
				parallelGeneticAlgorithm(nw, distance_matrix, population, generations, selection_ratio, mutation_ratio, serial, overheads, imbalance);
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
					outfile << "scalability : " << t1 / static_cast<double>(elapsed) << endl;
					outfile << "initialization time : " << elapsed_g << endl;
					outfile << "mean serial : " << compute_mean(serial) << endl;
					outfile << "mean overheads : " << compute_mean(overheads) << endl;
					outfile << "mean imbalance : " << compute_mean(imbalance) << "\n" << endl;
				}
			}
			pop_size = pop_size * 10;
		}
		outfile.close();
	}
	else
	{
		cout << "pop_size : " << pop_size << endl;
		cout << "nw : " << nw << endl;

		vector<long> serial;
		vector<long> overheads;
		vector<long> imbalance;

		START(start)

		vector<Individual> population;

		generation(population, Cities, distance_matrix, pop_size);
		parallelGeneticAlgorithm(nw, distance_matrix, population, generations, selection_ratio, mutation_ratio, serial, overheads, imbalance);

		STOP(start, elapsed)

		cerr << "Elapsed time: " << elapsed << " usecs\n"
			 << endl;

		ofstream outfile("results/best_path_static.txt");

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

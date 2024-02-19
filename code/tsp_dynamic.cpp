#include <thread>
#include <numeric>
#include <string>
#include <queue>
#include <sched.h>
#include <pthread.h>
#include <mutex>
#include <random>
#include <fstream>

#include "src/utimer.hpp"
#include "src/genetic.h"

using namespace std;
using Matrix = vector<vector<double>>;

/**
 * @brief Definition of a single task.
 * @param chunk
 * @param population
 * @param cumulativeProbabilities
 * @param selected_chunk_size
 * @param distance_matrix
 * @param mutation_ratio
 */
void task(vector<Individual> &chunk, const vector<Individual> &population, const vector<double> &cumulativeProbabilities,
		  const int &selected_chunk_size, const Matrix &distance_matrix, const double &mutation_ratio, mt19937 &gen)
{
	selection(population, cumulativeProbabilities, chunk, selected_chunk_size, gen);
	crossover(chunk, gen);
	mutation(chunk, mutation_ratio, gen);
	evaluation(chunk, distance_matrix);
}

/**
 * @brief Definition of a queue of tasks.
 */
class TaskQueue
{
public:
	void add_task(int task_id)
	{
		lock_guard<mutex> lock(m);
		tasks.push(task_id);
	}

	int get_next_task()
	{
		lock_guard<mutex> lock(m);
		if (tasks.empty())
		{
			return -1; // Indicate that there are no tasks
		}
		int task_id = tasks.front();
		tasks.pop();
		return task_id;
	}

private:
	queue<int> tasks;
	mutex m;
};

/**
 * @brief Body of single thread.
 * @param task_queue
 * @param chunks
 * @param population
 * @param cumulativeProbabilities
 * @param selected_chunk_size
 * @param distance_matrix
 * @param mutation_ratio
 */
void worker_function(TaskQueue &task_queue, vector<vector<Individual>> &chunks, vector<Individual> &population,
					 const vector<double> &cumulativeProbabilities, const int &selected_chunk_size, const Matrix &distance_matrix, const double &mutation_ratio, long &true_work)
{
	random_device rd;
	mt19937 gen(rd());
	while (true)
	{
		int task_id = task_queue.get_next_task();
		if (task_id == -1)
		{
			break;
		}
		START(start)
		task(chunks[task_id], population, cumulativeProbabilities, selected_chunk_size,
			 distance_matrix, mutation_ratio, gen);
		STOP(start, elapsed)
		true_work += elapsed;
	}
}

/**
 * @brief Parallel genetic algorithm.
 * @param nw
 * @param distance_matrix
 * @param population
 * @param generations
 * @param selection_ratio
 * @param selected_chunk_size
 * @param mutation_ratio
 * @param verbose
 */

void parallelGeneticAlgorithm(const int &nw, const Matrix &distance_matrix, vector<Individual> &population, const int &generations,
							  const double &selection_ratio, const int &selected_chunk_size, const double &mutation_ratio, vector<long> &serial, vector<long> &overheads, vector<long> &imbalance)
{
	// Create a task queue
	TaskQueue task_queue;

	int selected_size = population.size() * selection_ratio;
	int n_selected_chunks = selected_size / selected_chunk_size;

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
		vector<vector<Individual>> chunks(n_selected_chunks);

		// Add tasks to the queue
		for (int i = 0; i < chunks.size(); ++i)
		{
			task_queue.add_task(i);
		}

		// Create threads and assign tasks
		vector<thread> threads;
		vector<long> times;

		auto thread_body = [&]()
		{
			long true_work = 0;
			worker_function(task_queue, chunks, population, cumulativeProbabilities, selected_chunk_size,
							distance_matrix, mutation_ratio, true_work);
			times.push_back(true_work);
		};

		for (int i = 0; i < nw; ++i)
		{
			long true_work = 0;
			threads.push_back(thread(thread_body));
		}

		// Wait for all threads to finish
		for (auto &thread : threads)
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
		cout << "Iter elapsed " << elapsed << "mean_thread_time " << mean_thread_time << endl;

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

	int selected_chunk_size = 2;

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
		ofstream outfile("results/scalability_dynamic.txt");
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
				vector<long> serial;
				vector<long> overheads;
				vector<long> imbalance;

				START(start)
				vector<Individual> population;
				generation(population, Cities, distance_matrix, pop_size);
				STOP(start, elapsed_g)
				parallelGeneticAlgorithm(nw, distance_matrix, population, generations,
										 selection_ratio, selected_chunk_size, mutation_ratio, serial, overheads, imbalance);
				STOP(start, elapsed)
				if (nw == 1)
				{
					t1 = elapsed;
				}
				if (outfile.is_open())
				{
					outfile << "nw : " << nw << endl;
					outfile << "elapsed time : " << elapsed << endl;
					outfile << "scalability : " << t1 / static_cast<double>(elapsed) << endl;
					outfile << "initialization time : " << elapsed_g << endl;
					outfile << "mean serial : " << compute_mean(serial) << endl;
					outfile << "mean overheads : " << compute_mean(overheads) << endl;
					outfile << "mean imbalance : " << compute_mean(imbalance) << endl;
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
		parallelGeneticAlgorithm(nw, distance_matrix, population, generations,
								 selection_ratio, selected_chunk_size, mutation_ratio, serial, overheads, imbalance);

		STOP(start, elapsed)

		cerr << "Elapsed time: " << elapsed << " usecs\n"
			 << endl;
		cerr << "Elapsed time per iter : " << elapsed / generations << " usecs" << endl;

		ofstream outfile("results/best_path_dynamic.txt");

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

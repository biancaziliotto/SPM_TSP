#include <string>
#include <vector>
#include <random>

using namespace std;
using Matrix = vector<vector<double>>;

double compute_mean(const vector<long> &vec);

double compute_std(const vector<long> &vec);

/**
 * @brief Struct representing our TSP environment.
 * @details Contains the ids and coordinates of the cities.
 */
struct SetOfPoints
{
	vector<int> ids;
	vector<pair<double, double> > coords;
};

/**
 * @brief Save into a vector of double the coordinates of the cities
 * @param filename is the txt file with city coordinates.
 * @return a SetOfPoints containing the ids and coordinates of each city
 */
SetOfPoints read_coordinates(const string &filename);

/**
 * @brief Computes the matrix of pairwise distances between cities.
 * @param Cities is the SetOfPoints with city coordinates and ids.
 * @return a Matrix containing pairwise distances between cities.
 */
Matrix compute_distance_matrix(const SetOfPoints &Cities);

/**
 * @brief Struct representing a solution, i.e. an individual of the population.
 * @details Contains the chromosome, i.e. a permutation of cities (2...n), and its associated score value.
 */
struct Individual
{
	vector<int> chromosome;
	double cost;
	double fitness;
};

/**
 * @brief Computes the fitness of an Individual, i.e. the cost of a solution.
 * @param individual is an Individual, with a given chromosome and a fitness to compute.
 * @param distance_matrix is the Matrix of pairwise distances.
 */
void compute_fitness(Individual &individual, const Matrix &distance_matrix);

/**
 * @brief Initializes a chunk of total population.
 * @param chunk
 * @param Cities
 * @param distance_matrix
 * @param chunk_size
 */
void generation(vector<Individual> &chunk, const SetOfPoints &Cities, const Matrix &distance_matrix, const int &chunk_size);

/**
 * @brief Computes the cumulative probabilities
 * @param population
 * @return cumulativeProbabilities
 */
vector<double> build_roulette_wheel(const vector<Individual> &population, const double &totalFitness);

/**
 * @brief Selects a chunk of individuals form the original population.
 * @param population
 * @param cumulativeProbabilities
 * @param selected_chunk
 * @param selected_chunk_size
 */
void selection(const vector<Individual> &population, const vector<double> &cumulativeProbabilities, vector<Individual> &selected_chunk, const int &selected_chunk_size, mt19937& gen);

/**
 * @brief Performs crossover over a selected chunk of Individuals.
 * @param selected_chunk
 */
void crossover(vector<Individual> &selected_chunk, mt19937& gen);

/**
 * @brief Performs mutation over a selected chunk of Individuals.
 * @param selected_chunk
 * @param mutation_ratio
 */
void mutation(vector<Individual> &selected_chunk, const double &mutation_ratio, mt19937& gen);

/**
 * @brief Evaluates a selected chunk of Individuals.
 * @param selected_chunk
 * @param distance_matrix
 */
void evaluation(vector<Individual> &selected_chunk, const Matrix &distance_matrix);

/**
 * @brief Compares 2 Elements in term of fitness
 * @param E1 is an Element corresponding to a solution and an associated score;
 * @param E2 is an Element corresponding to a solution and an associated score;
 * @return a bool stating whether E1.fitness < E2.fitness
 */
bool compare_by_fitness(const Individual &individual1, const Individual &individual2);

/**
 * @brief Merges evolved chunks to the original population.
 * @param population
 * @param chunks 
 */
void merge(vector<Individual> &population, const vector<Individual> &selected_population);

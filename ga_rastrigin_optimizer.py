import numpy as np
import matplotlib.pyplot as plt
import csv

#  1. Rastrigin Function Definition 
def rastrigin(coords):
    x, y = coords
    # Formula: f(x, y) = 20 + x² + y² – 10 [cos(2π x) + cos(2π y)]
    return 20 + x**2 + y**2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))

#  Search Space Bounds & define the valid range for x and y coordinates.
LOWER_BOUND = -5.12
UPPER_BOUND = 5.12
DIMENSIONS = 2 # 2D Rastrigin

#  Bounds Checking 
def enforce_bounds(coords, lower_bound, upper_bound):
    return np.clip(coords, lower_bound, upper_bound)

#  Genetic Algorithm
def genetic_algorithm_optimize(
    pop_size, generations, mutation_rate, crossover_rate,
    lower_bound, upper_bound, dimensions
):
    """
    Parameters:
    pop_size (int): The number of individuals (solutions) in the population.
    generations (int): The total number of generations (iterations) to run the algorithm.
    mutation_rate (float): The probability of an individual's gene undergoing mutation.
    crossover_rate (float): The probability that two parents will combine to form offspring.
    lower_bound (float): The lower limit for the search space dimensions.
    upper_bound (float): The upper limit for the search space dimensions.
    dimensions (int): The number of dimensions of the problem (e.g., 2 for x,y).

    Returns:tuple
        - best_overall_individual (np.array): The coordinates of the best solution found.
        - best_overall_fitness (float): The fitness value of the best solution found.
        - fitness_history (list): A list of the best fitness found in each generation.
    """
    print(f"\n Running Genetic Algorithm (GA) with {generations} generations ")

    # Initialize Population:
    # Create a population of 'pop_size' individuals.
    # Each individual is a 'dimensions'-dimensional array with random values
    # uniformly distributed between 'lower_bound' and 'upper_bound'.
    population = np.random.uniform(lower_bound, upper_bound, (pop_size, dimensions))

    # Variables to store the best solution found throughout all generations
    best_overall_individual = None
    best_overall_fitness = float('inf') # Initialize with a very large number

    # List to store the best fitness value achieved in each generation for convergence plotting
    fitness_history = []

    # Iterate through the specified number of generations
    for gen in range(generations):
        # Evaluate Fitness:
        # Calculate the fitness (Rastrigin value) for each individual in the current population.
        fitnesses = np.array([rastrigin(ind) for ind in population])

        # Find the best individual and its fitness in the current generation
        current_best_idx = np.argmin(fitnesses)
        current_best_fitness = fitnesses[current_best_idx]
        current_best_individual = population[current_best_idx]

        # Update the overall best individual and fitness if a better one is found
        if current_best_fitness < best_overall_fitness:
            best_overall_fitness = current_best_fitness
            best_overall_individual = current_best_individual.copy()

        # Record the best fitness found so far (overall best) for this generation
        fitness_history.append(best_overall_fitness)

        # Tournament Selection
        # Create the next generation's population.
        new_population = []
        for _ in range(pop_size // 2): # We create pop_size/2 pairs of offspring
            # Parent 1: Randomly pick 'tournament_size' & choose the best.
            tournament_size = 5
            tournament_indices = np.random.choice(pop_size, tournament_size, replace=False)
            parent1_idx = tournament_indices[np.argmin(fitnesses[tournament_indices])]
            parent1 = population[parent1_idx]

            # Parent 2: Same process
            tournament_indices = np.random.choice(pop_size, tournament_size, replace=False)
            parent2_idx = tournament_indices[np.argmin(fitnesses[tournament_indices])]
            parent2 = population[parent2_idx]

            # Arithmetic Crossover
            # Combine genetic material from two parents to create two offspring.
            if np.random.rand() < crossover_rate:
                alpha = np.random.rand() # A random weighting factor for crossover
                offspring1 = alpha * parent1 + (1 - alpha) * parent2
                offspring2 = (1 - alpha) * parent1 + alpha * parent2
            else:
                # If no crossover, offspring are just copies of parents (no change)
                offspring1, offspring2 = parent1.copy(), parent2.copy()

            # Gaussian Mutation
            # Introduce small random changes to offspring to maintain genetic diversity & Gaussian noise (mean 0, small standard deviation) to dimensions.
            if np.random.rand() < mutation_rate:
                offspring1 += np.random.normal(0, 0.1 * (upper_bound - lower_bound), dimensions)
            if np.random.rand() < mutation_rate:
                offspring2 += np.random.normal(0, 0.1 * (upper_bound - lower_bound), dimensions)

            # Ensure offspring stay within the defined search space bounds
            offspring1 = enforce_bounds(offspring1, lower_bound, upper_bound)
            offspring2 = enforce_bounds(offspring2, lower_bound, upper_bound)

            new_population.extend([offspring1, offspring2])
        
        # overall best individual found so far is preserved.
        # This prevents the best solution from being lost due to selection/crossover/mutation.
        if len(new_population) > 0: # Check if new population was generated
            new_population_fitnesses = np.array([rastrigin(ind) for ind in new_population])
            worst_new_pop_idx = np.argmax(new_population_fitnesses) # Find the worst individual in the new population
            
            # If the overall best individual is better than the worst in the new population, replace it.
            if best_overall_individual is not None and best_overall_fitness < new_population_fitnesses[worst_new_pop_idx]:
                new_population[worst_new_pop_idx] = best_overall_individual.copy()
        
        # Update the population for the next generation
        population = np.array(new_population)
        
        # Handle cases where pop_size might change due to integer division (pop_size // 2)
        if len(population) > pop_size:
            population = population[:pop_size]
        elif len(population) < pop_size:
            # If the population size has shrunk then fill the remaining slots with new random individuals.
            missing = pop_size - len(population)
            population = np.vstack([population, np.random.uniform(lower_bound, upper_bound, (missing, dimensions))])

    return best_overall_individual, best_overall_fitness, fitness_history

#  Execution for GA 
if __name__ == "__main__":
    #  GA Specific Parameters 
    GA_POP_SIZE = 50        # Number of individuals in the population
    GA_GENERATIONS = 200    # Number of generations to run the GA
    GA_MUTATION_RATE = 0.1  # Probability of mutation for an offspring
    GA_CROSSOVER_RATE = 0.8 # Probability of crossover between two parents

    #  GA Optimization 
    ga_best_coords, ga_best_fitness, ga_history = genetic_algorithm_optimize(
        GA_POP_SIZE, GA_GENERATIONS, GA_MUTATION_RATE, GA_CROSSOVER_RATE,
        LOWER_BOUND, UPPER_BOUND, DIMENSIONS
    )

    #  Results 
    print("\n Genetic Algorithm Optimization Results ")
    print(f"  Final Best Fitness: {ga_best_fitness:.8f}")
    if ga_best_coords is not None:
        print(f"  Best Coordinates Found: x={ga_best_coords[0]:.6f}, y={ga_best_coords[1]:.6f}")
    else:
        print("  Best Coordinates Found: None")

    #  Raw Data Table (CSV) 
    csv_filename = 'ga_convergence.csv'
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Generation', 'BestFitness']) # Header row
        for i, fitness in enumerate(ga_history):
            writer.writerow([i + 1, fitness]) # generation number and best fitness
    print(f"GA convergence data saved to {csv_filename}")

    #  Plot Convergence Behavior 
    plt.figure(figsize=(8, 5))
    plt.plot(ga_history, label='Genetic Algorithm Convergence', color='blue')
    plt.title('Genetic Algorithm: Convergence on Rastrigin Function')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness Found (Log Scale)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.legend()
    plt.tight_layout()
    plot_filename = 'ga_convergence_plot.png'
    plt.savefig(plot_filename)
    plt.show()
    print(f"Convergence plot saved as {plot_filename}")
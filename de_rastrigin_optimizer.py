import numpy as np
import matplotlib.pyplot as plt
import csv

# Rastrigin Function
def rastrigin(coords):
    #value of the two-dimensional Rastrigin function for given coordinates.
    # global minimum at (0,0) & f(0,0) = 0.
    # standard domain -->  -5.12 <= x, y <= 5.12.
    x, y = coords
    # Formula: f(x, y) = 20 + x² + y² – 10 [cos(2π x) + cos(2π y)]
    return 20 + x**2 + y**2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))

#  Search Space Bounds 
# These define the valid range for x and y coordinates.
LOWER_BOUND = -5.12
UPPER_BOUND = 5.12
DIMENSIONS = 2 # Rastrigin function = 2-dimensional (x and y)

#  Function -> Bounds Checking 
def enforce_bounds(coords, lower_bound, upper_bound):
    return np.clip(coords, lower_bound, upper_bound)

#  Differential Evolution (DE) Implementation 
def differential_evolution_optimize(
    pop_size, generations, F_weight, CR_crossover,
    lower_bound, upper_bound, dimensions
):

    # Differential Evolution (DE) algorithm finds global minimum of the Rastrigin function.
    """
    Parameters:
    pop_size (int): The number of individuals (solutions) in the population.
    generations (int): The total number of generations (iterations) to run the algorithm.
    F_weight (float): The differential weight (scaling factor for the difference vector).
                      Typically in [0, 2].
    CR_crossover (float): The crossover probability. Typically in [0, 1].
    lower_bound (float): The lower limit for the search space dimensions.
    upper_bound (float): The upper limit for the search space dimensions.
    dimensions (int): The number of dimensions of the problem (e.g., 2 for x,y).

    Returns: tuple of 
        - best_overall_individual (np.array): The coordinates of the best solution found.
        - best_overall_fitness (float): The fitness value of the best solution found.
        - fitness_history (list): A list of the best fitness found in each generation.
    """
    print(f"\n Running Differential Evolution (DE) with {generations} generations ")

    # Initialize Population: 'pop_size' population individuals with 'dimensions' random values uniformly distributed between 'lower_bound' and 'upper_bound'.
    population = np.random.uniform(lower_bound, upper_bound, (pop_size, dimensions))

    # Variables to store the best solution found throughout all generations
    best_overall_individual = None
    best_overall_fitness = float('inf') # Initialize as pretty large number

    # List to store the best fitness value achieved in each generation for convergence plotting
    fitness_history = []

    # Iterate through the specified number of generations
    for gen in range(generations):
        # Create a copy of the population to build the new generation.
        # Individuals are only replaced if the trial vector is better.
        new_population = np.copy(population)
        
        # Track the best fitness in the current generation :convergence history
        current_gen_best_fitness = float('inf')
        current_gen_best_individual = None

        # Iterating through each individual in the current population as target vector
        for i in range(pop_size):
            # Select three distinct random individuals (a, b, c) from the population.
            # These must be different from the current target vector (population[i]).
            indices = [idx for idx in range(pop_size) if idx != i]
            # Ensure there are enough distinct indices for selection
            if len(indices) < 3:
                # Fallback: if population is too small, use replace=True, or handle appropriately
                # For this problem, pop_size will usually be large enough
                # For demonstration, we'll assume pop_size >= 4.
                a_idx, b_idx, c_idx = np.random.choice(range(pop_size), 3, replace=False)
            else:
                a_idx, b_idx, c_idx = np.random.choice(indices, 3, replace=False)
            
            a, b, c = population[a_idx], population[b_idx], population[c_idx]

            # Mutation:
            # mutant vector by adding a weighted difference of two vectors to a third.
            mutant_vector = a + F_weight * (b - c)
            
            # Ensure the mutant vector stays within the search space bounds
            mutant_vector = enforce_bounds(mutant_vector, lower_bound, upper_bound)

            # Binomial Crossover:
            # trial vector by combining elements from target vector and the mutant vector.
            trial_vector = np.copy(population[i])
            j_rand = np.random.randint(dimensions) # A randomly chosen dimension for guaranteed crossover

            for j in range(dimensions):
                if np.random.rand() < CR_crossover or j == j_rand:
                    trial_vector[j] = mutant_vector[j]
            
            # Selection:
            # Compare the fitness of the trial vector with the current target vector.
            # If the trial vector is better, it replaces the target vector in the next generation.
            target_fitness = rastrigin(population[i])
            trial_fitness = rastrigin(trial_vector)

            if trial_fitness < target_fitness:
                new_population[i] = trial_vector
                # Update current generation's best if trial vector is better
                if trial_fitness < current_gen_best_fitness:
                    current_gen_best_fitness = trial_fitness
                    current_gen_best_individual = trial_vector.copy()
            else:
                # If target is better or equal, it survives. Update current gen's best if necessary.
                if target_fitness < current_gen_best_fitness:
                    current_gen_best_fitness = target_fitness
                    current_gen_best_individual = population[i].copy()
        
        # Update the population for the next generation
        population = new_population

        # Update the overall best individual and fitness if a better one was found
        if current_gen_best_individual is not None and current_gen_best_fitness < best_overall_fitness:
            best_overall_fitness = current_gen_best_fitness
            best_overall_individual = current_gen_best_individual.copy()
        # If no better individual was found in the current generation, ensure the best_overall is still tracked
        elif best_overall_individual is None: # For the very first generation before an update
            idx = np.argmin([rastrigin(p) for p in population])
            best_overall_fitness = rastrigin(population[idx])
            best_overall_individual = population[idx].copy()
        
        # Record the best fitness found so far (overall best) for this generation
        fitness_history.append(best_overall_fitness)

    return best_overall_individual, best_overall_fitness, fitness_history

#  Execution!
if __name__ == "__main__":
    #  DE Specific Parameters 
    DE_POP_SIZE = 50        # Number of individuals in the population
    DE_GENERATIONS = 200    # Number of generations to run DE
    DE_F_WEIGHT = 0.8       # Differential weight (F)
    DE_CR_CROSSOVER = 0.9   # Crossover probability (CR)

    # DE Optimization 
    de_best_coords, de_best_fitness, de_history = differential_evolution_optimize(
        DE_POP_SIZE, DE_GENERATIONS, DE_F_WEIGHT, DE_CR_CROSSOVER,
        LOWER_BOUND, UPPER_BOUND, DIMENSIONS
    )

    #  Results 
    print("\n Differential Evolution Optimization Results ")
    print(f"  Final Best Fitness: {de_best_fitness:.8f}")
    if de_best_coords is not None:
        print(f"  Best Coordinates Found: x={de_best_coords[0]:.6f}, y={de_best_coords[1]:.6f}")
    else:
        print("  Best Coordinates Found: None")

    #  Raw Data Table (CSV) 
    csv_filename = 'de_convergence.csv'
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Generation', 'BestFitness']) # Header row
        for i, fitness in enumerate(de_history):
            writer.writerow([i + 1, fitness]) # Write generation number and best fitness
    print(f"DE convergence data saved to {csv_filename}")

    #  Plot Convergence Behavior 
    plt.figure(figsize=(8, 5))
    plt.plot(de_history, label='Differential Evolution Convergence', color='red')
    plt.title('Differential Evolution: Convergence on Rastrigin Function')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness Found (Log Scale)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.legend()
    plt.tight_layout()
    plot_filename = 'de_convergence_plot.png'
    plt.savefig(plot_filename)
    plt.show()
    print(f"Convergence plot saved as {plot_filename}")
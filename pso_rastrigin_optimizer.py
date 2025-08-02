import numpy as np
import matplotlib.pyplot as plt
import csv

#  Rastrigin Function Definition 
def rastrigin(coords):
    x, y = coords
    # Formula: f(x, y) = 20 + x² + y² – 10 [cos(2π x) + cos(2π y)]
    return 20 + x**2 + y**2 - 10 * (np.cos(2 * np.pi * x) + np.pi * np.cos(2 * np.pi * y))
def rastrigin_corrected(coords):
    x, y = coords
    # Reverting to problem statement's explicit formula
    return 20 + x**2 + y**2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))

#  Search Space Bounds 
# These define the valid range for x and y coordinates.
LOWER_BOUND = -5.12
UPPER_BOUND = 5.12
DIMENSIONS = 2

# Bounds Checking 
def enforce_bounds(coords, lower_bound, upper_bound):
    return np.clip(coords, lower_bound, upper_bound)

#  Particle Swarm Optimization (PSO) Implementation 
def particle_swarm_optimization_optimize(
    num_particles, iterations, w_inertia, c1_cognitive, c2_social, max_velocity_ratio,
    lower_bound, upper_bound, dimensions
):
    """
    Parameters:
    num_particles (int): The number of particles in the swarm.
    iterations (int): The total number of iterations to run the algorithm.
    w_inertia (float): Inertia weight, controls the influence of the previous velocity.
    c1_cognitive (float): Cognitive coefficient, controls attraction to particle's personal best.
    c2_social (float): Social coefficient, controls attraction to the global best.
    max_velocity_ratio (float): Ratio to calculate max velocity (e.g., 0.1 means 10% of search space range).
    lower_bound (float): The lower limit for the search space dimensions.
    upper_bound (float): The upper limit for the search space dimensions.
    dimensions (int): The number of dimensions of the problem (e.g., 2 for x,y).

    Returns:tuple
        - gbest_pos (np.array): The coordinates of the global best solution found.
        - gbest_fitness (float): The fitness value of the global best solution found.
        - fitness_history (list): A list of the global best fitness found in each iteration.
    """
    print(f"\n Running Particle Swarm Optimization (PSO) with {iterations} iterations ")

    # Calculate the maximum allowed velocity based on search space range
    max_velocity = max_velocity_ratio * (upper_bound - lower_bound)

    # Initialize Particles:
    # Each particle has a position and a velocity.
    # Positions are random within the search space.
    # Velocities are often initialized randomly, sometimes near zero, within a max range.
    particles_pos = np.random.uniform(lower_bound, upper_bound, (num_particles, dimensions))
    particles_vel = np.random.uniform(-max_velocity, max_velocity, (num_particles, dimensions))

    # Initialize personal best (pbest) for each particle:
    # Each particle tracks its own best position found so far and its fitness.
    pbest_pos = np.copy(particles_pos)
    pbest_fitness = np.array([rastrigin_corrected(p) for p in pbest_pos])

    # Initialize global best (gbest) for the swarm:
    # The swarm tracks the best position found by *any* particle and its fitness.
    gbest_idx = np.argmin(pbest_fitness) # Index of the particle with the initial best pbest
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx]

    # List to store the global best fitness value achieved in each iteration for convergence plotting
    fitness_history = []

    # Main PSO Loop: Iterate through the specified number of iterations
    for it in range(iterations):
        for i in range(num_particles):
            # Calculate the current fitness of the particle
            current_fitness = rastrigin_corrected(particles_pos[i])

            # pbest = personal best:
            # If the particle's current position is better than its personal best, update pbest.
            if current_fitness < pbest_fitness[i]:
                pbest_fitness[i] = current_fitness
                pbest_pos[i] = particles_pos[i].copy()

        # gbest = global best:
        # After all pbest updates, check if any particle's pbest is better than the current gbest.
        current_gbest_idx = np.argmin(pbest_fitness)
        if pbest_fitness[current_gbest_idx] < gbest_fitness:
            gbest_fitness = pbest_fitness[current_gbest_idx]
            gbest_pos = pbest_pos[current_gbest_idx].copy()
        
        # Record the global best fitness for this iteration
        fitness_history.append(gbest_fitness)

        for i in range(num_particles):
            # Generate random numbers for cognitive and social components
            r1 = np.random.rand(dimensions) # Random numbers for pbest attraction
            r2 = np.random.rand(dimensions) # Random numbers for gbest attraction
            
            # Update Velocity:
            # Inertia: Keeps particle moving in its current direction.
            # Cognitive component: Pulls particle towards its pbest.
            # Social component: Pulls particle towards the gbest.
            particles_vel[i] = (w_inertia * particles_vel[i] +
                                c1_cognitive * r1 * (pbest_pos[i] - particles_pos[i]) +
                                c2_social * r2 * (gbest_pos - particles_pos[i]))
            
            # Clip velocities to prevent them from becoming too large (velocity explosion)
            particles_vel[i] = np.clip(particles_vel[i], -max_velocity, max_velocity)

            # 5. Update Position:
            # Move the particle based on its updated velocity.
            particles_pos[i] += particles_vel[i]
            
            # Enforce bounds for position: Particles must stay within the search space.
            particles_pos[i] = enforce_bounds(particles_pos[i], lower_bound, upper_bound)
        
    return gbest_pos, gbest_fitness, fitness_history

#  Execution Block for PSO 
if __name__ == "__main__":
    #  PSO Specific Parameters 
    PSO_NUM_PARTICLES = 50      # Number of particles in the swarm
    PSO_ITERATIONS = 200        # Number of iterations to run PSO
    PSO_W = 0.7                 # Inertia weight (often decays over time, but fixed here for simplicity)
    PSO_C1 = 1.5                # Cognitive coefficient (attraction to personal best)
    PSO_C2 = 1.5                # Social coefficient (attraction to global best)
    # Max velocity ratio: controls how fast particles can move. A ratio of 0.1 means max_vel = 0.1 * (UPPER_BOUND - LOWER_BOUND)
    PSO_MAX_VELOCITY_RATIO = 0.1 

    # PSO Optimization 
    pso_best_coords, pso_best_fitness, pso_history = particle_swarm_optimization_optimize(
        PSO_NUM_PARTICLES, PSO_ITERATIONS, PSO_W, PSO_C1, PSO_C2, PSO_MAX_VELOCITY_RATIO,
        LOWER_BOUND, UPPER_BOUND, DIMENSIONS
    )

    #Results 
    print("\n Particle Swarm Optimization Results ")
    print(f"  Final Best Fitness: {pso_best_fitness:.8f}")
    print(f"  Best Coordinates Found: x={pso_best_coords[0]:.6f}, y={pso_best_coords[1]:.6f}")

    #Raw Data Table (CSV) 
    csv_filename = 'pso_convergence.csv'
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Iteration', 'BestFitness']) # Header row
        for i, fitness in enumerate(pso_history):
            writer.writerow([i + 1, fitness]) # Write iteration number and best fitness
    print(f"PSO convergence data saved to {csv_filename}")

    # Plot Convergence Behavior 
    plt.figure(figsize=(8, 5))
    plt.plot(pso_history, label='Particle Swarm Optimization Convergence', color='green')
    plt.title('Particle Swarm Optimization: Convergence on Rastrigin Function')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness Found (Log Scale)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.legend()
    plt.tight_layout()
    plot_filename = 'pso_convergence_plot.png'
    plt.savefig(plot_filename)
    plt.show()
    print(f"Convergence plot saved as {plot_filename}")
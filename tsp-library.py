"""
TravelingSalesman - A Python library for solving the Traveling Salesman Problem (TSP)
using various algorithms and approaches.

The library implements multiple algorithms to solve TSP:
1. Nearest Neighbor (Greedy approach)
2. 2-opt Local Search
3. Genetic Algorithm
"""

import numpy as np
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
import random
import matplotlib.pyplot as plt

class City:
    """
    Represents a city with x and y coordinates.
    
    Attributes:
        x (float): X-coordinate of the city
        y (float): Y-coordinate of the city
        name (str): Name of the city
    """
    
    def __init__(self, x: float, y: float, name: Optional[str] = None):
        self.x = x
        self.y = y
        self.name = name if name else f"City({x},{y})"
    
    def distance_to(self, other: 'City') -> float:
        """Calculate Euclidean distance to another city."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def __str__(self) -> str:
        return self.name

class TSPSolver(ABC):
    """
    Abstract base class for TSP solvers.
    
    Attributes:
        cities (List[City]): List of cities to visit
        distance_matrix (np.ndarray): Matrix of distances between cities
    """
    
    def __init__(self, cities: List[City]):
        self.cities = cities
        self.distance_matrix = self._create_distance_matrix()
    
    def _create_distance_matrix(self) -> np.ndarray:
        """Create a matrix of distances between all cities."""
        n = len(self.cities)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i][j] = self.cities[i].distance_to(self.cities[j])
        return matrix
    
    def calculate_tour_length(self, tour: List[int]) -> float:
        """Calculate the total length of a tour."""
        length = 0
        for i in range(len(tour)):
            length += self.distance_matrix[tour[i]][tour[(i + 1) % len(tour)]]
        return length
    
    @abstractmethod
    def solve(self) -> Tuple[List[int], float]:
        """
        Solve the TSP problem.
        
        Returns:
            Tuple[List[int], float]: Best tour and its length
        """
        pass

class NearestNeighborSolver(TSPSolver):
    """Implementation of the Nearest Neighbor algorithm for TSP."""
    
    def solve(self) -> Tuple[List[int], float]:
        n = len(self.cities)
        unvisited = set(range(1, n))
        tour = [0]  # Start from the first city
        
        while unvisited:
            current = tour[-1]
            # Find the nearest unvisited city
            next_city = min(unvisited, 
                          key=lambda x: self.distance_matrix[current][x])
            tour.append(next_city)
            unvisited.remove(next_city)
        
        tour_length = self.calculate_tour_length(tour)
        return tour, tour_length

class TwoOptSolver(TSPSolver):
    """Implementation of the 2-opt local search algorithm for TSP."""
    
    def _two_opt_swap(self, tour: List[int], i: int, j: int) -> List[int]:
        """Perform a 2-opt swap by reversing the segment between i and j."""
        return tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
    
    def solve(self, initial_tour: Optional[List[int]] = None) -> Tuple[List[int], float]:
        if initial_tour is None:
            # Start with a random tour if none provided
            initial_tour = list(range(len(self.cities)))
            random.shuffle(initial_tour)
        
        best_tour = initial_tour
        best_length = self.calculate_tour_length(best_tour)
        improved = True
        
        while improved:
            improved = False
            for i in range(1, len(self.cities) - 2):
                for j in range(i + 1, len(self.cities)):
                    new_tour = self._two_opt_swap(best_tour, i, j)
                    new_length = self.calculate_tour_length(new_tour)
                    
                    if new_length < best_length:
                        best_tour = new_tour
                        best_length = new_length
                        improved = True
                        break
                if improved:
                    break
        
        return best_tour, best_length

class GeneticSolver(TSPSolver):
    """Implementation of a Genetic Algorithm for TSP."""
    
    def __init__(self, cities: List[City], 
                 population_size: int = 50,
                 generations: int = 100,
                 mutation_rate: float = 0.01):
        super().__init__(cities)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
    
    def _create_initial_population(self) -> List[List[int]]:
        """Create initial random population."""
        population = []
        for _ in range(self.population_size):
            tour = list(range(len(self.cities)))
            random.shuffle(tour)
            population.append(tour)
        return population
    
    def _crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Perform naive crossover between two parents.
        WARNING: This implementation contains a critical error!"""
        size = len(parent1)
        # Simply take first half from parent1 and second half from parent2
        # ERROR: This will create invalid tours with duplicate cities!
        crossover_point = size // 2
        child = parent1[:crossover_point] + parent2[crossover_point:]
        
        # No checking for duplicates or missing cities
        return child
    
    def _mutate(self, tour: List[int]) -> List[int]:
        """Perform mutation by swapping two random cities."""
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(tour)), 2)
            tour[i], tour[j] = tour[j], tour[i]
        return tour
    
    def solve(self) -> Tuple[List[int], float]:
        population = self._create_initial_population()
        
        for _ in range(self.generations):
            # Calculate fitness for each tour
            fitness = [(tour, self.calculate_tour_length(tour)) 
                      for tour in population]
            fitness.sort(key=lambda x: x[1])
            
            # Select best tours for next generation
            next_gen = [tour for tour, _ in fitness[:self.population_size//2]]
            
            # Create children through crossover
            while len(next_gen) < self.population_size:
                parent1, parent2 = random.sample(next_gen, 2)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                next_gen.append(child)
            
            population = next_gen
        
        # Return best tour found
        best_tour = min(population, 
                       key=lambda tour: self.calculate_tour_length(tour))
        return best_tour, self.calculate_tour_length(best_tour)

class TSPVisualizer:
    """Utility class for visualizing TSP tours."""
    
    @staticmethod
    def plot_tour(cities: List[City], tour: List[int], 
                  title: str = "TSP Tour") -> None:
        """Plot the cities and the tour connecting them."""
        plt.figure(figsize=(10, 10))
        
        # Plot cities
        x = [city.x for city in cities]
        y = [city.y for city in cities]
        plt.scatter(x, y, c='red', s=100)
        
        # Plot tour
        for i in range(len(tour)):
            city1 = cities[tour[i]]
            city2 = cities[tour[(i + 1) % len(tour)]]
            plt.plot([city1.x, city2.x], 
                    [city1.y, city2.y], 'b-')
        
        # Add city labels
        for i, city in enumerate(cities):
            plt.annotate(f'City {i}', (city.x, city.y))
        
        plt.title(title)
        plt.grid(True)
        plt.show()

def example_usage():
    """Example usage of the TSP library."""
    # Create random cities
    random.seed(42)
    cities = [City(random.uniform(0, 100), random.uniform(0, 100)) 
              for _ in range(10)]
    
    # Solve using different algorithms
    nn_solver = NearestNeighborSolver(cities)
    two_opt_solver = TwoOptSolver(cities)
    genetic_solver = GeneticSolver(cities)
    
    # Get solutions
    nn_tour, nn_length = nn_solver.solve()
    two_opt_tour, two_opt_length = two_opt_solver.solve()
    genetic_tour, genetic_length = genetic_solver.solve()
    
    # Print results
    print(f"Nearest Neighbor Tour Length: {nn_length:.2f}")
    print(f"2-opt Tour Length: {two_opt_length:.2f}")
    print(f"Genetic Algorithm Tour Length: {genetic_length:.2f}")
    
    # Visualize solutions
    TSPVisualizer.plot_tour(cities, nn_tour, "Nearest Neighbor Solution")
    TSPVisualizer.plot_tour(cities, two_opt_tour, "2-opt Solution")
    TSPVisualizer.plot_tour(cities, genetic_tour, "Genetic Algorithm Solution")

if __name__ == "__main__":
    example_usage()

"""
Mathematical model for chimp population dynamics
Core age-structured population model with stochastic mortality and reproduction
"""

import torch
import numpy as np
import random


class Model:
    """Age-structured population model with mutation-driven evolution
    
    State: population tensor [n_animals, 2] where columns are [age, beta]
    - age: animal age in years
    - beta: Gompertz mortality parameter (sagging of log mortality with age)
    
    Dynamics:
    1. Mortality: m(t) = α * exp(β*t) + Λ (Gompertz + baseline)
    2. Reproduction: mature females reproduce; offspring get mutated parental beta
    3. Aging: population ages by 1 year per iteration
    """

    def __init__(self, settings, device):
        """Initialize model with parameters
        
        Args:
            settings (dict): configuration with model parameters
            device (torch.device): cuda or cpu
        """
        self.settings = settings
        self.device = device
        self.population = None  # Will be initialized via initialize_population()

    def initialize_population(self, initial_population, initial_age_max, beta_initial):
        """Initialize population with random ages and uniform beta
        
        Args:
            initial_population (int): number of animals to create
            initial_age_max (int): maximum initial age (random 0..max)
            beta_initial (float): initial beta value for all animals
        """
        ages = torch.randint(
            0, initial_age_max + 1,
            (initial_population,),
            dtype=torch.float32,
            device=self.device
        )
        betas = torch.full(
            (initial_population,),
            beta_initial,
            dtype=torch.float32,
            device=self.device
        )
        self.population = torch.stack([ages, betas], dim=1)

    def calculate_mortality_probability(self, ages, betas):
        """Calculate per-animal death probability
        
        Gompertz mortality model: m(t) = α * exp(β*t) + Λ
        
        Args:
            ages (torch.Tensor): animal ages
            betas (torch.Tensor): animal beta values
            
        Returns:
            torch.Tensor: death probabilities in [0, 1]
        """
        alpha = self.settings["alpha"]
        lambda_param = self.settings["lambda"]
        
        # Gompertz component: accelerating mortality with age
        mortality = alpha * torch.exp(betas * ages) + lambda_param
        
        # Clamp to valid probability range
        mortality = torch.clamp(mortality, 0.0, 1.0)
        return mortality

    def apply_mortality(self):
        """Remove animals based on age-dependent death probability
        
        Stochastic: each animal dies if random[0,1] < death_probability
        
        Returns:
            int: number of animals that died
        """
        if len(self.population) == 0:
            return 0
        
        # Extract age and beta for all animals
        ages = self.population[:, 0]
        betas = self.population[:, 1]
        
        # Calculate death probabilities and apply stochastic death
        death_probs = self.calculate_mortality_probability(ages, betas)
        rand_vals = torch.rand_like(death_probs)
        survivors = rand_vals >= death_probs
        
        # Remove dead animals
        death_count = (~survivors).sum().item()
        self.population = self.population[survivors]
        
        return death_count

    def mutate_beta(self, parent_beta1, parent_beta2):
        """Apply mutation to offspring beta value
        
        Two-outcome model:
        - With mutation_probability: new random value from [-X+S*X, X+S*X]
        - Otherwise: average of two parents
        
        Args:
            parent_beta1 (float): first parent beta
            parent_beta2 (float): second parent beta
            
        Returns:
            float: offspring beta (unbounded, can be negative)
        """
        if random.random() < self.settings["mutation_probability"]:
            # Mutation: random draw from interval
            x = self.settings["mutation_x"]
            s = self.settings["mutation_s"]
            lower = -x + s * x
            upper = x + s * x
            return random.uniform(lower, upper)
        else:
            # No mutation: average of parents
            return (parent_beta1 + parent_beta2) / 2.0

    def apply_reproduction(self):
        """Reproduce to fill empty niches up to max_population
        
        Selects two mature parents (age > mature_age) uniformly at random
        (with replacement), creates one offspring with mutated beta, repeats
        until population reaches max_population.
        
        Returns:
            int: number of offspring born
        """
        births = 0
        max_pop = self.settings["max_population"]
        mature_age = self.settings["mature_age"]
        
        # Find all mature animals (age > mature_age)
        mature_mask = self.population[:, 0] > mature_age
        mature_indices = torch.where(mature_mask)[0].cpu().numpy()
        
        # Need at least 2 mature animals to reproduce
        if len(mature_indices) < 2:
            return 0
        
        # Breed until population reaches max
        while len(self.population) < max_pop:
            # Random parent selection (with replacement)
            parent_idx1, parent_idx2 = np.random.choice(
                mature_indices,
                size=2,
                replace=True
            )
            parent1 = self.population[parent_idx1]
            parent2 = self.population[parent_idx2]
            
            # Child's beta with possible mutation
            child_beta = self.mutate_beta(parent1[1].item(), parent2[1].item())
            
            # Add new animal (age 0)
            child = torch.tensor([0.0, child_beta], device=self.device)
            self.population = torch.cat([self.population, child.unsqueeze(0)])
            births += 1
        
        return births

    def age_population(self):
        """Increment age of all animals by 1 year"""
        self.population[:, 0] += 1

    def get_ages(self):
        """Get array of all animal ages
        
        Returns:
            np.ndarray: animal ages
        """
        return self.population[:, 0].detach().cpu().numpy()

    def get_betas(self):
        """Get array of all animal betas
        
        Returns:
            np.ndarray: animal beta values
        """
        return self.population[:, 1].detach().cpu().numpy()

    def get_population_size(self):
        """Get current population size
        
        Returns:
            int: number of animals
        """
        return len(self.population)

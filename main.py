import random
import math

# --- Parâmetros Variáveis (AUMENTADOS) ---
POPULATION_SIZE = 30     # População Aumentada: 30
GENERATIONS = 20         # Gerações Aumentadas: 20
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.01
INTERVAL_START = -10.0
INTERVAL_END = 10.0
PRECISION = 4
TOURNAMENT_SIZE = 2

# --- Funções do AG (Sem alteração) ---
def fitness_function(x):
    return x**2 - 3*x + 4

def get_bit_length(start, end, precision):
    range_val = end - start
    length = math.log2(range_val * (10 ** precision))
    return math.ceil(length)

CHROMOSOME_LENGTH = get_bit_length(INTERVAL_START, INTERVAL_END, PRECISION)

def decode_chromosome(chromosome):
    binary_string = "".join(map(str, chromosome))
    integer_value = int(binary_string, 2)
    range_val = INTERVAL_END - INTERVAL_START
    max_integer_value = 2**CHROMOSOME_LENGTH - 1
    x = INTERVAL_START + (integer_value / max_integer_value) * range_val
    return round(x, PRECISION)

def initialize_population(size, length):
    population = []
    for _ in range(size):
        individual = [random.randint(0, 1) for _ in range(length)]
        population.append(individual)
    return population

def selection(population, fitnesses, k=TOURNAMENT_SIZE):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitnesses)), k)
        winner = max(tournament, key=lambda item: item[1])[0]
        selected.append(winner)
    return selected

def crossover(parent1, parent2, rate=CROSSOVER_RATE):
    if random.random() < rate:
        point = random.randint(1, CHROMOSOME_LENGTH - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[point:] + parent1[:point]
        return child1, child2
    else:
        return parent1, parent2

def mutation(chromosome, rate=MUTATION_RATE):
    mutated_chromosome = chromosome[:]
    for i in range(len(mutated_chromosome)):
        if random.random() < rate:
            mutated_chromosome[i] = 1 - mutated_chromosome[i]
    return mutated_chromosome

# --- Algoritmo Genético Principal ---
def genetic_algorithm_test():
    population = initialize_population(POPULATION_SIZE, CHROMOSOME_LENGTH)
    
    best_overall_x = None
    best_overall_fitness = -float('inf')
    
    # Lista para acompanhar a melhor aptidão em cada geração
    generation_results = []

    for generation in range(GENERATIONS):
        decoded_x_values = [decode_chromosome(individual) for individual in population]
        fitnesses = [fitness_function(x) for x in decoded_x_values]

        current_best_fitness = max(fitnesses)
        current_best_index = fitnesses.index(current_best_fitness)
        current_best_x = decoded_x_values[current_best_index]

        if current_best_fitness > best_overall_fitness:
            best_overall_fitness = current_best_fitness
            best_overall_x = current_best_x
        
        generation_results.append((generation + 1, current_best_x, current_best_fitness))

        # 2. Seleção
        parents = selection(population, fitnesses)

        # 3. Reprodução (Crossover e Mutação)
        next_population = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            if i + 1 < len(parents):
                parent2 = parents[i + 1]
            else:
                next_population.append(mutation(parent1))
                break
            
            child1, child2 = crossover(parent1, parent2, CROSSOVER_RATE)
            
            mutated_child1 = mutation(child1, MUTATION_RATE)
            mutated_child2 = mutation(child2, MUTATION_RATE)
            
            next_population.extend([mutated_child1, mutated_child2])

        population = next_population[:POPULATION_SIZE]

    # Imprime os resultados de cada geração
    for gen, x, f_x in generation_results:
        print(f"--- Geração {gen} ---")
        print(f"Melhor x da Geração: {x:.{PRECISION}f}, f(x): {f_x:.{PRECISION}f}")

    print("\n--- Resultado Final ---")
    print(f"Melhor x Encontrado: {best_overall_x:.{PRECISION}f}")
    print(f"Valor Máximo da Função (f(x)): {best_overall_fitness:.{PRECISION}f}")
    
    return best_overall_x, best_overall_fitness

if __name__ == "__main__":
    random.seed(42) # Mantendo a seed para comparação
    print(f"Comprimento do Cromossomo (Bits): {CHROMOSOME_LENGTH}")
    genetic_algorithm_test()
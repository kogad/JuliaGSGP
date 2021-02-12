mutable struct Config{S, T, U}
    population_size::Int
    num_generations::Int
    p_crossover::Float64
    p_mutation::Float64
    mutation_step::Float64
    min_depth::Int
    max_depth::Int
    tournament_size::Int
    pset::PrimitiveSet
    create_rand_tree::S
    create_population::T
    fitness::U

    p_mate::Float64
end

function Config(;population_size,
                 num_generations,
                 p_crossover,
                 p_mutation,
                 mutation_step,
                 min_depth,
                 max_depth,
                 tournament_size,
                 pset,
                 create_rand_tree,
                 create_population,
                 fitness,
                 p_mate = 0.25
                 )
    Config(population_size, num_generations, p_crossover, p_mutation, mutation_step, min_depth, max_depth, tournament_size, pset, create_rand_tree, create_population, fitness, p_mate)
end
struct Dataset{T, S}
    # if T is a Matrix, each of its columns should correspond to each of input data
    input::T
    target::S
end

function update_fitness!(fitness, pop)
    for i in 1:length(pop)
        fitness[i] = entry_pool[pop[i]].train_fitness
    end
end

function update_test_fitness!(fitness, pop)
    for i in 1:length(pop)
        fitness[i] = entry_pool[pop[i]].test_fitness
    end
end
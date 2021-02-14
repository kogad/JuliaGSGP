# Deterministic GSGP
function dgsgp(cfg, X_train, y_train, X_test, y_test; rec=Record(), seed=1, verbose=true)
    Random.seed!(seed)
    init_id()
    init_ref_counter(cfg)
    init_entry_pool()

    data_train = Dataset(X_train, y_train)
    data_test  = Dataset(X_test, y_test)

    init_entries = cfg.create_population(cfg, data_train, data_test)
    add_entry!.(init_entries)

    pop = getfield.(init_entries, :index)
    new_pop = similar(pop)
    old_pop = similar(pop)

    fitness = getfield.(init_entries, :train_fitness)

    register!(rec, "train_fitness", train_fitness, Float64)
    register!(rec, "test_fitness", test_fitness, Float64)
    add_record!(rec, 1, pop, cfg)

    verbose && print_latest_fitness(rec, 1)

    overwrite = false

    for g in 2:cfg.num_generations
        overwrite = g > 2 
        elite_idx = argmin(fitness)
        new_pop[1] = pop[elite_idx]
        avoid_overwrite(elite_idx)
        for i in 2:cfg.population_size
            rand_num = rand()

            if rand_num < cfg.p_crossover
                p1 = tournament_selection(pop, fitness, cfg)
                p2 = tournament_selection(pop, fitness, cfg)
                offspring = deterministic_crossover(p1, p2, data_train, data_test, cfg, overwrite)
                new_pop[i] = offspring.index
            else
                p = tournament_selection(pop, fitness, cfg)
                offspring = gsmutation(p, data_train, data_test, cfg, overwrite)
                new_pop[i] = offspring.index
            end
            # TODO: reproduction
        end

        update_fitness!(fitness, new_pop)
        add_record!(rec, 1, pop, cfg)

        g > 2 && update_entry_pool!(old_pop)


        old_pop = copy(pop)
        pop = copy(new_pop)

        verbose && print_latest_fitness(rec, g)
    end

    return pop[argmin(fitness)], rec
end

function deterministic_crossover(t1, t2, data_train, data_test, cfg, overwrite)
    increment_ref(entry_pool[t1])
    increment_ref(entry_pool[t2])

    local r, offspring
    if overwrite
        offspring = deterministic_crossover_entry(entry_pool[t1], entry_pool[t2], data_train, data_test, cfg)
    else
        offspring = deterministic_crossover_entry(entry_pool[t1], entry_pool[t2], data_train, data_test, cfg, index=get_counter_size()+1)
    end

    add_entry!(offspring)
    set_ref(offspring)

    # to avoid overwritten 
    # need to decrement at the end of current generation
    increment_ref(offspring)

    return offspring
end

function deterministic_crossover_entry(t1, t2, data_train, data_test, cfg; index=get_usable_index())

    r = deterministic_coeff(t1.train_semantics, t2.train_semantics, data_train.target)

    train_semantics = Vector{Float64}(undef, length(data_train.target))
    @inbounds @simd for i in 1:length(train_semantics)
        train_semantics[i] = r * t1.train_semantics[i] + (1-r) * t2.train_semantics[i]
    end

    test_semantics = Vector{Float64}(undef, length(data_test.target))
    @inbounds @simd for i in 1:length(test_semantics)
        test_semantics[i] = r * t1.test_semantics[i] + (1-r) * t2.test_semantics[i]
    end

    train_fitness = cfg.fitness(train_semantics, data_train)
    test_fitness = cfg.fitness(test_semantics, data_test)

    id = get_new_id()
    #index = get_usable_index()

    return Entry(id, index, crossover, dummy_node(), (t1.id, t2.id, -1), (t1.index, t2.index, -1), train_semantics, test_semantics, train_fitness, test_fitness)
end

function dgsgp_opt_mate(cfg, data_train, data_test; seed=1, verbose=true)
    Random.seed!(seed)
    init_id()
    init_ref_counter(cfg)
    init_entry_pool()

    init_entries = cfg.create_population(cfg, data_train, data_test)
    add_entry!.(init_entries)

    pop = getfield.(init_entries, :index)
    new_pop = similar(pop)
    old_pop = similar(pop)

    fitness = getfield.(init_entries, :train_fitness)
    rec = Record()
    register!(rec, "train_fitness", train_fitness, Float64)
    register!(rec, "test_fitness", test_fitness, Float64)
    add_record!(rec, 1, pop, cfg)

    verbose && println(1, ": ", minimum(fitness))
    overwrite = false


    for g in 2:cfg.num_generations
        overwrite = g != 2 
        elite_idx = argmin(fitness)
        new_pop[1] = pop[elite_idx]
        avoid_overwrite(elite_idx)
        for i in 2:cfg.population_size
            rand_num = rand()

            if rand_num < cfg.p_crossover
                p1 = tournament_selection(pop, fitness, cfg)
                local p2
                if rand() < cfg.p_mate
                    p2 = optimal_mate(p1, data_train.target, pop)
                else
                    p2 = tournament_selection(pop, fitness, cfg)
                end
                offspring = deterministic_crossover(p1, p2, data_train, data_test, cfg, overwrite)
                new_pop[i] = offspring.index
            else
                p = tournament_selection(pop, fitness, cfg)
                offspring = gsmutation(p, data_train, data_test, cfg, overwrite)
                new_pop[i] = offspring.index
            end
        end

        update_fitness!(fitness, new_pop)
        add_record!(rec, g, new_pop, cfg)

        g > 2 && update_entry_pool!(old_pop)

        old_pop = copy(pop)
        pop = copy(new_pop)

        verbose && println(g, ": ", minimum(fitness))
    end

    return pop[argmin(fitness)], rec
end



deterministic_coeff(t1, t2, p) = ifelse(t1 ≈ t2, 0.5, (norm(t1-t2) - norm(p-t1)*costheta(t2-t1, p-t1))/norm(t1-t2))

costheta(v1, v2) = dot(v1, v2)/(norm(v1)*norm(v2))


function optimal_mate(p1, target, pop)
    c = Vector{Float64}(undef, length(pop))
    idx = 0
    for i in 1:length(c)
        if pop[i] == p1
            idx = i
            continue
        end
        c[i] = costheta(target - sem(p1), sem(pop[i]) - sem(p1))
    end
    c[idx] = -Inf

    return pop[argmax(c)]
end

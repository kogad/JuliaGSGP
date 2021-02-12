function prgp(cfg, X_train, y_train, X_test, y_test; rec=Record(), seed=1, verbose=true)
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

    train_rec = rec
    add_record!(rec, 1, pop, cfg)

    overwrite = false
    verbose && print_latest_fitness(rec, 1)

    for g in 2:cfg.num_generations
        overwrite = g == 2 
        elite_idx = argmin(fitness)
        new_pop[1] = pop[elite_idx]
        avoid_overwrite(elite_idx)
        for i in 2:cfg.population_size
            rand_num = rand()

            if rand_num < cfg.p_crossover
                p1 = tournament_selection(pop, fitness, cfg)
                p2 = tournament_selection(pop, fitness, cfg)
                offspring = projection_crossover(p1, p2, data_train, data_test, cfg, overwrite)
                new_pop[i] = offspring.index
            else
                p = tournament_selection(pop, fitness, cfg)
                offspring = projection_mutation(p, data_train, data_test, cfg, overwrite)
                new_pop[i] = offspring.index
            end
            # TODO: reproduction
        end

        update_fitness!(fitness, new_pop)
        add_record!(rec, g, pop, cfg)

        g > 2 && update_entry_pool!(old_pop)

        old_pop = copy(pop)
        pop = copy(new_pop)

        verbose && print_latest_fitness(rec, g)
    end

    return pop[argmin(fitness)], rec
end

optimal_coeff(v1, v2, t) = pinv([v1 v2])*t

function projection_crossover(t1, t2, data_train, data_test, cfg, overwrite)
    increment_ref(entry_pool[t1])
    increment_ref(entry_pool[t2])

    local r, offspring
    if overwrite
        offspring = projection_crossover_entry(entry_pool[t1], entry_pool[t2], data_train, data_test, cfg)
    else
        offspring = projection_crossover_entry(entry_pool[t1], entry_pool[t2], data_train, data_test, cfg, index=get_counter_size()+1)
    end

    add_entry!(offspring)
    set_ref(offspring)

    # to avoid overwritten 
    # need to decrement at the end of current generation
    increment_ref(offspring)

    return offspring
end

function projection_crossover_entry(t1, t2, data_train, data_test, cfg; index=get_usable_index())

    r = optimal_coeff(t1.train_semantics, t2.train_semantics, data_train.target)

    train_semantics = Vector{Float64}(undef, length(data_train.target))
    @inbounds @simd for i in 1:length(train_semantics)
        train_semantics[i] = r[1] * t1.train_semantics[i] + r[2] * t2.train_semantics[i]
    end

    test_semantics = Vector{Float64}(undef, length(data_test.target))
    @inbounds @simd for i in 1:length(test_semantics)
        test_semantics[i] = r[1] * t1.test_semantics[i] + r[2] * t2.test_semantics[i]
    end

    train_fitness = cfg.fitness(train_semantics, data_train)
    test_fitness = cfg.fitness(test_semantics, data_test)

    id = get_new_id()

    return Entry(id, index, crossover, dummy_node(), (t1.id, t2.id, -1), (t1.index, t2.index, -1), train_semantics, test_semantics, train_fitness, test_fitness)
end


function projection_mutation(t, data_train, data_test, cfg, overwrite)
    increment_ref(entry_pool[t])

    local offspring, r1, r2
    if overwrite
        r1 = Entry(cfg.create_rand_tree(cfg.pset, cfg.min_depth, cfg.max_depth), data_train, data_test, cfg)
        add_entry!(r1)
        set_ref(r1)
        increment_ref(r1)

        r2 = Entry(cfg.create_rand_tree(cfg.pset, cfg.min_depth, cfg.max_depth), data_train, data_test, cfg)
        add_entry!(r2)
        set_ref(r2)
        increment_ref(r2)

        offspring = projection_mutation_entry(entry_pool[t], r1, r2,data_train, data_test, cfg)
    else
        r1 = Entry(cfg.create_rand_tree(cfg.pset, cfg.min_depth, cfg.max_depth), data_train, data_test, cfg, index=get_counter_size()+1)
        add_entry!(r1)
        set_ref(r1)
        increment_ref(r1)

        r2 = Entry(cfg.create_rand_tree(cfg.pset, cfg.min_depth, cfg.max_depth), data_train, data_test, cfg, index=get_counter_size()+1)
        add_entry!(r2)
        set_ref(r2)
        increment_ref(r2)

        offspring = projection_mutation_entry(entry_pool[t], r1, r2, data_train, data_test, cfg, index=get_counter_size()+1)
    end

    add_entry!(offspring)
    set_ref(offspring)
    
    # to avoid overwritten 
    # need to decrement at the end of current generation
    increment_ref(offspring)

    return offspring
end


function projection_mutation_entry(t, r1, r2, data_train, data_test, cfg;index=get_usable_index())

    coeff = optimal_coeff(t.train_semantics, r1.train_semantics-r2.train_semantics, data_train.target)

    train_semantics = Vector{Float64}(undef, length(data_train.target))
    @inbounds @simd for i in 1:length(train_semantics)
        train_semantics[i] = coeff[1]*t.train_semantics[i] + coeff[2]*(r1.train_semantics[i] - r2.train_semantics[i])
    end

    test_semantics = Vector{Float64}(undef, length(data_test.target))
    @inbounds @simd for i in 1:length(test_semantics)
        test_semantics[i] = coeff[1]*t.test_semantics[i] + coeff[2]*(r1.test_semantics[i] - r2.test_semantics[i])
    end

    train_fitness = cfg.fitness(train_semantics, data_train)
    test_fitness = cfg.fitness(test_semantics, data_test)

    id = get_new_id()

    return Entry(id, index, mutation, dummy_node(), (t.id, r1.id, r2.id), (t.index, r1.index, r2.index), train_semantics, test_semantics, train_fitness, test_fitness)
end
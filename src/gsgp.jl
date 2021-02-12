function gsgp(cfg, X_train, y_train, X_test, y_test; rec=Record(), seed=rand(UInt), verbose=true)
    Random.seed!(seed)

    # these should be executed before main process of GSGP
    init_id()
    init_ref_counter(cfg)
    init_entry_pool()

    # for simplicity
    data_train = Dataset(X_train, y_train)
    data_test  = Dataset(X_test, y_test)

    # initial population (individuals are treated as "entry")
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

    # control whether entries that are no longer referenced can be overwritten
    overwrite = false

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
                offspring = gscrossover(p1, p2, data_train, data_test, cfg, overwrite)
                new_pop[i] = offspring.index
            else
                p = tournament_selection(pop, fitness, cfg)
                offspring = gsmutation(p, data_train, data_test, cfg, overwrite)
                new_pop[i] = offspring.index
            end
        end

        update_fitness!(fitness, new_pop)

        g > 2 && update_entry_pool!(old_pop)

        add_record!(rec, g, pop, cfg)

        old_pop = copy(pop)
        pop = copy(new_pop)

        verbose && print_latest_fitness(rec, g)
    end

    return pop[argmin(fitness)], rec
end

function gscrossover(t1, t2, data_train, data_test, cfg, overwrite)
    increment_ref(entry_pool[t1])
    increment_ref(entry_pool[t2])

    local r, offspring
    if overwrite
        r = Entry(cfg.create_rand_tree(cfg.pset, cfg.min_depth, cfg.max_depth), data_train, data_test, cfg) 
        add_entry!(r)
        set_ref(r)
        increment_ref(r)

        offspring = Entry(entry_pool[t1], entry_pool[t2], r, crossover, data_train, data_test, cfg)
    else
        r = Entry(cfg.create_rand_tree(cfg.pset, cfg.min_depth, cfg.max_depth), data_train, data_test, cfg, index=get_counter_size()+1)
        add_entry!(r)
        set_ref(r)
        increment_ref(r)

        offspring = Entry(entry_pool[t1], entry_pool[t2], r, crossover, data_train, data_test, cfg, index=get_counter_size()+1)
    end

    add_entry!(offspring)
    set_ref(offspring)

    # to avoid overwritten 
    increment_ref(offspring)

    return offspring
end

function gsmutation(t, data_train, data_test, cfg, overwrite)
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

        offspring = Entry(entry_pool[t], r1, r2, mutation, data_train, data_test, cfg)
    else
        r1 = Entry(cfg.create_rand_tree(cfg.pset, cfg.min_depth, cfg.max_depth), data_train, data_test, cfg, index=get_counter_size()+1)
        add_entry!(r1)
        set_ref(r1)
        increment_ref(r1)

        r2 = Entry(cfg.create_rand_tree(cfg.pset, cfg.min_depth, cfg.max_depth), data_train, data_test, cfg, index=get_counter_size()+1)
        add_entry!(r2)
        set_ref(r2)
        increment_ref(r2)

        offspring = Entry(entry_pool[t], r1, r2, mutation, data_train, data_test, cfg, index=get_counter_size()+1)
    end

    add_entry!(offspring)
    set_ref(offspring)
    
    # to avoid overwritten
    increment_ref(offspring)

    return offspring
end

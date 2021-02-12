function tournament_selection(pop, fitness, cfg)
    tournament = rand(1:length(fitness), cfg.tournament_size)
    winner = tournament[argmin(fitness[tournament])]
    return pop[winner]
end

random_selection(pop) = rand(pop)
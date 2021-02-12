@testset "PrGP" begin
    
    v1 = [1, 0, 0]
    v2 = [0, 1, 0]
    t  = [3, 3, 4]
    @test optimal_coeff(v1, v2, t) ≈ [3, 3]

    t  = [3, 3, 100101010]
    @test optimal_coeff(v1, v2, t) ≈ [3, 3]

    v1 = [1, 0, 1]
    v2 = [0, 1, 0]
    t  = [3, 3, 4]
    @test optimal_coeff(v1, v2, t) ≈ [3.5, 3.0]

    v1 = [1, 1, 1]
    v2 = [-1, 1, 1]
    @test optimal_coeff(v1, v2, t) ≈ [3.25, 0.25]

    v1 = v2 = [1,1,1]
    t = [10, 10, 10,]
    @test optimal_coeff(v1, v2, t) ≈ [5, 5]


    cfg = Config(population_size=100, num_generations=10, p_crossover=0.5, p_mutation=0.5, mutation_step=0.01, min_depth=1, max_depth=6, tournament_size=4, pset=PrimitiveSet(), create_rand_tree=generate_grow, create_population=ramped_half_and_half, fitness=rmse)
    dataset = Dataset(zeros(1, 1), [3, 3, 4])

    e1 = Entry(1, 1, JuliaGSGP.none, JuliaGSGP.dummy_node(), (-1, -1, -1), (-1, -1, -1), [1, 0, 1], [1, 0, 1], 10, 10)
    e2 = Entry(1, 1, JuliaGSGP.none, JuliaGSGP.dummy_node(), (-1, -1, -1), (-1, -1, -1), [0, 1, 0], [0, 1, 0], 10, 10)
    e3 = projection_crossover_entry(e1, e2, dataset, dataset, cfg)

    @test e3.train_semantics ≈ [3.5, 3., 3.5]

end
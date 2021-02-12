using JuliaGSGP
using CSV, DataFrames
using ScikitLearn: CrossValidation.train_test_split

# df = CSV.read("path/to/dataset", DataFrame)
df = df[2:end]
data = convert(Matrix, df)
X = data[:, 1:end-1]
y = data[:, end]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
X_train = X_train' # columns corresponds to one instances
X_test = X_test'

aq(a, b) = a / sqrt(1+b^2)

pset = PrimitiveSet()
add_function!(pset, +, 2) #(pset, function, arity)
add_function!(pset, -, 2)
add_function!(pset, *, 2)
add_function!(pset, aq, 2)

for i in 1:size(X)[2]
    add_variable!(pset, i)
end

cfg = Config(population_size=100, num_generations=100, p_crossover=0.5, p_mutation=0.5, mutation_step=0.01, min_depth=1, max_depth=6, tournament_size=4, pset=pset, create_rand_tree=generate_grow, create_population=ramped_half_and_half, fitness=rmse)

best_idx, rec = gsgp(cfg, X_train, y_train, X_test, y_test);

# `reconstruct_tree()` will take long time to execute.
# tree = reconstruct_tree(best_idx, cfg.mutation_step)
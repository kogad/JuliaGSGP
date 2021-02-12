# JuliaGSGP

`JuliaGSGP` is a Julia implementation of Geometric Semantic Genetic Programming (GSGP) based on [1].

`JuliaGSGP` also includes implementations of Deterministic GSGP (with optimal mate selection)[2] and PrGP[3].

Documentation will be provided in the future.

# Exapmle

```julia
using JuliaGSGP
using CSV, DataFrames
using ScikitLearn: CrossValidation.train_test_split

df = CSV.read("path/to/dataset", DataFrame)
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
tree = reconstruct_tree(best_idx, cfg,mutation_step)
```

# Reference

[1] Vanneschi L., Castelli M., Manzoni L., Silva S. (2013) A New Implementation of Geometric Semantic GP and Its Application to Problems in Pharmacokinetics. In: Krawiec K., Moraglio A., Hu T., Etaner-Uyar A.Ş., Hu B. (eds) Genetic Programming. EuroGP 2013. Lecture Notes in Computer Science, vol 7831. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-37207-0_18

[2] A. Hara, J. Kushida and T. Takahama, "Deterministic Geometric Semantic Genetic Programming with Optimal Mate Selection," 2016 IEEE International Conference on Systems, Man, and Cybernetics (SMC), Budapest, 2016, pp. 003387-003392, doi: 10.1109/SMC.2016.7844757.

[3] Graff, Mario & Tellez, Eric & Villaseñor García, Elio & Miranda-Jiménez, Sabino. (2015). Semantic Genetic Programming Operators Based on Projections in the Phenotype Space. Research in Computing Science. 94. 10.13053/rcs-94-1-6.

@enum Operator crossover mutation none

struct Entry
    id::Int 
    index::Int # index in `entry_pool`
    operator::Operator
    tree::GPNode
    id_entry::Tuple{Int, Int, Int} # parents' id
    index_entry::Tuple{Int, Int, Int} #parent's index
    train_semantics::Vector{Float64}
    test_semantics::Vector{Float64}
    train_fitness::Float64
    test_fitness::Float64
end

function Entry(tree, data_train, data_test, cfg; index=get_usable_index())
    train_semantics = Vector{Float64}(undef, length(data_train.target))
    for i in 1:length(train_semantics)
        train_semantics[i] =  evaluate(tree, view(data_train.input, :, i))
    end

    test_semantics = Vector{Float64}(undef, length(data_test.target))
    for i in 1:length(test_semantics)
        test_semantics[i] =  evaluate(tree, view(data_test.input, :, i))
    end

    train_fitness = cfg.fitness(train_semantics, data_train)
    test_fitness = cfg.fitness(test_semantics, data_test)

    id = get_new_id()
    # index = get_usable_index()

    return Entry(id, index, none, tree, (-1, -1, -1), (-1, -1, -1), train_semantics, test_semantics, train_fitness, test_fitness)
end

function Entry(t1, t2, t3, op, data_train, data_test, cfg; index=get_usable_index())
    if op == crossover
        return crossover_entry(t1, t2, t3, data_train, data_test, cfg, index)
    elseif op == mutation
        return mutation_entry(t1, t2, t3, data_train, data_test, cfg, index)
    end
end

sem(idx) = entry_pool[idx].train_semantics
test_sem(idx) = entry_pool[idx].test_semantics

function crossover_entry(t1, t2, r, data_train, data_test, cfg, index)
    train_semantics = Vector{Float64}(undef, length(data_train.target))
    @inbounds @simd for i in 1:length(train_semantics)
        train_semantics[i] = sigmoid(r.train_semantics[i]) * t1.train_semantics[i] + (1 - sigmoid(r.train_semantics[i])) * t2.train_semantics[i]
    end

    test_semantics = Vector{Float64}(undef, length(data_test.target))
    @inbounds @simd for i in 1:length(test_semantics)
        test_semantics[i] = sigmoid(r.test_semantics[i]) * t1.test_semantics[i] + (1 - sigmoid(r.test_semantics[i])) * t2.test_semantics[i]
    end

    train_fitness = cfg.fitness(train_semantics, data_train)
    test_fitness = cfg.fitness(test_semantics, data_test)

    id = get_new_id()
    #index = get_usable_index()

    return Entry(id, index, crossover, dummy_node(), (t1.id, t2.id, r.id), (t1.index, t2.index, r.index), train_semantics, test_semantics, train_fitness, test_fitness)
end

function mutation_entry(t, r1, r2, data_train, data_test, cfg, index)
    train_semantics = Vector{Float64}(undef, length(data_train.target))
    @inbounds @simd for i in 1:length(train_semantics)
        train_semantics[i] = t.train_semantics[i] + cfg.mutation_step * (sigmoid(r1.train_semantics[i]) - sigmoid(r2.train_semantics[i]))
    end

    test_semantics = Vector{Float64}(undef, length(data_test.target))
    @inbounds @simd for i in 1:length(test_semantics)
        test_semantics[i] = t.test_semantics[i] + cfg.mutation_step * (sigmoid(r1.test_semantics[i]) - sigmoid(r2.test_semantics[i]))
    end

    train_fitness = cfg.fitness(train_semantics, data_train)
    test_fitness = cfg.fitness(test_semantics, data_test)

    id = get_new_id()
    #index = get_usable_index()

    return Entry(id, index, mutation, dummy_node(), (t.id, r1.id, r2.id), (t.index, r1.index, r2.index), train_semantics, test_semantics, train_fitness, test_fitness)
end

const id_counter = [0]

function init_id()
    id_counter[1] = 0
end

function get_new_id()
    id_counter[1] += 1
    return id_counter[1]
end

# population like the one in Fig.2 in [2]
const entry_pool = Entry[]

function init_entry_pool()
    while !isempty(entry_pool)
        pop!(entry_pool)
    end
end

function add_entry!(entry)
    if entry.index <= length(entry_pool)
        entry_pool[entry.index] = entry
    elseif entry.index == length(entry_pool)+1
        push!(entry_pool, entry)
    else
        d = entry.index - length(entry_pool)
        append!(entry_pool, Vector{Entry}(undef, d))
        entry_pool[entry.index] = entry
    end
end
get_entry_pool_size() = length(entry_pool)


# reference management

const ref_counter = PriorityQueue{Int, Int}()

function set_ref(entry)
    ref_counter[entry.index] = 0 # new individuals are not referenced
end

function init_ref_counter(cfg)
    while !isempty(ref_counter)
        dequeue!(ref_counter)
    end

    for i in 1:cfg.population_size
        ref_counter[i] = 0
    end
end

function increment_ref(entry)
    ref_counter[entry.index] += 1
end

function avoid_overwrite(idx)
    ref_counter[idx] += 1
end

# return available (unreferenced) index
function get_usable_index()
    if isempty(ref_counter)
        return 1
    elseif peek(ref_counter)[2] == 0
        return peek(ref_counter)[1]
    else
        return length(ref_counter) + 1
    end
end

get_counter_size() = length(ref_counter)


function update_entry_pool!(pop)
    # new individual's ref_counter is 1 to avoid overwritten whether referenced or not, so it has to be decremented
    for i in pop
        decrement_ref!(i)
    end
    _update_entry_pool!(pop)
end

# decrement unreferenced entry's parents' ref_counter recursively
function _update_entry_pool!(pop)
    s = Set{Int}()
    for i in pop
        if ref_counter[i] == 0
            for j in entry_pool[i].index_entry
                if j != -1
                    decrement_ref!(j)
                    push!(s, j)
                end
            end
        end
    end

    isempty(s) && return
    _update_entry_pool!(s)
end

function decrement_ref!(entry)
    ref_counter[entry.index] -= 1
end

function decrement_ref!(index::Int)
    ref_counter[index] -= 1
end


is_rand_tree(index) = entry_pool[index].index_entry[1] == -1


# reconstruction
function reconstruct_tree(idx, ms)
    entry = entry_pool[idx]
    refs = entry.index_entry
    if entry.operator == none
        return entry.tree
    end

    if entry.operator == crossover
        t1 = reconstruct_tree(refs[1], ms)
        t2 = reconstruct_tree(refs[2], ms)
        c =  crossover_tree(t1, t2, entry_pool[refs[3]])
        return c
    else
        t1 = reconstruct_tree(refs[1], ms)
        return mutation_tree(t1, entry_pool[refs[2]], entry_pool[refs[3]], ms)
    end
end

function crossover_tree(t1, t2, r)
    add_node = GPNode(PrimitiveFunction(+, 2))
    sub_node = GPNode(PrimitiveFunction(-, 2))
    mul_node1 = GPNode(PrimitiveFunction(*, 2))
    mul_node2 = GPNode(PrimitiveFunction(*, 2))

    sig_node = GPNode(PrimitiveFunction(sigmoid, 1))
    sig_node.children[1] = r.tree

    if t1 isa GPNode
        mul_node1.children[1] = t1
    else
        mul_node1.children[1] = t1.tree
    end
    mul_node1.children[2] = sig_node 

    sub_node.children[1] = GPNode(Const(1.0))
    sub_node.children[2] =  sig_node

    if t2 isa GPNode
        mul_node2.children[1] = t2
    else
        mul_node2.children[1] = t2.tree
    end
    mul_node2.children[2] = sub_node

    add_node.children[1] = mul_node1
    add_node.children[2] = mul_node2

    return add_node
end

function mutation_tree(t, r1, r2, ms)
    add_node = GPNode(PrimitiveFunction(+, 2))
    sub_node = GPNode(PrimitiveFunction(-, 2))
    mul_node = GPNode(PrimitiveFunction(*, 2))

    sig_node1 = GPNode(PrimitiveFunction(sigmoid, 1))
    sig_node2 = GPNode(PrimitiveFunction(sigmoid, 1))
    sig_node1.children[1] = r1.tree
    sig_node2.children[1] = r2.tree

    sub_node.children[1] = sig_node1
    sub_node.children[2] = sig_node2 

    mul_node.children[1] = GPNode(Const(ms))
    mul_node.children[2] = sub_node

    if t isa GPNode
        add_node.children[1] = t
    else
        add_node.children[1] =  t.tree
    end
    add_node.children[2] = mul_node

    return add_node
end
abstract type Primitive end
abstract type Terminal <: Primitive end

struct PrimitiveFunction{T} <: Primitive
    func::T
    arity::Int
end

struct Variable{T} <: Terminal
    name::T
end

struct Const{T <: Real} <: Terminal
    value::T
end

struct RandConst <: Terminal
end

protected_div(x, y) = ifelse(y == 0, 1, x / y)
sigmoid(x) = 1 / (1 + exp(-x))

mutable struct PrimitiveSet
    functions::Vector{PrimitiveFunction}
    terminals::Vector{Terminal}

    terminal_ratio::Float64
    function_ratio::Float64

    PrimitiveSet() = new([], [], 0, 0)
end

arity(x::PrimitiveFunction) = x.arity
arity(x::Terminal) = 0

function update_ratio!(pset)
    function_ratio = length(pset.functions) / (length(pset.functions) + length(pset.terminals))
    pset.function_ratio = function_ratio
    pset.terminal_ratio = 1 - function_ratio
end

function add_function!(pset::PrimitiveSet, func::Function, arity::Int)
    prim = PrimitiveFunction(func, arity)
    push!(pset.functions, prim)
    update_ratio!(pset)
end

function add_const!(pset::PrimitiveSet, val)
    term = Const(val)
    push!(pset.terminals, term)
    update_ratio!(pset)
end

function add_rand!(pset)
    push!(pset.terminals, RandConst())
    update_ratio!(pset)
end

function add_variable!(pset, varname)
    var = Variable(varname)
    push!(pset.terminals, var)
    update_ratio!(pset)
end

struct GPNode{T <: Primitive}
    primitive::T
    children::Vector{GPNode}
end

GPNode(prim::PrimitiveFunction) = GPNode(prim, Vector{GPNode}(undef, prim.arity))
GPNode(prim::Terminal) = GPNode(prim, GPNode[])

dummy() = error("dummy node")
dummy_node() = GPNode(PrimitiveFunction(dummy, 0))

function Base.rand(x::Vector{Terminal})
    r = rand(1:length(x))
    if x[r] isa RandConst
        return Const(rand())
    else
        return x[r]
    end
end


function generate(pset, min_depth, max_depth, terminal_condition)
    local node
    if terminal_condition(pset, 0, min_depth, max_depth)
        prim = rand(pset.terminals)
        node = GPNode(prim)
    else
        prim = rand(pset.functions)
        node = GPNode(prim)
        _generate!(node, pset, 1, min_depth, max_depth, terminal_condition)
    end
    return node
end

function _generate!(node, pset, depth, min_depth, max_depth, terminal_condition)
    for i in 1:node.primitive.arity
        if terminal_condition(pset, depth, min_depth, max_depth)
            prim = rand(pset.terminals)
            node.children[i] = GPNode(prim)
        else
            prim = rand(pset.functions)
            node.children[i] = GPNode(prim)
        end
        if node.children[i].primitive isa PrimitiveFunction
            _generate!(node.children[i], pset, depth+1, min_depth, max_depth, terminal_condition)
        end
    end
end

cond_full(pset, depth, min_depth, max_depth) = depth == max_depth
cond_grow(pset, depth, min_depth, max_depth) = depth == max_depth || (depth >= min_depth && rand() < pset.terminal_ratio)

generate_full(pset, min_depth, max_depth) = generate(pset, max_depth, max_depth, cond_full)
generate_grow(pset, min_depth, max_depth) = generate(pset, min_depth, max_depth, cond_grow)

function ramped_half_and_half(cfg, data_train, data_test)
    entries = Vector{Entry}(undef, cfg.population_size)

    index = 1
    for i in 1:cfg.population_size÷2
        tree = generate_full(cfg.pset, cfg.min_depth, cfg.max_depth)
        entries[i] = Entry(tree, data_train, data_test, cfg, index=index)
        index += 1
    end

    for i in cfg.population_size÷2+1:cfg.population_size
        tree = generate_grow(cfg.pset, cfg.min_depth, cfg.max_depth)
        entries[i] = Entry(tree, data_train, data_test, cfg, index=index)
        index += 1
    end

    return entries
end

function evaluate(node::GPNode{PrimitiveFunction{T}}, vars) where T
    if node.primitive.arity == 2
        arg1 = evaluate(node.children[1], vars)
        arg2 = evaluate(node.children[2], vars)
        return node.primitive.func(arg1, arg2)
    elseif node.primitive.arity == 1
        arg = evaluate(node.children[1], vars)
        return node.primitive.func(arg)
    else
        error("functions with arity > 2 are not supported arity")
    end
end
evaluate(node::GPNode{Const{T}}, vars) where T <: Real =  node.primitive.value
evaluate(node::GPNode{Variable{T}}, vars) where T = vars[node.primitive.name]


# Interface to AbstractTrees.jl
function AbstractTrees.children(node::GPNode)
    if node.primitive isa PrimitiveFunction
        return node.children
    end
    return ()
end

function AbstractTrees.printnode(io::IO, node::GPNode)
    prim = node.primitive
    if prim isa PrimitiveFunction
        print(io, prim.func)
    elseif prim isa Variable
        print(io,prim.name)
    else
        print(io, prim.value)
    end
end
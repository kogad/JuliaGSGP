import Base.isempty

struct Record
    names::Vector{String}
    functions::Dict{String, Function}
    data::Dict{String, AbstractVector}

    Record() = new(String[], Dict(), Dict())
end

Base.isempty(rec::Record) = isempty(rec.names)

function register!(rec, name, func, data_type)
    push!(rec.names, name)
    rec.functions[name] = func
    rec.data[name] = data_type[]
end

function add_record!(rec, g, pop, cfg)
    for name in rec.names
        push!(rec.data[name], rec.functions[name](g, pop, cfg))
    end
end

train_fitness(g, pop, cfg) = @views minimum(getfield.(entry_pool[pop], :train_fitness))
test_fitness(g, pop, cfg) = @views minimum(getfield.(entry_pool[pop], :test_fitness))
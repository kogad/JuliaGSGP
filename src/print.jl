
function show_op(o::Operator)
    o == crossover && return 'c'
    o == mutation && return 'm'
    return '-'
end

function show_ref(e::Entry)
    e.index in keys(ref_counter) && return ref_counter[e.index]
    return -1
end

function Base.show(io::IO, e::Entry)
    println(io, "  ID Index  Op    Entry(id)         Entry(index)    ref")
    @printf(io, "%4d  %4d   %c ", e.id, e.index, show_op(e.operator))
    @printf(io, "(%4d %4d %4d)   ",  e.id_entry[1], e.id_entry[2], e.id_entry[3])
    @printf(io, "(%4d %4d %4d)", e.index_entry[1], e.index_entry[2], e.index_entry[3])
    @printf(io, "  %3d", show_ref(e))
end

function Base.show(io::IO, x::MIME"text/plain", ve::Vector{Entry})
    println(io, "  ID Index  Op    Entry(id)         Entry(index)    ref")
    for (i, e) in enumerate(ve)
        @printf(io, "%4d  %4d   %c ", e.id, e.index, show_op(e.operator))
        @printf(io, "(%4d %4d %4d)   ", e.id_entry[1], e.id_entry[2], e.id_entry[3])
        @printf(io, "(%4d %4d %4d)", e.index_entry[1], e.index_entry[2], e.index_entry[3])
        @printf(io, "   %3d", show_ref(e))
        i  == length(ve) || println(io)
    end
end

function print_latest_fitness(rec, g)
    if g == 1
        println("     train_fitness  test_fitness")
    end
    @printf("%4d %10.4f     %10.4f\n", g, rec.data["train_fitness"][end], rec.data["test_fitness"][end])
end
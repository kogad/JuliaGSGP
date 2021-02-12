function mse(semantics::Vector{T}, data) where T
    res = 0
    @simd for i in 1:length(semantics)
        res += (data.target[i] - semantics[i])^2
    end
    res /= length(data.target)
end

function mse(tree, data)
    res = 0
    for i in 1:length(data.target)
        output = evaluate(tree, view(data.input, :, i))
        res += (data.target[i] - output)^2
    end
    res /= length(data.target)
end
    
rmse(x, data) = sqrt(mse(x, data))

function se(tree, data)
    res = 0
    for i in 1:length(data.target)
        output = evaluate(tree, view(data.input, :, i))
        res += (data.target[i] - output)^2
    end
    res
end

function se(semantics::Vector{T}, data) where T
    res = 0
    @simd for i in 1:length(semantics)
        res += (data.target[i] - semantics[i])^2
    end
    res
end
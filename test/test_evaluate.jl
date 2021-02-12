@testset "evaluate" begin

    @test evaluate(GPNode(Const(1)), []) == 1
    @test evaluate(GPNode(Const(1.25)), []) == 1.25

    @test evaluate(GPNode(Variable(1)), [3.14]) == 3.14
    @test evaluate(GPNode(Variable(1)), [3.14]) == 3.14


    add = PrimitiveFunction(+, 2)
    mul = PrimitiveFunction(*, 2)
    sub = PrimitiveFunction(-, 2)

    t = GPNode(mul, GPNode[
        GPNode(sub, GPNode[
            GPNode(Const(10)),
            GPNode(Variable(1))
        ]),
        GPNode(add, GPNode[
            GPNode(Variable(1))
            GPNode(Variable(2))
        ])
    ])
    t_func(vars) = (10 - vars[1]) * (vars[1] + vars[2])

    @test evaluate(t, [0, 1]) == t_func([0, 1])
    @test evaluate(t, [123, 456]) == t_func([123, 456])

    @test evaluate(t, [0., 1.]) == t_func([0., 1.])


end
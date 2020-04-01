using MyExample
using Test

using LinearAlgebra: Eigen
import LinearAlgebra: eigen, \, det, logdet, inv


#test

@testset "MyExample.jl" begin
    # Write your own tests here.

    @test NSKRRegressor(randn(4,4), [2,2], "zeros").λ_ == [2,2]

    @test NSKRRegressor([2,2], "zeros").λ_ == [2,2]

    @test length(Mps(randn(2,2), randn(4,4)).matrices) == 2

    model = NSKRRegressor([2,2], "zeros")
    fit(model, eigen.([randn(2,2), randn(5,5)]), randn(2,5), "traininds")



end

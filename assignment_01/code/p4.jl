using LinearAlgebra
x1 = [38.2, 27.0,7.4,7.4,1.6,0.7,3.5]
x2 = [36.7, 27.0, 7.1, 4.2, 1.1, 1.1, 3.1]

println("L1 norm")
println(norm(x1-x2,1))
println("")

println("L2 norm")
println(norm(x1-x2,2))
println("")

println("L10 norm")
println(norm(x1-x2,10))
println("")

println("L100 norm")
println(norm(x1-x2,100))

c = [5.,5.,2.,2.,0.5,0.1,1.0]
k = 5


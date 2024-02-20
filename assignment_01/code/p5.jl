using LinearAlgebra
using Plots
plotly()

F1 = [2.,5.,1.4, 4.7,1.7,4.4]
F2 = [5.,8.,4.4,7.7,4.7,7.4]
F3 = [6.,9.,5.4,8.7,5.7,8.4]
F1A = [F1[i] for i in [1,3]]
F2A = [F2[i] for i in [1,3]]
F3A = [F3[i] for i in [1,3]]
F1B = [F1[i] for i in [2,4,5,6]]
F2B = [F2[i] for i in [2,4,5,6]]
F3B = [F3[i] for i in [2,4,5,6]]


# plot points
scatter3d(F1A,F2A,F3A,name="A",label="A", xlabel = "F1", ylabel = "F2", zlabel = "F3") #,size =(5,8,9))
scatter3d(F1A,F2A,F3A,name="A",label="A", xlabel = "F1", ylabel = "F2", zlabel = "F3") #,size =(5,8,9))
scatter3d!(F1B,F2B,F3B,name="B",label="B", show=true)
savefig("plot.html")

# plot projection
scatter3d([F1A;F1A],[F2A;F2A],[F3A;zeros(length(F3A))],name="A",label="A", xlabel = "F1", ylabel = "F2", zlabel = "F3") #,size =(5,8,9))
scatter3d!([F1B;F1B],[F2B;F2B],[F3B;zeros(length(F3B))],name="B",label="B")
savefig("plotproj.html")
#scatter3d!(F1A,F2A,zeros(length(F1A)),name="A",label="A")
#scatter3d!(F1B,F2B,zeros(length(F1B)),name="B",label="B")

# x1 = [38.2 27.0 7.4 7.4 1.6 0.7 3.5]
# x2 = [36.7 27.0 7.1 4.2 1.1 1.1 3.1]
# c = [5.0 5.0 2.0 2.0 0.5 0.1 1.0]

# for p in [1 2 10 100]
#   println(p)
#   println(norm(x1,p))
#   println(norm(x1+c,p))
# end

# ps = [1 2 10 100]

# k = 5
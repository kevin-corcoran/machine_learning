using PlotlyJS, LinearAlgebra #, Colors


function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end

# Linear discriminant function
z(x,y) = (18 .- 2*y .- 3*x)./4
xs = collect(0:(5/0.01))*0.01
ys = collect(0:(5/0.01))*0.01
X, Y = meshgrid(xs,ys)
# Z1 = z(X,Y)

# Points
# positive
P = [2 2 3; 
     3 3 2;
     1 2 3;
     1 4 1;
     4 4 4;
     2 2 2]
# negatives
N = [3 3 1;
     1 1 1;
     3 3 2;
     0 4 2;
     4 0 0;
     0 0 3]

layout = Layout(
    scene=attr(
        xaxis=attr(
            range=[0,5]
        ),
        yaxis=attr(
            range=[0,5]
        ),
        zaxis=attr(
            range=[0,5]
        ),
    ),
    legend=attr(
        yanchor="top",
        xanchor="right",
        orientation="v"
    )
)

# const_color = [ RGB{Float64}(0.5,1,0.5) for _ in 1:2 ]
trace1 = surface(
        x=X,
        y=Y,
        z=z(X,Y),
        showscale=false,
        # colorscale=["blue"]
        # surfacecolor="blue"
        # surfacecolor = RGB{Float64}(0.5,1,0.5)
    )

trace2 = scatter3d(
        x=P[:, 1], 
        y=P[:,2], 
        z=P[:,3],
        mode = "markers",
        color = "green",
        name = "Positive"
    )

trace3 = scatter3d(
        x=N[:, 1], 
        y=N[:,2], 
        z=N[:,3],
        mode = "markers",
        color = "blue",
        name = "Negative"
    )

trace4 = scatter3d(
    x = [0, 3],
    y = [0, 2],
    z = [0, 4]
)

p = plot([trace1, trace2, trace3, trace4], layout)
savefig(p, "disc.html")

# normal to plane
w = [3;2;4]

# n̂ = n/norm(n,2)

t = 18
mp = []
for i in 1:size(P)[1] # num of rows
    margin = (dot(w,P[i,:]') - t)/norm(w,2)
    append!(mp, margin)
    # incorrectly classified
    if margin < 0
        println(P[i,:])
    end
end

t = 18
mn = []
for i in 1:size(N)[1] # num of rows
    margin = -(dot(w,N[i,:]') - t)/norm(w,2)
    append!(mn, margin)
    # incorrectly classified
    if margin < 0
        println(N[i,:])
    end
end


Pt = [(P[row, :]) for row in 1:size(P, 1)]
Nt = [(N[row, :]) for row in 1:size(N, 1)]
pts = vcat(Pt, Nt)

# a) margin for all points
zi = vcat(mp,mn)

# display
marg = [t for t in zip(pts,zi)]
println("a) Margin for points:\n")
display(marg)

# b) 0-1 loss return 1 if z <= 0 else return 0
function zeroone(z)
    return z .<= 0
end
# zeroone(zi)
println("b) 0-1 loss:\n")
display([t for t in zip(pts, zeroone(zi))])

# c) hinge-loss return 1-z if z <= 1 else return 0
function hinge(z)
    return [zi <= 1 ? 1-zi : 0 for zi in z]
end
# hinge(zi)
println("c) hinge-loss:\n")
display([t for t in zip(pts, hinge(zi))])

# d) squared loss
function squared(z)
    return [zi <= 1 ? (1-zi)^2 : 0 for zi in z]
end
# squared(zi)
println("d) squared-loss:\n")
display([t for t in zip(pts, squared(zi))])


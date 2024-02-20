function ProcessData(data)
    D = zeros( length(data), length(parse.(Float64,split(data[2]))) )
    for i = 1:length(data)
        row = parse.(Float64,split(data[i]))
        D[i,:] = row
    end
    return D
end

function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end

function Discriminant(centroid0, centroid1)
    diff = centroid0 .- centroid1
    t = sum(centroid0.^2 - centroid1.^2)/(2*diff[3])
    # normal to plane
    w = [diff[1]/diff[3], diff[2]/diff[3], 1]
    z0_1(x,y) = t .- (w[1]*x + w[2]*y)
    return z0_1, w, t
end

function PlotBoundary(x_range,y_range, z_range, Z, train_class0, train_class1, name0, name1)
    dx = dy =  0.01
    xs = collect(x_range[1]:x_range[2]/dx)*dx
    ys = collect(x_range[1]:y_range[2]/dy)*dy
    X, Y = meshgrid(xs, ys)

    layout = Layout(
        # scene=attr(
        #     xaxis=attr(
        #         range=[x_range[1],x_range[2]]
        #     ),
        #     yaxis=attr(
        #         range=[y_range[1],y_range[2]]
        #     ),
        #     zaxis=attr(
        #         range=[z_range[1],z_range[2]]
        #     ),
        # ),
        legend=attr(
            yanchor="top",
            xanchor="right",
            orientation="v"
        )
    )

    discriminant0_1 = surface(
            x=X,
            y=Y,
            z=Z(X,Y),
            showscale=false,
        )
    pts_class0 = scatter3d(
            x=train_class0[:,1], 
            y=train_class0[:,2], 
            z=train_class0[:,3],
            mode = "markers",
            color = "green",
            name = name0
        )
    pts_class1 = scatter3d(
            x = train_class1[:,1],
            y = train_class1[:,2],
            z = train_class1[:,3],
            mode = "markers",
            color = "red",
            name = name1
        )

    return plot([discriminant0_1, pts_class0, pts_class1], layout)

end

using Statistics, LinearAlgebra #, Plotly

function ProcessData(data)
    D = zeros( length(data), length(parse.(Float64,split(data[2]))) )
    for i = 1:length(data)
        row = parse.(Float64,split(data[i]))
        D[i,:] = row
    end
    return D
end

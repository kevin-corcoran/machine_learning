# module utils
## using Pkg
## Pkg.activate("utils")
## include("utils.jl")
## using .utils: ProcessData, ProcessKey

# export ProcessKey, ProcessData
function ProcessData(data)
    D = zeros( length(data), length(parse.(Float64,split(data[2]))) )
    for i = 2:length(data)
        row = parse.(Float64,split(data[i]))
        D[i,:] = row
    end
    return D[2:end, :]
end

function ProcessKey(key)
    keylist = split.(key, "=")
    feature_names = [replace(l[1], " " => "") for l in keylist]
    feature_num = 2:length(feature_names)+1
    features = collect(zip(feature_num, feature_names))

    lst = ["{", "}", " "]
    for i = 1:length(keylist)
        for s in lst
            keylist[i][2] = replace(keylist[i][2], s => "")
        end
    end

    vals = [split.(elem[2], ",") for elem in keylist]
    vals = [split.(elem, ":") for elem in vals]
    vals_ = []
    for v in vals
        append!(vals_, [[(v_[1], parse(Int64, v_[2])) for v_ in v]])
        # for v_ in v
        #     println(v_[1]," ", v_[2])
        # end
    end
    feature_values = Dict.(vals_)


    return Dict(zip(features, feature_values))
end
    
# end
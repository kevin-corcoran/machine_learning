using D3Trees
include("utils.jl")

data = readlines("../training.txt")
key_ = readlines("../Key.txt")
test1 = readlines("../test1.txt")
test2 = readlines("../test2.txt")
test3 = readlines("../test3.txt")

# Process Data
D = ProcessData(data)
test_data1 = ProcessData(test1)
test_data2 = ProcessData(test2)
test_data3 = ProcessData(test3)

# Create dictionary of key features with value dictionary of feature values
Key = ProcessKey(key_)
delete!(Key, (6, "GoodMovie"))
features = keys(Key)


# Begin GrowTree Algorithm
function entropy(P, N)
    # impurity measure minimum when p̂ = 0 or 1 and maximum when p̂ = 0.5
    # P: number of positives in data split
    # N: number of negatives in data split
    p̂ = P/(P+N) # proportion of positives
    return -p̂*log2(p̂) - (1-p̂)*log2(1-p̂)
end

function BestSplit(D, F)
    # F; Features Key
    # D; Data
    # Return literals that split the data with highest sum(impurity values) 
    # e.g. Budget => splits data by Low, Medium, High. Return D_low, D_med, D_high?
    features = keys(F)
    goodmovie = D[:,end] .== 1
    impurity = []
    return_feature_key = Vector{Tuple{Int64, String}}() # []
    for k1 in features
        feature_index = k1[1] # 2 for Budget
        feature_name = k1[2] # e.g "Budget"
        feature_value_names = keys(F[k1]) # Low, Medium, or High for Budget
        feature_values = values(F[k1]) # 0, 1, 2 for Budget
        # println(feature_value_names)
        # println(feature_values)
        positives = 0
        negatives = 0
        temp_impurity = []
        total_num = 0
        for k2 in feature_value_names
        value_name = k2 # Low, Medium, High
        value = Key[k1][k2] # 0, 1, 2

        # index of feature (e.g. "Budget") with value (e.g. 0 for name "Low")
        index_with_value   = findall(D[:, feature_index] .== value)
        with_value = D[:, feature_index] .== value
        goodmovie_with_value = [with_value[i] == 1 & goodmovie[i] == 1 for i in 1:length(with_value)]
        index_of_goodmovie_with_value  = findall(goodmovie_with_value)
        positives = length(index_of_goodmovie_with_value)
        negatives = length(index_with_value) - positives
        num_data = positives+negatives
        total_num += num_data
        append!(temp_impurity, num_data * entropy(positives, negatives))
        # println(temp_impurity)
        
        end
        # Append impurity with sum of features impurity
        append!(impurity, sum(temp_impurity)/total_num)
        push!(return_feature_key, k1)
    end
    if !isempty(impurity)
        min_impurity, index = findmin(impurity)
        return return_feature_key[index]
    else
        return ()
    end
end
S = BestSplit(D,Key)

function SplitData(D,F,S)
    # F; Features Key
    # D; Data
    # S: Feature (key) to split by
    if isempty(S)
        return []
    end
    goodmovie = D[:,end] .== 1
    feature_index = S[1]
    Di = []
    i = 1
    for (key, value) in F[S]
        # index of feature (e.g. "Budget") with value (e.g. 0 for name "Low")
        index_with_value   = findall(D[:, feature_index] .== value)
        append!(Di, [D[index_with_value, :]])
        # println("key: ", key," value: ", value)
        # println("index in Di ", i)
        i += 1
    end
    return Di
end
Di = SplitData(D, Key, S)

function Homogenous(D)
    return all(D[:, end] .== 1) || all(D[:, end] .== 0)
end

F = copy(Key)
# T_label = Vector{String}()
# T_label = []
node = []
child = []
DDl = []
DD = Dict()
function GrowTree(D, F, label, count)
    # F; Features
    # D; Data
    # returns; list length is number of nodes, split by feature, 
    # where the children are determined by BestSplit() using entropy impurity (a measure of positives and negatives)

    if Homogenous(D)
        println("Homogenous ", label)
        push!(DDl, (label, D))
        push!(DD, label => D)
        return label
    end
    S = BestSplit(D,F)
    # println(S)
    Di = SplitData(D, F, S)
    # println("L: ", label)
    # println("D: ", D[:, 1])
    # for (key, value) in F[S]
    i = 1
    if !isempty(S)
        # push!(T, S[2])
        for (key, value) in F[S]
            key = string(key)
            feature_index = S[1]
            index_with_value   = findall(Di[i][:, feature_index] .== value)
            # println(typeof(key), " ", key)
            if !isempty(Di[i])
                delete!(F, S)
                println("Best Split: ", S, ", child: ", key, ", Recursive call number: ", count, ", i: ", i)
                push!(node, (S,key))
                push!(DDl, (key, Di[i][index_with_value, :]))
                push!(DD, key => Di[i][index_with_value, :])
                count += 1
                push!(child,  GrowTree(Di[i], F, key, count))
                # push!(T_label, GrowTree(Di[i], F, key, count))
                # println("Root: ", S)
                # println("Children: ", T_label)
            else
                # push!(T_label, key)
                println("Di is empty ", key)
                # return key
                push!(child, key)
            end
            i += 1
            # index of feature (e.g. "Budget") with value (e.g. 0 for name "Low")
            # index_with_value   = findall(D[:, feature_index] .== value)
        end
        return (S, child)
        # delete!(F, S)
    else
        println("S is empty", ", label: ", label)
        push!(DDl, (label, D))
        push!(DD, label => D)
        return label
        # push!(child, label)
        # push!()
    end

    # push!(T, T_label)
    # println("T_label ", T_label)
    println("Does this print? ", label)

    return label
    # return (S, child)


        # println(key)
        # println(D[index_with_value, 1])
        # println(index_with_value)
        # with_value = D[:, feature_index] .== value
        # goodmovie_with_value = [with_value[i] == 1 & goodmovie[i] == 1 for i in 1:length(with_value)]
        # index_of_goodmovie_with_value  = findall(goodmovie_with_value)
end
TT = GrowTree(D, F, " ", 0)

# Didn't quite figure out how to make this tree programatically..
children = [[2,12,13], [3], [4,8,14], [5], [6,7], [], [], [9], [10,11], [], [], [], [], []]
text = ["Budget", "Low", "Genre", "Comedy", "Famous Actors", "Yes: 9 P", "No: 11 N", "Drama", "Director", "Great: 13 P", "Unknown: 10 N", "Medium: 3 P, 7 N, 8 N, 14 P, 15 P", "High: 4 P, 5 N, 6 P, 12 P, 16 P", "Documentary: 1 N, 2 N"]
tree = D3Tree(children, text = text)
inbrowser(tree, "firefox")

# Correct impurity measure
children = []
text = []
tree = D3Tree(children, text = text)
inbrowser(tree, "firefox")


# print leaves
leaves = ["Yes", "No", "Great", "Unknown", "Medium", "High", "Documentary"]
num = []

for l in leaves
    println(l)
    println(DD[l])
end

for (k,v) in DD
    println(k)
    println(v)
end

# for l in DDl
#     println(l[1])
#     println(l[2])
# end
function SplitData2(D,F,S)
    # F; Features Key
    # D; Data
    # S: Feature (key) to split by
    if isempty(S)
        return []
    end
    goodmovie = D[:,end] .== 1
    feature_index = S[1]
    Di = []
    for (key, value) in F[S]
        # index of feature (e.g. "Budget") with value (e.g. 0 for name "Low")
        index_with_value   = findall(D[:, feature_index] .== value)
        push!(Di, [(key, value), D[index_with_value, :]])
    end
    return Di
end
Di = SplitData2(D, Key, S)

# Function to build tree on training data
function LeafString(feature)
    return_str = ""
    for i in 1:size(feature,1)
        train_num = string(feature[i,1])
        train_num = replace(train_num, ".0" => "")
        train_val = string(feature[i, end])
        train_val = replace(train_val, "0.0" => "N")
        train_val = replace(train_val, "1.0" => "P")
        return_str *=  train_num * train_val * " "
    end
    return return_str
end

function BuildTree(D, F)
# D = test_data1
    split1 = SplitData2(D, F, (2, "Budget"))
    low = split1[1][2]
    med = split1[2][2] # leaf
    med_str = LeafString(med)
    high = split1[3][2] # leaf
    high_str = LeafString(high)
    # delete!(F, (2, "Budget"))
    split2 = SplitData2(low, F, (3, "Genre"))
    comedy = split2[1][2]
    documentary = split2[2][2] # leaf
    doc_str = LeafString(documentary)
    drama = split2[3][2]
    # delete!(F, (3, "Genre"))
    split3 = SplitData2(comedy, F, (4, "FamousActors"))
    yes = split3[1][2] # leaf
    yes_str = LeafString(yes)
    no = split3[2][2] # leaf
    no_str = LeafString(no)

    split4 = SplitData2(drama, F, (5, "Director"))
    great = split4[1][2] # leaf
    great_str = LeafString(great)
    unknown = split4[2][2] # leaf
    unk_str = LeafString(unknown)

    children = [[2,12,13], [3], [4,8,14], [5], [6,7], [], [], [9], [10,11], [], [], [], [], []]
    text = ["Budget", "Low", "Genre", "Comedy", "Famous Actors", "Yes: $yes_str", "No: $no_str", "Drama", "Director", "Great: $great_str", "Unknown: $unk_str", "Medium: $med_str", "High: $high_str", "Documentary: $doc_str"]
    return D3Tree(children, text = text)
end

F = copy(Key)
tree = BuildTree(test_data1, F)
inbrowser(tree, "firefox")
tree = BuildTree(test_data2, F)
inbrowser(tree, "firefox")
tree = BuildTree(test_data3, F)
inbrowser(tree, "firefox")

# Tree = []
# F = copy(Key)
# leaves = ["Yes", "No", "Great", "Unknown", "Medium", "High", "Documentary"]
# splits = [(2, "Budget"), "Low", (3, "Genre"), "Comedy", (4, "FamousActors"), "Drama", (5, "Director")]
# D = test_data1
# for i in 1:length(splits)
#     d = []
#     split = 0
#     next_split = 0
#     if typeof(splits[i]) == Tuple{Int64, String}
#         split = SplitData2(D, F, splits[i])
#         # push![d, split]
#         for s in split
#             if s[1][1] in splits
#                 next_split = s[2]
#             end
#         end
#     # else
#     #     split = SplitData2(next_split, F, splits[i+1])
#     #     # push![d, split]
#     end
#     for s in split
#         if s[1][1] in leaves
#             println(s[1])
#         end
#     end
# end
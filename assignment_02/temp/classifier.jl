using Pkg
Pkg.activate("TrainClassifier")
include("TrainClassifier.jl")
using .TrainClassifier: ts, ws

using Statistics, LinearAlgebra #, Plotly
include("utils.jl")

test1_data_dir = readlines("../data/testing1.txt")
test1_label_dir = readlines("../data/testing1_label.txt")

test1_data = ProcessData(test1_data_dir)
test1_labels = ProcessData(test1_label_dir)
prediction = zeros(size(test1_data, 1), 1)

# Classify points
function Classify(w, t, data, prediction, label0, label1, sgn)
    m0 = zeros(1,3)
    i0 = []
    m1 = zeros(1,3)
    i1 = []
    for i in 1:size(data, 1) # num of rows
        margin = sgn*(dot(w,data[i,:]') - t)/norm(w,2)
        # incorrectly classified
        # for class 0 and class 1 sgn = -1
        if margin <= 0.0 # belongs to class 0
            m0 = vcat(m0, data[i,:][:,:]')
            push!(i0, i) # save indices
            prediction[i,1] = label0
        else # belongs to class 1
            m1 = vcat(m1, data[i,:][:,:]')
            push!(i1, i) # index of point
            prediction[i,1] = label1
        end
    end

    return (m0[2:end, :], i0), (m1[2:end, :], i1)
end

# Split data on either side of decision boudary for class 0 and class 1
(class0, indices0), (class1, indices1) = Classify(ws[1], ts[1], test1_data, prediction, 0, 1, -1)
pred_class1 = prediction[indices1,1]

# find class 2 in class labeled 1
(temp1, tempi1), (class2, tempi2) = Classify(ws[2], ts[2], class1, pred_class1, 1, 2, -1)
# update prediction
prediction[indices1, 1] = pred_class1

# find class 2 in class labeled 0
pred_class0 = prediction[indices0, 1]
(t2, t2i), (t0, t0i) = Classify(ws[3], ts[3], class0, prediction, 2, 0, -1)
# update prediction
prediction[indices0, 1] = pred_class0

total_predicted_correct = sum(test1_labels .== prediction)

prediction = prediction[:,1]
# m0 = Classify(w0_1, t0_1, train_class0, 1)
# m1 = Classify(w0_1, t0_1, train_class1, -1)
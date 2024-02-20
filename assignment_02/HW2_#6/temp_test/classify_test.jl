include("utils_test.jl")
include("trainclassify.jl")
# Train classifier
train_input_dir = "data/training1.txt"
train_label_dir = "data/training1_label.txt"
(ws,ts) = Train(train_input_dir, train_label_dir)
# pred = Classifier(test_input_dir, w, t)

# Test classifier
test_input_dir = "data/testing1.txt"
test_label_dir = "data/testing1_label.txt"

test1_data_dir  = readlines(test_input_dir)
# test1_label_dir = readlines(test_label_dir)

test1_data = ProcessData(test1_data_dir)
# test1_labels = ProcessData(test1_label_dir)
prediction = zeros(size(test1_data, 1), 1)

data = test1_data
w = ws[1]; t = ts[1]
sgn = -1
label0 = 0
label1 = 1

# Classify
# Classify(w, t, data, prediction, label0, label1, sgn)
# (class0, indices0), (class1, indices1) = Classify(ws[1], ts[1], test1_data, prediction, 0, 1, -1)
# pred_class1 = prediction[indices1,1]
m0 = zeros(1,3)
i0 = []
# m1 = []
m1 = zeros(1,3)
i1 = []
for i in 1:size(data, 1) # num of rows
    global m0, m1, i1, i0
    margin = sgn*(dot(w,data[i,:]') - t)/norm(w,2)
    # push!(m0, margin)
    # incorrectly classified
    # for class 0 and class 1 sgn = +1
    if margin <= 0.0 # belongs to class 0
        # push!(m0, data[i,:])
        m0 = vcat(m0, data[i,:][:,:]')
        push!(i0, i) # save indices
        prediction[i,1] = label0
    else # belongs to class 1
        # push!(m1, data[i,:])
        m1 = vcat(m1, data[i,:][:,:]')
        push!(i1, i) # index of point
        prediction[i,1] = label1
    end
end
(class0, indices0), (class1, indices1) = (m0[2:end, :], i0), (m1[2:end, :], i1)


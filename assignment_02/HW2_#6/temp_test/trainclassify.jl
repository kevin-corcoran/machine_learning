# using Pkg
# Pkg.activate("TrainClassifier")
# include("TrainClassifier.jl")
# using .TrainClassifier: ts, ws

using Statistics, LinearAlgebra #, Plotly
# include("utils.jl")

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

    # export ts, ws
function Train(train_input_dir, train_label_dir)
    train_data_dir = readlines(train_input_dir)
    train_label_dir = readlines(train_label_dir)
    # train_data_dir = readlines("../data/training1.txt")
    # train_label_dir = readlines("../data/training1_label.txt")
    # train2_data_dir = readlines("../data/training2.txt")
    # train2_label_dir = readlines("../data/training2_label.txt")

    train_data = ProcessData(train_data_dir)
    # train2_data = ProcessData(train2_data_dir)
    # max_x, max_index_x = findmax(train_data[:, 1])
    # min_x, min_index_x = findmin(train_data[:, 1])
    # max_y, max_index_y = findmax(train_data[:, 2])
    # min_y, min_index_y = findmin(train_data[:, 2])
    # max_z, max_index_z = findmax(train_data[:, 3])
    # min_z, min_index_z = findmin(train_data[:, 3])

    train_labels = ProcessData(train_label_dir)
    # train2_labels = ProcessData(train2_label_dir)

    index_class0 = findall(train_labels[:,1] .== 0.0)
    index_class1 = findall(train_labels[:,1] .== 1.0)
    index_class2 = findall(train_labels[:,1] .== 2.0)

    # index2_class0 = findall(train2_labels[:,1] .== 0.0)
    # index2_class1 = findall(train2_labels[:,1] .== 1.0)
    # index2_class2 = findall(train2_labels[:,1] .== 2.0)

    train_class0 = train_data[index_class0, :]
    train_class1 = train_data[index_class1, :]
    train_class2 = train_data[index_class2, :]

    # train2_class0 = train2_data[index2_class0, :]
    # train2_class1 = train2_data[index2_class1, :]
    # train2_class2 = train2_data[index2_class2, :]

    centroid0 = mean(train_class0, dims=1)
    centroid1 = mean(train_class1, dims=1)
    centroid2 = mean(train_class2, dims=1)

    # centroid0_2 = mean(train2_class0, dims=1)
    # centroid1_2 = mean(train2_class1, dims=1)
    # centroid2_2 = mean(train2_class2, dims=1)

    centroids = [centroid0, centroid1, centroid2]

    # Discriminant function between class 0 and class 1

    z0_1, w0_1, t0_1 = Discriminant(centroid0, centroid1)
    # z0_1_2, w0_1_2, t0_1_2 = Discriminant(centroid0_2, centroid1_2)
    # p0_1 = PlotBoundary((min_x, max_x), (min_y, max_y), (min_z, max_z), z0_1, train_class0, train_class1, "Class 0", "Class 1")

    # Discriminant function between class 1 and class 2
    z1_2, w1_2, t1_2 = Discriminant(centroid1, centroid2)
    # z1_2_2, w1_2_2, t1_2_2 = Discriminant(centroid1_2, centroid2_2)
    # p1_2 = PlotBoundary((min_x, max_x), (min_y, max_y), (min_z, max_z), z1_2, train_class1, train_class2, "Class 1", "Class 2")

    # Discriminant function between class 2 and class 0
    z2_0, w2_0, t2_0 = Discriminant(centroid2, centroid0)
    # z2_0_2, w2_0_2, t2_0_2 = Discriminant(centroid2_2, centroid0_2)
    # p2_0 = PlotBoundary((min_x, max_x), (min_y, max_y), (min_z, max_z), z2_0, train_class2, train_class0, "Class 2", "Class 0")


    ts1 = [t0_1, t1_2, t2_0]
    ws1 = [w0_1, w1_2, w2_0]

    # ts2 = [t0_1_2, t1_2_2, t2_0_2]
    # ws2 = [w0_1_2, w1_2_2, w2_0_2]

    # ts = (ts1+ts2)/2 # average
    # ws = [(w[1] + w[2])/2 for w in zip(ws1, ws2)]
    return (ws1, ts1)

end

# Classify points
function Classify(w, t, data, prediction, label0, label1, sgn)
    # m0 = []
    m0 = zeros(1,3)
    i0 = []
    # m1 = []
    m1 = zeros(1,3)
    i1 = []
    for i in 1:size(data, 1) # num of rows
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

    return (m0[2:end, :], i0), (m1[2:end, :], i1)
    # if size(m1, 1) > size(m2, 1)
    #     return m1
    # else
    #     return m2
    # end
end

function Classifier(test_input_dir, ws, ts, pred_file = "result")
    # test1_data_dir = readlines("../data/testing1.txt")
    # test1_label_dir = readlines("../data/testing1_label.txt")

    test1_data_dir  = readlines(test_input_dir)
    # test1_label_dir = readlines(train_label_dir)

    test1_data = ProcessData(test1_data_dir)
    # test1_labels = ProcessData(test1_label_dir)
    prediction = zeros(size(test1_data, 1), 1)


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

    # total_predicted_correct = sum(test1_labels .== prediction)

    prediction = prediction[:,1]

    return prediction
end
# m0 = Classify(w0_1, t0_1, train_class0, 1)
# m1 = Classify(w0_1, t0_1, train_class1, -1)

# train_input_dir = "../data/training1.txt"
# train_label_dir = "../data/training1_label.txt"
# test_input_dir = "../data/testing1.txt"
# (w,t) = Train(train_input_dir, train_label_dir)
# pred = Classifier(test_input_dir, w, t)

# test1_label_dir = readlines("../data/testing1_label.txt")
# test1_labels = ProcessData(test1_label_dir)
# sum(pred .== test1_labels)
# Classify points
function Classify(w, t, data, prediction, label0, label1, sgn)
    # m0 = []
    m0 = zeros(1,3)
    i0 = []
    # m1 = []
    m1 = zeros(1,3)
    i1 = []
    for i in 1:size(data, 1) # num of rows
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

    return (m0[2:end, :], i0), (m1[2:end, :], i1)
    # if size(m1, 1) > size(m2, 1)
    #     return m1
    # else
    #     return m2
    # end
end

function Classifier(test_input_dir, ws, ts, pred_file = "result")
    # test1_data_dir = readlines("../data/testing1.txt")
    # test1_label_dir = readlines("../data/testing1_label.txt")

    test1_data_dir  = readlines(test_input_dir)
    # test1_label_dir = readlines(train_label_dir)

    test1_data = ProcessData(test1_data_dir)
    # test1_labels = ProcessData(test1_label_dir)
    prediction = zeros(size(test1_data, 1), 1)


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

    # total_predicted_correct = sum(test1_labels .== prediction)

    prediction = prediction[:,1]

    return prediction
end
# m0 = Classify(w0_1, t0_1, train_class0, 1)
# m1 = Classify(w0_1, t0_1, train_class1, -1)

# train_input_dir = "../data/training1.txt"
# train_label_dir = "../data/training1_label.txt"
# test_input_dir = "../data/testing1.txt"
# (w,t) = Train(train_input_dir, train_label_dir)
# pred = Classifier(test_input_dir, w, t)

# test1_label_dir = readlines("../data/testing1_label.txt")
# test1_labels = ProcessData(test1_label_dir)
# sum(pred .== test1_labels)

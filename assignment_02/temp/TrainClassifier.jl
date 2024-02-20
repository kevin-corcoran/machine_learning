module TrainClassifier
    using Statistics, LinearAlgebra #, Plotly
    include("utils.jl")
    include("trainclassify.jl")

    export ts, ws

    train_data_dir = readlines("../data/training1.txt")
    train_label_dir = readlines("../data/training1_label.txt")
    train2_data_dir = readlines("../data/training2.txt")
    train2_label_dir = readlines("../data/training2_label.txt")

    train_data = ProcessData(train_data_dir)
    train2_data = ProcessData(train2_data_dir)
    max_x, max_index_x = findmax(train_data[:, 1])
    min_x, min_index_x = findmin(train_data[:, 1])
    max_y, max_index_y = findmax(train_data[:, 2])
    min_y, min_index_y = findmin(train_data[:, 2])
    max_z, max_index_z = findmax(train_data[:, 3])
    min_z, min_index_z = findmin(train_data[:, 3])

    train_labels = ProcessData(train_label_dir)
    train2_labels = ProcessData(train2_label_dir)

    index_class0 = findall(train_labels[:,1] .== 0.0)
    index_class1 = findall(train_labels[:,1] .== 1.0)
    index_class2 = findall(train_labels[:,1] .== 2.0)

    index2_class0 = findall(train2_labels[:,1] .== 0.0)
    index2_class1 = findall(train2_labels[:,1] .== 1.0)
    index2_class2 = findall(train2_labels[:,1] .== 2.0)

    train_class0 = train_data[index_class0, :]
    train_class1 = train_data[index_class1, :]
    train_class2 = train_data[index_class2, :]

    train2_class0 = train2_data[index2_class0, :]
    train2_class1 = train2_data[index2_class1, :]
    train2_class2 = train2_data[index2_class2, :]

    centroid0 = mean(train_class0, dims=1)
    centroid1 = mean(train_class1, dims=1)
    centroid2 = mean(train_class2, dims=1)

    centroid0_2 = mean(train2_class0, dims=1)
    centroid1_2 = mean(train2_class1, dims=1)
    centroid2_2 = mean(train2_class2, dims=1)

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


    ts = [t0_1, t1_2, t2_0]
    ws = [w0_1, w1_2, w2_0]

    # ts2 = [t0_1_2, t1_2_2, t2_0_2]
    # ws2 = [w0_1_2, w1_2_2, w2_0_2]

    # ts = (ts1+ts2)/2 # average
    # ws = [(w[1] + w[2])/2 for w in zip(ws1, ws2)]
end
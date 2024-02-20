using Plots
plotly()

classifiers = [(0,0),(0,4)]
c1 = (9,16); c2 = (5,16); c3 = (3,13); c4 = (2,10); c5 = (0,4); c6 = (0,0)
scatter(c1,label="C1",xlabel = "FP",ylabel= "TP",title="Coverage Plot")
scatter!(c2, label="C2")
scatter!(c3, label="C3")
scatter!(c4, label="C4")
scatter!(c5, label="C5")
scatter!(c6, label="C6")
savefig("coverage_plot")


c1 = c1./(9,16); c2 = c2./(9,16); c3 = c3./(9,16); c4 = c4./(9,16); c5 = c5./(9,16); c6 = c6./(9,16)
scatter(c1,label="C1",xlabel = "FPR",ylabel= "TPR",title="ROC Plot")
scatter!(c2, label="C2")
scatter!(c3, label="C3")
scatter!(c4, label="C4")
scatter!(c5, label="C5")
scatter!(c6, label="C6")
savefig("ROC_plot")





scatter(classifiers,labels="A")

using Gadfly
X = [1, 2, 2, 3, 3, 3, 4]
Y = [4, 4, 7, 7, 9, 1, 8]
Labels = ["bill", "susan", "megan", "eric", "fran", "alex", "fred"]

plot(x=X, y=Y, label=Labels, Geom.point, Geom.label)
# 
# using PlotlyJS
# # x and y given as arrays
# scatter(classifiers, mode="markers")
# scatter(classifiers,name="B")
xs = [0,0]
ys = [0,4]

using PlotlyJS, Distributions

# Generate example data
x = rand(Uniform(3, 6), 500)
y = rand(Uniform(3, 4.5), 500)
x2 = rand(Uniform(3, 6), 500)
y2 = rand(Uniform(4.5, 6), 500)

# Add first scatter trace with medium sized markers
trace1 = scatter(
    mode="markers",
    x=x,
    y=y,
    opacity=0.5,
    marker=attr(
        color="LightSkyBlue",
        size=20,
        line=attr(
            color="MediumPurple",
            width=2
        )
    ),
    name="Opacity 0.5"
)

# Add second scatter trace with medium sized markers
# and opacity 1.0
trace2 = scatter(
    mode="markers",
    x=x2,
    y=y2,
    marker=attr(
        color="LightSkyBlue",
        size=20,
        line=attr(
            color="MediumPurple",
            width=2
        )
    ),
    name="Opacity 1.0"
)

# Add trace with large markers
trace3 = scatter(
    mode="markers",
    x=[2, 2],
    y=[4.25, 4.75],
    opacity=0.5,
    marker=attr(
        color="LightSkyBlue",
        size=80,
        line=attr(
            color="MediumPurple",
            width=8
        )
    ),
    showlegend=false
)

plot([trace1, trace2, trace3])
L = [3 1 2 2 3 5;
     5 5 3 3 5 3;
     5 3 5 3 5 2;
     2 5 5 5 2 1;
     3 5 1 3 5 2]

S = length(L)
# number of classes
# k = maximum(L)
k = 4
N1 = sum(x->x==1, L)
# prior probabilities with even distribution
π = [1/k for _ in 1:k]
class = [1;2;3;5]
# a) relative frequency
rel_freq = [sum(x->x==i, L)/S for i in class]
    # b) laplace correction
lap_corr = [(sum(x->x==i, L) + 1)/(S+k) for i in class]
# c) m estimate m=5
m = 5
m_est5 = [(sum(x->x==i, L) + m*π[1])/(S+m) for i in class]
# d) m estimate m=20 and even pseudocounts
m = 20
m_est20 = [(sum(x->x==i, L) + m*π[1])/(S+m) for i in class]

trace1 = scatter(
        x= 1:k, 
        y= rel_freq,
        name = "Relative Frequency"
    )
trace2 = scatter(
        x= 1:k, 
        y= lap_corr,
        name = "Laplace Correction"
    )
trace3 = scatter(
        x= 1:k, 
        y= m_est5,
        name = "m = 5"
    )
trace4 = scatter(
        x= 1:k, 
        y= m_est20,
        name = "m = 20"
    )
layout = Layout(
    xaxis = attr(
        tickmode = "array",
        tickvals = 1:k,
        ticktext = ["1", "2", "3", "5"]
    )
)
p = plot([trace1, trace2, trace3, trace4], layout)
savefig(p, "prob3.png")
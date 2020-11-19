using Pkg, Random, Plots, Distributions, LinearAlgebra, QuadGK, LaTeXStrings;
Pkg.develop(path="./BayesOpt");
using BayesOpt;
gp = GaussianProcess(ZeroMean(), Matern52(), 0.001);


n = 250;
x = reshape(LinRange(0, 50, n), 1, :);
g = X -> (0.25 * sin.(0.25 * X)) + (0.75 * cos.(X)) + 3 * cos.(pi * (X .- 25) / 50);
g_int, err = quadgk(g, 0, 50);
f = X -> g(X) .- (g_int / 50)

acEI   = ExpectedImprovement()
data = OptimizationData(ZeroMean(), Matern52(), 0.001, acEI, [0.0], [50.0])

for i in 1:10
	local p = ProposeAndEvaluateNextPoint!(data, x -> vec(f(x))[1])
	println("Proposed: $(p) Objective: $(f(p))")
end

gp2 = ConditionGP(GaussianProcess(data.mean, data.kernel, data.sigma), data.tX, data.tY);
std2 = Std(gp2, x);

p2 = plot(title="Bayesian Optimization Test");
plot!(p2, vec(x), Mean(gp2, x), ribbon=((1.96 * std2),(1.96 * std2)), linewidth=2, label=L"\mu");
scatter!(p2, vec(data.tX), data.tY, label="Observations");
plot!(p2, vec(x), vec(f(x)), label=L"f(x)", linecolor=:red);

display(plot(p2, size=(1600,900)))

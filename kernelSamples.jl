using Pkg, Random, Plots, Distributions, LinearAlgebra, LaTeXStrings;
Pkg.develop(path="./BayesOpt");
using BayesOpt;


n = 1000;
x = reshape(LinRange(0, 10, n), 1, :);
v = LinRange(0.05, 3.5, 100);

dist = MvNormal(Diagonal(ones(size(x)[2])));

sample = rand(dist);

anim = @animate for i in v
	K = Matern(nu=i);
	gp = GaussianProcess(ZeroMean(), K, 1e-16)
	C = cholesky(Cov(gp, x) + gp.sigma * I).L;
	plot(vec(x), vec(C * sample), xlims=(0,10), ylims=(-3,3), title="Matern sample", lab="nu = $(round(i, digits=3))", linecolor=[:red, :green, :blue][(trunc(Int, i) % 3) + 1]);
end
gif(anim, "maternSamples.gif", fps=10)

a = LinRange(0.01, 2.0, 100)
anim = @animate for i in a
	K = RationalQuadratic(alpha=i);
	gp = GaussianProcess(ZeroMean(), K, 1e-8)
	C = cholesky(Cov(gp, x) + gp.sigma * I).L;
	plot(vec(x), vec(C * sample), xlims=(0,10), ylims=(-3,3), title="Rational Quadratic sample", lab="nu = $(round(i, digits=3))", linecolor=[:red, :green, :blue][(trunc(Int, i) % 3) + 1]);
end
gif(anim, "rqSamples.gif", fps=10)

using Random, Plots, Distributions, LinearAlgebra, QuadGK, LaTeXStrings;
include("./GPs/GP.jl");
gp = GP.GaussianProcess(GP.ZeroMean(), GP.SquaredExponential(l=1.5), 0.01);

n = 250;
x = reshape(LinRange(0, 50, n), 1, :);
g = X -> (0.25 * sin.(0.25 * X)) + (0.75 * cos.(X)) + 3 * cos.(pi * (X .- 25) / 50);
g_int, err = quadgk(g, 0, 50);
f = X -> vec(g(X)) .- (g_int / 50)

ns = 20;
tx = reshape(LinRange(5, 45, ns), 1, :) + 3.0 * (rand(Float32, (1, ns)) .- 0.5)
ty = f(tx)
ydist = MvNormal(ty, 0.1^2 * I);
ty = vec(rand(ydist))

gp2 = GP.ConditionGP(gp, tx, ty);

dist = MvNormal(GP.Mean(gp, x), GP.Cov(gp, x) + gp.sigma * I);
dist2 = MvNormal(GP.Mean(gp2, x), GP.Cov(gp2, x) + gp2.sigma * I);

std = GP.Std(gp, x);
std2 = GP.Std(gp2, x);

# p = plot();
p2 = plot();
p3 = plot(title="Normalized Acquisition Functions");
# plot!(p, vec(x), GP.Mean(gp, x), ribbon=((1.96 * std),(1.96 * std)), linewidth=2, label=L"\mu");
plot!(p2, vec(x), GP.Mean(gp2, x), ribbon=((1.96 * std2),(1.96 * std2)), linewidth=2, label=L"\mu");
for i = 1:10
	# plot!(p, vec(x), vec(rand(dist)), linealpha=0.1, label="");
	plot!(p2, vec(x), vec(rand(dist2)), linealpha=0.1, label="");
end
scatter!(p2, vec(tx), ty, label="Observations");
plot!(p2, vec(x), f(x), label=L"f(x)", linecolor=:red);

acEI   = GP.ExpectedImprovement()
acKG   = GP.KnowledgeGradientCP()
acUCB  = GP.UpperConfidenceBound(beta=4.0)
acPI   = GP.ProbabilityOfImprovement(tau=0.1)
acMES  = GP.MutualInformationMES(gp2, [0.0], [50.0])
acOPES = GP.MutualInformationOPES(gp2, [0.0], [50.0])

acfns = [
	 ("EI", acEI),
	 ("KGCP", acKG),
	 ("UCB (B=4.0)", acUCB),
	 ("PoI (t=0.1)", acPI),
	 ("MES", acMES),
	 ("OPES", acOPES),
];

i = size(acfns, 1)
for (lab,fn) in acfns
	local ac = GP.Acquire(fn, gp2, x, tx, ty)
	ac = vec((1 / (size(acfns, 1) + 1)) * (ac .- minimum(ac)) ./ (maximum(ac) - minimum(ac)))
	hline!(p3, [(i / size(acfns, 1))], linestyle = :dot, linewidth=0.25, color=:black, label="");
	plot!(p3, vec(x), ac .+ (i  / size(acfns, 1)), ribbon=(ac, fill(0, size(ac))), label=lab, grid=false, yticks=false);
	global i -= 1
end

l = @layout [a b]
display(plot(p2, p3, layout=l, size=(1600,900)))
# println("log marginal likelihood: $(GP.LogMarginalLikelihood(gp2, tx, ty))")
# savefig("acquisitionFns.png")

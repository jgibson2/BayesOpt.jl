module EmmoryProblem

using Pkg, Random, Plots, Distributions, LinearAlgebra, SpecialFunctions, LaTeXStrings;
include("./BayesOpt/src/BayesOpt.jl")
using .BayesOpt;
# Pkg.develop(path="./BayesOpt");
# using BayesOpt;

n = 250;
x = reshape(LinRange(0, 1, n), 1, :)
f = x -> tan.(0.5 .* x) - cos.(13 .* x) .+ (x .+ 0.5).^3 .* erf.(sin.(x))
gp = GaussianProcess(ConstantMean(mean(f(x))), Matern52(), sqrt(1e-5));

acEI   = ExpectedImprovement()
acUCB0  = UpperConfidenceBound(beta=0.0)
acUCB1  = UpperConfidenceBound(beta=1.0)
acUCB2  = UpperConfidenceBound(beta=2.0)
acUCB4  = UpperConfidenceBound(beta=4.0)
acPI0   = ProbabilityOfImprovement(tau=0.0)
acPI1   = ProbabilityOfImprovement(tau=0.1)
acPI5   = ProbabilityOfImprovement(tau=0.5)
acPI10   = ProbabilityOfImprovement(tau=1.0)
acPI30   = ProbabilityOfImprovement(tau=3.0)

acfns = [
	 ("EI", acEI),
	 ("UCB (B=0.0)", acUCB0),
	 ("UCB (B=1.0)", acUCB1),
	 ("UCB (B=2.0)", acUCB2),
	 ("UCB (B=4.0)", acUCB4),
	 ("PoI (t=0.0)", acPI0),
	 ("PoI (t=0.1)", acPI1),
	 ("PoI (t=0.5)", acPI5),
	 ("PoI (t=1.0)", acPI10),
	 ("PoI (t=3.0)", acPI30),
];

#=
for (lab,fn) in acfns
	data = OptimizationData(gp.mean, gp.kernel, gp.sigma, fn, [0.0], [1.0]) 
	println("Method: $(lab)")
	anim = @animate for i in 1:20
		local pt = ProposeAndEvaluateNextPoint!(data, f)
		println("Proposed: $(pt) Objective: $(f(pt))")
		gpi = ConditionGP(GaussianProcess(data.mean, data.kernel, data.sigma), data.tX, data.tY);
		std = Std(gpi, x);
		
		p2 = plot(title="$(lab) (Observation $(i))", legend=:bottomright);
		plot!(p2, vec(x), Mean(gpi, x), ribbon=((1.96 * std),(1.96 * std)), linewidth=2, label=L"\mu");
		scatter!(p2, vec(data.tX), data.tY, label="Observations");
		plot!(p2, vec(x), vec(f(x)), label=L"f(x)", linecolor=:red);
	end

	gif(anim, "emmory_$(lab).gif", fps=2)
end

data = OptimizationData(gp.mean, gp.kernel, gp.sigma, ProbabilityOfImprovement(), [0.0], [1.0]) 
schedule = vec([0 0.0001 0.001 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.15 0.2 0.25 0.3 0.4 0.5 0.75 1 1.5 2 3]);
println("Method: PoI")
anim = @animate for i in 1:size(schedule, 1)
	data.acquisitionFunction = ProbabilityOfImprovement(tau=schedule[i]);
	local pt = ProposeAndEvaluateNextPoint!(data, f)
	println("Proposed: $(pt) Objective: $(f(pt))")
	gpi = ConditionGP(GaussianProcess(data.mean, data.kernel, data.sigma), data.tX, data.tY);
	std = Std(gpi, x);
	
	p2 = plot(title="PoI (α = $(schedule[i])) (Observation $(i))", legend=:bottomright);
	plot!(p2, vec(x), Mean(gpi, x), ribbon=((1.96 * std),(1.96 * std)), linewidth=2, label=L"\mu");
	scatter!(p2, vec(data.tX), data.tY, label="Observations");
	plot!(p2, vec(x), vec(f(x)), label=L"f(x)", linecolor=:red);
end
gif(anim, "emmory_PoI.gif", fps=2)
=#


data = OptimizationData(gp.mean, gp.kernel, gp.sigma, UpperConfidenceBound(), [0.0], [1.0]) 
println("Method: UCB")
anim = @animate for i in 1:25
	b = 2.0 * log((i+4)^2 * pi^2 / 60)
	data.acquisitionFunction = UpperConfidenceBound(beta=b);
	local pt = ProposeAndEvaluateNextPoint!(data, f)
	println("Proposed: $(pt) Objective: $(f(pt))")
	gpi = ConditionGP(GaussianProcess(data.mean, data.kernel, data.sigma), data.tX, data.tY);
	std = Std(gpi, x);
	
	p2 = plot(title="UCB (β = $(round(b; digits=3))) (Observation $(i))", legend=:bottomright);
	plot!(p2, vec(x), Mean(gpi, x), ribbon=((1.96 * std),(1.96 * std)), linewidth=2, label=L"\mu");
	scatter!(p2, vec(data.tX), data.tY, label="Observations");
	plot!(p2, vec(x), vec(f(x)), label=L"f(x)", linecolor=:red);
end
gif(anim, "emmory_UCB.gif", fps=2)

end

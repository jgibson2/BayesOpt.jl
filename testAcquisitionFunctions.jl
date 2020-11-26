module TestAcquisitionFunctions
using Pkg, Random, Plots, Distributions, LinearAlgebra, QuadGK, LaTeXStrings, Optim;
# Pkg.develop(path="./BayesOpt");
# using BayesOpt;

include("./BayesOpt/src/BayesOpt.jl")
using .BayesOpt;


gp = GaussianProcess(ZeroMean(), Matern52(), 1.0);

n = 250;
x = reshape(LinRange(0, 30, n), 1, :);

tx = reshape([
3.09309309309309,
3.63363363363363,	      
7.68768768768769,
8.58858858858859,
14.984984984985	,
18.8888888888889,
24.954954954955	,
28.1681681681682,
28.2582582582583,
28.2582582582583], 1, :)

ty = [-1.16739385336939,
1.18786558167983,
1.02278010479536,
-0.215092970757165,
1.50473523050663,
0.229681339802749,
-1.41095107805059,
0.0474316198194434,
3.47304866304729,
0.560705881263232]

gp2 = ConditionGP(gp, tx, ty);

dist2 = MvNormal(Mean(gp2, x), Cov(gp2, x));

std = Std(gp, x);
std2 = Std(gp2, x);

p2 = plot();
p3 = plot(title="Normalized Acquisition Functions");
plot!(p2, vec(x), Mean(gp2, x), ribbon=((1.96 * std2),(1.96 * std2)), linewidth=2, label=L"\mu");
for i = 1:10
	plot!(p2, vec(x), vec(rand(dist2)), linealpha=0.25, label="");
end
scatter!(p2, vec(tx), ty, label="Observations");

acEI   = ExpectedImprovement()
acKG   = KnowledgeGradientCP()
acUCB  = UpperConfidenceBound(beta=4.0)
acPI   = ProbabilityOfImprovement(tau=0.1)
acMES  = MutualInformationMES(gp2, [0.0], [30.0])
acOPES = MutualInformationOPES(gp2, [0.0], [30.0])
acMIBatchEI = MutualInformationPenalizedBatch(acEI)
acCVBatchEI = CovariancePenalizedBatch(acEI)
acLocalBatchEI = LocalPenalizedBatch(acEI)

batchSize = 5

dataMI = BatchOptimizationData(ZeroMean(), Matern52(), 1.0, acMIBatchEI, [0.0], [30.0], tx, ty) 
pointsMI = ProposeNextBatch(dataMI; restarts=5, batchSize=batchSize)

dataCV = BatchOptimizationData(ZeroMean(), Matern52(), 1.0, acMIBatchEI, [0.0], [30.0], tx, ty) 
pointsCV = ProposeNextBatch(dataCV; restarts=5, batchSize=batchSize)

dataL = BatchOptimizationData(ZeroMean(), Matern52(), 1.0, acLocalBatchEI, [0.0], [30.0], tx, ty) 
pointsL = ProposeNextBatch(dataL; restarts=5, batchSize=batchSize)

samplerATS = ATSSampleBatch(ZeroMean(), Matern52(), 1.0; bounds_length_scale=(0.5, 2.5))
sampledATS_xs, sampledATS_ys = AcquireBatch(samplerATS, gp, [0.0], [30.0], tx, ty; batchSize=20, return_ys=true);
scatter!(p2, vec(sampledATS_xs), sampledATS_ys, label="ATS Sampled Optima", markershape = :x, markersize = 3, color = :green);

samplerTS = ThompsonSampleBatch()
sampledTS_xs, sampledTS_ys = AcquireBatch(samplerTS, gp, [0.0], [30.0], tx, ty; batchSize=20, return_ys=true);
scatter!(p2, vec(sampledTS_xs), sampledTS_ys, label="TS Sampled Optima", markershape = :+, markersize = 3, color = :red);

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
	local ac = AcquisitionScore(fn, gp2, x, tx, ty)
	ac = vec((1 / (size(acfns, 1) + 1)) * (ac .- minimum(ac)) ./ (maximum(ac) - minimum(ac)))
	hline!(p3, [(i / size(acfns, 1))], linestyle = :dot, linewidth=0.25, color=:black, label="");
	plot!(p3, vec(x), ac .+ (i  / size(acfns, 1)), ribbon=(ac, fill(0, size(ac))), label=lab, grid=false, yticks=false);
	global i -= 1
end
vline!(p3, vec(pointsMI), linestyle = :dash, linewidth=0.75, color=:blue, label="MI EI Batch Locations");
vline!(p3, vec(pointsCV), linestyle = :dash, linewidth=0.75, color=:red, label="CV EI Batch Locations");
vline!(p3, vec(pointsL), linestyle = :dash, linewidth=0.75, color=:green, label="Local EI Batch Locations");

l = @layout [a b]
display(plot(p2, p3, layout=l, size=(1600,900)))
# println("log marginal likelihood: $(LogMarginalLikelihood(gp2, tx, ty))")
savefig("acquisitionFns.png")

end #module

using Optim, Distributions;

module BayesOpt

export
	ZeroMean,
	ConstantMean,
	FunctionMean,
	Kernel,
	SquaredExponential,
	RationalQuadratic,
	Matern,
	Matern12,
	Matern32,
	Matern52,
	SpectralMixture,
	Mean,
	Cov,
	Std,
	GaussianProcess,
	ConditionGP,
	LogMarginalLikelihood,
	AcquireScore,
	ExpectedImprovement,
	KnowledgeGradientCP,
	ProbabilityOfImprovement,
	UpperConfidenceBound,
	MutualInformationMES,
	MutualInformationOPES,
	MutualInformationPenalizedBatch,
	CovariancePenalizedBatch,
	ThompsonSampler,
	ATSSampler,
	SampleOptima,
	OptimizationData,
	ProposeNextPoint,
	ProposeAndEvaluateNextPoint!;

include("GP.jl")
include("AcquisitionFunctions.jl")
include("BatchAcquisitionFunctions.jl")

mutable struct OptimizationData
	mean::Union{ZeroMean,ConstantMean,FunctionMean}
	kernel::Union{Kernel,
		SquaredExponential,
		RationalQuadratic,
		Matern,
		Matern12,
		Matern32,
		Matern52,
		SpectralMixture}
	sigma::Float32
	acquisitionFunction::Union{ExpectedImprovement,
		KnowledgeGradientCP,
		ProbabilityOfImprovement,
		UpperConfidenceBound,
		MutualInformationMES,
		MutualInformationOPES,
		MutualInformationPenalizedBatch,
		CovariancePenalizedBatch}
	lbounds::AbstractVector{Float32}
	ubounds::AbstractVector{Float32}
	tX::Union{AbstractArray{Float32, 2}, Nothing}
	tY::Union{AbstractVector{Float32}, Nothing}
	OptimizationData(gp, ac, lbounds, ubounds) = new(gp.mean, gp.kernel, gp.sigma, ac, lbounds, ubounds, nothing, nothing)
	OptimizationData(m, k, s, ac, lbounds, ubounds) = new(m, k, s, ac, lbounds, ubounds, nothing, nothing)
	OptimizationData(m, k, s, ac, lbounds, ubounds, tX, tY) = new(m, k, s, ac, lbounds, ubounds, tX, tY)
end

function ProposeNextPoint(p::OptimizationData; restarts=20, batchSize=1)
	dist = Product(Uniform.(p.lbounds, p.ubounds))
	point = rand(dist, batchSize)
	if p.tX != nothing && p.tY != nothing
		gp = ConditionGP(GaussianProcess(p.mean, p.kernel, p.sigma), p.tX, p.tY)
		best_f = Inf;
		f = x -> -1 * sum(AcquireScore(p.acquisitionFunction, gp, reshape(x, (size(x, 1), batchSize)), p.tX, p.tY))
		for iter in 1:restarts
			initial_x = rand(dist, batchSize)
			result = optimize(
					  f,
					  hcat([p.lbounds for _ in 1:batchSize]...),
					  hcat([p.ubounds for _ in 1:batchSize]...),
					  initial_x,
					  Fminbox(LBFGS()),
					  Optim.Options(show_trace=false,
							 f_tol=1e-4,
							 x_tol=1e-4,
							 g_tol=1e-4,
							 iterations=50))
			if best_f > Optim.minimum(result)
				point = Optim.minimizer(result)
				best_f = Optim.minimum(result)
			end
		end
	end
	reshape(point, (size(point, 1), batchSize))
end

function ProposeAndEvaluateNextPoint!(p::OptimizationData, f; restarts=20, batchSize=1)
	point = ProposeNextPoint(p; restarts=restarts, batchSize=batchSize)
	y = f(point)
	p.tX = p.tY == nothing ? point : hcat(p.tX, point)
	p.tY = p.tY == nothing ? [y] : vcat(p.tY, [y])
	point
end

end # module

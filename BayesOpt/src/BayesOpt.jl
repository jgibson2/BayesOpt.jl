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
	AcquisitionScore,
	ExpectedImprovement,
	KnowledgeGradientCP,
	ProbabilityOfImprovement,
	UpperConfidenceBound,
	MutualInformationMES,
	MutualInformationOPES,
	AcquirePoint,
	MutualInformationPenalizedBatch,
	CovariancePenalizedBatch,
	LocalPenalizedBatch,
	ThompsonSampleBatch,
	ATSSampleBatch,
	AcquireBatch,
	OptimizationData,
	BatchOptimizationData,
	ProposeNextPoint,
	ProposeAndEvaluateNextPoint!,
	ProposeNextBatch,
	ProposeAndEvaluateNextBatch!;

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
		MutualInformationOPES}
	lbounds::AbstractVector{Float32}
	ubounds::AbstractVector{Float32}
	tX::Union{AbstractArray{Float32, 2}, Nothing}
	tY::Union{AbstractVector{Float32}, Nothing}
	OptimizationData(gp, ac, lbounds, ubounds) = new(gp.mean, gp.kernel, gp.sigma, ac, lbounds, ubounds, nothing, nothing)
	OptimizationData(m, k, s, ac, lbounds, ubounds) = new(m, k, s, ac, lbounds, ubounds, nothing, nothing)
	OptimizationData(m, k, s, ac, lbounds, ubounds, tX, tY) = new(m, k, s, ac, lbounds, ubounds, tX, tY)
end

mutable struct BatchOptimizationData
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
	batchAcquisitionFunction::Union{
		ThompsonSampleBatch,
		ATSSampleBatch,
		MutualInformationPenalizedBatch,
		CovariancePenalizedBatch,
		LocalPenalizedBatch}
	lbounds::AbstractVector{Float32}
	ubounds::AbstractVector{Float32}
	tX::Union{AbstractArray{Float32, 2}, Nothing}
	tY::Union{AbstractVector{Float32}, Nothing}
	BatchOptimizationData(gp, ac, lbounds, ubounds) = new(gp.mean, gp.kernel, gp.sigma, ac, lbounds, ubounds, nothing, nothing)
	BatchOptimizationData(m, k, s, ac, lbounds, ubounds) = new(m, k, s, ac, lbounds, ubounds, nothing, nothing)
	BatchOptimizationData(m, k, s, ac, lbounds, ubounds, tX, tY) = new(m, k, s, ac, lbounds, ubounds, tX, tY)
end

function ProposeNextPoint(data::OptimizationData; restarts=20)
	dist = Product(Uniform.(data.lbounds, data.ubounds))
	point = rand(dist, 1)
	if data.tX != nothing && data.tY != nothing
		gp = ConditionGP(GaussianProcess(data.mean, data.kernel, data.sigma), data.tX, data.tY)
		point = AcquirePoint(data.acquisitionFunction, gp, data.lbounds, data.ubounds, data.tX, data.tY; restarts=restarts)
	end
	reshape(point, (size(point, 1), 1))
end

function ProposeAndEvaluateNextPoint!(data::OptimizationData, f; restarts=20)
	point = ProposeNextPoint(data; restarts=restarts)
	y = f(point)
	data.tX = data.tX == nothing ? point : hcat(data.tX, point)
	data.tY = data.tY == nothing ? vec(y) : vcat(data.tY, vec(y))
	point
end

function ProposeNextBatch(data::BatchOptimizationData; restarts=20, batchSize=5)
	dist = Product(Uniform.(data.lbounds, data.ubounds))
	batch = rand(dist, batchSize)
	if data.tX != nothing && data.tY != nothing
		gp = ConditionGP(GaussianProcess(data.mean, data.kernel, data.sigma), data.tX, data.tY)
		batch = AcquireBatch(data.batchAcquisitionFunction, gp, data.lbounds, data.ubounds, data.tX, data.tY; restarts=restarts, batchSize=batchSize)
	end
	reshape(batch, (size(batch, 1), batchSize))
end

function ProposeAndEvaluateNextBatch!(data::BatchOptimizationData, f; restarts=20, batchSize=5)
	batch = ProposeNextBatch(data; restarts=restarts, batchSize=batchSize)
	y = f(batch)
	data.tX = data.tX == nothing ? batch : hcat(data.tX, batch)
	data.tY = data.tY == nothing ? vec(y) : vcat(data.tY, vec(y))
	batch
end

end # module

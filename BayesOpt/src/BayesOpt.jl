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
	Acquire,
	ExpectedImprovement,
	KnowledgeGradientCP,
	ProbabilityOfImprovement,
	UpperConfidenceBound,
	MutualInformationMES,
	MutualInformationOPES,
	OptimizationData,
	ProposeNextPoint,
	ProposeAndEvaluateNextPoint!;

include("Kernels.jl");
include("GP.jl")
include("AcquisitionFunctions.jl")

mutable struct OptimizationData
	mean::Union{ZeroMean,ConstantMean,FunctionMean}
	kernel::Union{Kernel,SquaredExponential,RationalQuadratic,Matern,Matern12,Matern32,Matern52,SpectralMixture}
	sigma::Float32
	acquisitionFunction::Union{ExpectedImprovement,KnowledgeGradientCP,ProbabilityOfImprovement,UpperConfidenceBound,MutualInformationMES,MutualInformationOPES}
	lbounds::AbstractVector{Float32}
	ubounds::AbstractVector{Float32}
	tX::Union{AbstractArray{Float32, 2}, Nothing}
	tY::Union{AbstractVector{Float32}, Nothing}
	OptimizationData(gp, ac, lbounds, ubounds) = new(gp.mean, gp.kernel, gp.sigma, ac, lbounds, ubounds, nothing, nothing)
	OptimizationData(m, k, s, ac, lbounds, ubounds) = new(m, k, s, ac, lbounds, ubounds, nothing, nothing)
	OptimizationData(m, k, s, lbounds, ubounds, tX, tY) = new(m, k, s, ac, lbounds, ubounds, tX, tY)
end

function ProposeNextPoint(p::OptimizationData; restarts=20)
	dist = Product(Uniform.(p.lbounds, p.ubounds))
	point = rand(dist)
	if p.tX != nothing && p.tY != nothing
		gp = ConditionGP(GaussianProcess(p.mean, p.kernel, p.sigma), p.tX, p.tY)
		best_f = Inf;
		f = x -> -1 * Acquire(p.acquisitionFunction, gp, reshape(x, (size(x, 1), 1)), p.tX, p.tY)[1]
		for iter in 1:restarts
			initial_x = rand(dist)
			result = optimize(f, p.lbounds, p.ubounds, initial_x, Fminbox(LBFGS()))
			if best_f > Optim.minimum(result)
				point = Optim.minimizer(result)
				best_f = Optim.minimum(result)
			end
		end
	end
	reshape(point, (size(point, 1), 1))
end

function ProposeAndEvaluateNextPoint!(p::OptimizationData, f; restarts=20)
	point = ProposeNextPoint(p; restarts=restarts)
	y = f(point)
	p.tX = p.tY == nothing ? point : hcat(p.tX, point)
	p.tY = p.tY == nothing ? [y] : vcat(p.tY, [y])
	point
end

end # module

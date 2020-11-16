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
	SpectralMixure,
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
	MutualInformationOPES;

include("GP.jl")
include("AcquisitionFunctions.jl")

end # module

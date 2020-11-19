using LinearAlgebra, SpecialFunctions;

function Std(gp, X)
	sqrt.([ K(gp.kernel,x,x) for x=eachcol(X) ])
end

function Cov(gp, X, X2)
	[ K(gp.kernel,x,x2) for x=eachcol(X), x2=eachcol(X2) ]
end

function Cov(gp, X)
	Cov(gp, X, X)
end

function Mean(gp, X)
	μ(gp.mean, X)
end

struct ZeroMean
end

struct ConstantMean
	c
end

struct FunctionMean
	f
end

function μ(m::ZeroMean, X)
	vec(zeros(size(X)[2]))
end

function μ(m::ConstantMean, X)
	vec(m.c .* ones(size(X)[2]))
end

function μ(m::FunctionMean, X)
	vec(mapslices(x -> m.f(reshape(x, size(x,1), 1)), X; dims=1))
end

function ConditionGP(gp, x, y)
	X = copy(x); Y = copy(y);
	L = cholesky(Cov(gp, X) + (gp.sigma.^2 * I)).L;
	alpha = transpose(L) \ (L \ (Y - Mean(gp, X)));
	
	mean = FunctionMean(x -> Mean(gp, x) + Cov(gp, x, X) * alpha)
	v1 = x1 -> vec(L \ Cov(gp, X, x1))
	v2 = x2 -> vec(L \ Cov(gp, X, x2))
	kernel = Kernel((x1, x2) -> K(gp.kernel, x1, x2) -  dot(v1(x1), v2(x2)))
	
	GaussianProcess(mean, kernel, gp.sigma)
end

function LogMarginalLikelihood(gp, x, y)
	S = Cov(gp, x) + (gp.sigma^2 * I)
	L = cholesky(S).L;
	v = vec(L \ (y - Mean(gp, x)));
	lml = (-1/2) * (dot(v,v) + log(det(L)^2) + size(x)[2] * log(2 * pi));
	lml
end

struct GaussianProcess
	mean
	kernel
	sigma
	GaussianProcess(mean, kernel, sigma) = new(mean, kernel, sigma)
	GaussianProcess(kernel, sigma) = new(ZeroMean(), kernel, sigma)
end


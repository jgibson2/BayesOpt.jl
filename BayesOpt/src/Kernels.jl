using LinearAlgebra, SpecialFunctions;

struct Kernel
	k
end

struct SquaredExponential
	l
	s
	SquaredExponential(;l=1.0, s=1.0) = new(l, s)
end

struct RationalQuadratic
	alpha
	l
	s
	RationalQuadratic(;alpha=1.0, l=1.0, s=1.0) = new(alpha, l, s)
end

struct Matern
	nu
	l
	s
	Matern(;nu=0.5, l=1.0, s=1.0) = new(nu, l, s)
end
	
struct Matern12
	l
	s
	Matern(;l=1.0, s=1.0) = new(l, s)
end

struct Matern32
	l
	s
	Matern(;l=1.0, s=1.0) = new(l, s)
end

struct Matern52
	l
	s
	Matern(;l=1.0, s=1.0) = new(l, s)
end

struct SpectralMixture
	weights
	means
	sigmas
end

function K(kernel::Kernel, x, x2)
	kernel.k(x, x2)
end

function K(kernel::SquaredExponential, x, x2)
	kernel.s^2 * exp(-norm(x-x2)^2/(2*kernel.l^2))
end

function K(kernel::RationalQuadratic, x, x2)
	kernel.s^2 * (1 + norm(x-x2)^2/(2*kernel.alpha*kernel.l^2))^(-kernel.alpha)
end

function K(kernel::Matern, x, x2)
	kernel.s^2 / (gamma(kernel.nu) * 2^(kernel.nu-1) + eps()) * (sqrt(2 * kernel.nu) * (norm(x-x2) + eps()) / kernel.l)^kernel.nu * besselk(kernel.nu, sqrt(2 * kernel.nu) * (norm(x-x2) + eps()) / kernel.l)
end

function K(kernel::Matern12, x, x2)
	kernel.s^2 * exp(-norm(x-x2) / kernel.l)
end

function K(kernel::Matern32, x, x2)
	kernel.s^2 * (1 + (sqrt(3) * norm(x-x2) / kernel.l)) * exp(-sqrt(3) * norm(x-x2) / kernel.l)
end

function K(kernel::Matern52, x, x2)
	kernel.s^2 * (1 + (sqrt(5) * norm(x-x2) / kernel.l) + (5 * norm(x-x2)^2 / (3 * kernel.l^2))) * exp(-sqrt(5) * norm(x-x2) / kernel.l)
end

function K(kernel::SpectralMixture, x, x2)
	sum(map(
		(wms) -> wms[1] * exp(-2 * pi^2 * dot(transpose(x - x2) * wms[3], x - x2)) * cos(2 * pi * dot(x - x2, wms[2])), 
		zip(kernel.weights, kernel.means, kernel.sigmas)))
end


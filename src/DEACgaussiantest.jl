module DEACgaussiantest
using SpecialFunctions
using JLD
using NPZ

export heaviside, genGaussianDSF_HS, genGaussianDSF2_HS, genGaussianDSF, genGaussianISF, genGaussianMoment0,
genGaussianMoment1, genGaussianMoment3, genGaussianMomentn1

#See https://journals.aps.org/prb/pdf/10.1103/PhysRevB.82.174510
heaviside(x::AbstractFloat) = ifelse(x < 0, zero(x), ifelse(x > 0, one(x), oftype(x,0.5)))

function genGaussianDSF_HS(omega,p,mu,alpha)
    heaviside.(omega).*(p*exp.(-((omega.-mu).^2)/2/(alpha^2))/sqrt(2*pi)/alpha)
end

function genGaussianDSF2_HS(o1,o2,p,mu,alpha)
    (heaviside.(o1).*(p*exp.(-((o1.-mu).^2)/2/(alpha^2))/sqrt(2*pi)/alpha) + heaviside.(o2).*(p*exp.(-((o2.-mu).^2)/2/(alpha^2))/sqrt(2*pi)/alpha))/2    
end

function genGaussianDSF(omega,p,mu,alpha)
    p .* exp.(-((omega.-mu).^2) ./ 2 ./ (alpha^2)) ./ sqrt(2*pi) ./ alpha
end

function genGaussianISF(tau,p,mu,alpha)
    p .* exp.(tau .* (((alpha^2) .* tau) .- (2*mu)) ./ 2)
end

function genGaussianMoment0(p,mu,alpha)
    p
end

function genGaussianMoment1(p,mu,alpha)
    mu*p
end

function genGaussianMoment3(p,mu,alpha)
    mu * ( (3*(alpha^2)) + mu^2 ) * p
end

function genGaussianMomentn1(p,mu,alpha)
    sqrt(2) * p * dawson(mu/alpha/sqrt(2))/alpha
end

function getMoment(moment_function::Function,p::Float64,mu::Float64,alpha::Float64)
    moment_function(p,mu,alpha)
end

function getMoment(moment_function::Function,p::Array{Float64,1},mu::Float64,alpha::Float64)
    m = 0.0;
    for pp in p
        m += moment_function(pp,mu,alpha);
    end
    m
end

function getMoment(moment_function::Function,p::Float64,mu::Array{Float64},alpha::Float64)
    m = 0.0;
    for u in mu
        m += moment_function(p,u,alpha);
    end
    m
end

function getMoment(moment_function::Function,p::Float64,mu::Float64,alpha::Array{Float64})
    m = 0.0;
    for a in alpha
        m += moment_function(p,mu,a);
    end
    m
end

function getMoment(moment_function::Function,p::Array{Float64},mu::Array{Float64},alpha::Float64)
    size(p) == size(mu) || throw(DimensionMismatch("size of p not equal to size of mu"))
    m = 0.0;
    @inbounds for i = 1:size(p,1)
        pp = p[i]
        u = mu[i]
        m += moment_function(pp,u,alpha);
    end
    m
end

function getMoment(moment_function::Function,p::Array{Float64},mu::Float64,alpha::Array{Float64})
    size(p) == size(alpha) || throw(DimensionMismatch("size of p not equal to size of mu"))
    m = 0.0;
    @inbounds for i = 1:size(p,1)
        pp = p[i]
        a = alpha[i]
        m += moment_function(pp,mu,a);
    end
    m
end

function getMoment(moment_function::Function,p::Float64,mu::Array{Float64},alpha::Array{Float64})
    size(mu) == size(alpha) || throw(DimensionMismatch("size of mu not equal to size of alpha"))
    m = 0.0;
    @inbounds for i = 1:size(mu,1)
        u = mu[i]
        a = alpha[i]
        m += moment_function(p,u,a);
    end
    m
end

function getMoment(moment_function::Function,p::Array{Float64},mu::Array{Float64},alpha::Array{Float64})
    size(p) == size(mu) || throw(DimensionMismatch("size of p not equal to size of mu"))
    size(p) == size(alpha) || throw(DimensionMismatch("size of p not equal to size of alpha"))
    m = 0.0;
    @inbounds for i = 1:size(p,1)
        pp = p[i]
        u = mu[i]
        a = alpha[i]
        m += moment_function(pp,u,a);
    end
    m
end

function printMoments(p,mu,alpha)
    m = getMoment(genGaussianMomentn1,p,mu,alpha);
    println("inverse first moment: $m")
    m = getMoment(genGaussianMoment0,p,mu,alpha);
    println("zeroeth moment: $m")
    m = getMoment(genGaussianMoment1,p,mu,alpha);
    println("first moment: $m")
    m = getMoment(genGaussianMoment3,p,mu,alpha);
    println("third moment: $m")
end

function getISF(tau::Array{Float64},p,mu,alpha)
    getSpectrum(genGaussianISF,tau,p,mu,alpha)
end

function getDSF(omega::Array{Float64},p,mu,alpha)
    getSpectrum(genGaussianDSF,omega,p,mu,alpha)
end

function getSpectrum(spectrum_function::Function,x::Array{Float64},p::Float64,mu::Float64,alpha::Float64)
    spectrum_function(x,p,mu,alpha)
end

function getSpectrum(spectrum_function::Function,x::Array{Float64},p::Array{Float64,1},mu::Float64,alpha::Float64)
    s = zeros(size(x,1));
    for pp in p
        s += spectrum_function(x,pp,mu,alpha);
    end
    s
end

function getSpectrum(spectrum_function::Function,x::Array{Float64},p::Float64,mu::Array{Float64},alpha::Float64)
    s = zeros(size(x,1));
    for u in mu
        s += spectrum_function(x,p,u,alpha);
    end
    s
end

function getSpectrum(spectrum_function::Function,x::Array{Float64},p::Float64,mu::Float64,alpha::Array{Float64})
    s = zeros(size(x,1));
    for a in alpha
        s += spectrum_function(x,p,mu,a);
    end
    s
end

function getSpectrum(spectrum_function::Function,x::Array{Float64},p::Array{Float64},mu::Array{Float64},alpha::Float64)
    size(p) == size(mu) || throw(DimensionMismatch("size of p not equal to size of mu"))
    s = zeros(size(x,1));
    @inbounds for i = 1:size(p,1)
        pp = p[i]
        u = mu[i]
        s += spectrum_function(x,pp,u,alpha);
    end
    s
end

function getSpectrum(spectrum_function::Function,x::Array{Float64},p::Array{Float64},mu::Float64,alpha::Array{Float64})
    size(p) == size(alpha) || throw(DimensionMismatch("size of p not equal to size of mu"))
    s = zeros(size(x,1));
    @inbounds for i = 1:size(p,1)
        pp = p[i]
        a = alpha[i]
        s += spectrum_function(x,pp,mu,a);
    end
    s
end

function getSpectrum(spectrum_function::Function,x::Array{Float64},p::Float64,mu::Array{Float64},alpha::Array{Float64})
    size(mu) == size(alpha) || throw(DimensionMismatch("size of mu not equal to size of alpha"))
    s = zeros(size(x,1));
    @inbounds for i = 1:size(mu,1)
        u = mu[i]
        a = alpha[i]
        s += spectrum_function(x,p,u,a);
    end
    s
end

function getSpectrum(spectrum_function::Function,x::Array{Float64},p::Array{Float64},mu::Array{Float64},alpha::Array{Float64})
    size(p) == size(mu) || throw(DimensionMismatch("size of p not equal to size of mu"))
    size(p) == size(alpha) || throw(DimensionMismatch("size of p not equal to size of alpha"))
    s = zeros(size(x,1));
    @inbounds for i = 1:size(p,1)
        pp = p[i]
        u = mu[i]
        a = alpha[i]
        s += spectrum_function(x,pp,u,a);
    end
    s
end

function get_tau(tau_0::Float64,tau_max::Float64,tau_steps::Int64)
    collect(tau_0:tau_max/(tau_steps - 1):tau_max);
end

function saveDataFile(tau_0::Float64,tau_max::Float64,tau_steps::Int64,p,mu,alpha,isf_error; filename::String="qmcdata.jld", print_moments::Bool=false)
    tau = get_tau(tau_0,tau_max,tau_steps);
    saveDataFile(tau,p,mu,alpha,isf_error,filename=filename,print_moments=print_moments);
end

function saveDataFile(tau::Array{Float64,1},p,mu,alpha,isf_error; filename::String="qmcdata.jld", print_moments::Bool=false)
    isf = getISF(tau,p,mu,alpha);
    if print_moments
        printMoments(p,mu,alpha);
    end
    saveDataFile(tau,isf,isf_error,filename=filename);
end

function saveDataFile(tau::Array{Float64,1},isf::Array{Float64,1},isf_error::Float64; filename::String="qmcdata.jld")
    size(tau) == size(isf) || throw(DimensionMismatch("size of tau not equal to size of isf"))
    isf_error_array = fill(isf_error,size(tau,1));
    saveDataFile(tau,isf,isf_error_array,filename=filename);
end

function saveDataFile(tau::Array{Float64,1},isf::Array{Float64,1},isf_error::Array{Float64,1}; filename="qmcdata.jld")
    size(tau) == size(isf) || throw(DimensionMismatch("size of tau not equal to size of isf"))
    size(tau) == size(isf_error) || throw(DimensionMismatch("size of tau not equal to size of isf"))

    # Get filename extension and save using correct method
    filename_base, filename_extension = splitext(filename);
    if filename_extension == ".jld"
        save(filename,
             "tau",tau,
             "isf",isf,
             "isf_error",isf_error);
    elseif filename_extension == ".npz"
        npz_dict = Dict("tau" => tau,
                        "isf" => isf,
                        "isf_error" => isf_error);
        npzwrite(filename, npz_dict);
    else
        throw(ArgumentError("unrecoginized file extension '$(filename_extension)', please choose from [.jld, .npz]"))
    end
    nothing
end
end

module DEACpopulation
using Random;
const functionNames = ["randPop",
                       "flatPop",
                       "gaussPop",
                       "gaussPop2",
                       "gaussOne",
                       "gaussTwo",
                       "gaussThree"]
#Generate population functions
function randPop(rng::MersenneTwister,genomeSize::Int64,populationSize::Int64,omega::Array{Float64,1})
    rand(rng,genomeSize,populationSize)
end

function flatPop(rng::MersenneTwister,genomeSize::Int64,populationSize::Int64,omega::Array{Float64,1})
    rand(rng) .* ones(genomeSize,populationSize)
end

function gaussPop(rng::MersenneTwister,genomeSize::Int64,populationSize::Int64,omega::Array{Float64,1})
    p = (rand(rng,populationSize));
    mu = (rand(rng,populationSize)*maximum(omega));
    alpha = (rand(rng,populationSize)*maximum(omega)/4);
    (p'.*exp.(-((omega .- mu').^2)./(alpha'.^2)/2)./alpha'/sqrt(2*pi))
end

function gaussPop2(rng::MersenneTwister,genomeSize::Int64,populationSize::Int64,omega::Array{Float64,1})
    p = ones(populationSize);
    mu = Array{Float64}(undef,populationSize);
    for i = 0:populationSize-1
        mu[i+1]=omega[(i%genomeSize)+1];
    end
    alpha = (rand(rng::MersenneTwister,populationSize));
    (p'.*exp.(-((omega .- mu').^2)./(alpha'.^2)/2)./alpha'/sqrt(2*pi));
end

function gaussOne(rng::MersenneTwister,genomeSize::Int64,populationSize::Int64,omega::Array{Float64,1})
    gaussPop(rng,genomeSize,populationSize,omega)
end

function gaussTwo(rng::MersenneTwister,genomeSize::Int64,populationSize::Int64,omega::Array{Float64,1})
    gaussPop(rng,genomeSize,populationSize,omega) + gaussPop(rng,genomeSize,populationSize,omega)
end

function gaussThree(rng::MersenneTwister,genomeSize::Int64,populationSize::Int64,omega::Array{Float64,1})
    gaussPop(rng,genomeSize,populationSize,omega) + gaussPop(rng,genomeSize,populationSize,omega) + gaussPop(rng,genomeSize,populationSize,omega)
end

end

module DEACreject
using Random
const functionNames = ["smallerFit",
                       "smallChange1",
                       "keepAll"]

function smallerFit(rng::MersenneTwister,rInd::BitArray{1},cFitness::Array{Float64,1},fitnessP::Array{Float64,1})
    @inbounds for i = 1:size(rInd,1)
        rInd[i] = cFitness[i] < fitnessP[i];
    end
    nothing
end

function smallChange1(rng::MersenneTwister,rInd::BitArray{1},cFitness::Array{Float64,1},fitnessP::Array{Float64,1})
    @inbounds for i = 1:size(rInd,1)
        rInd[i] = rand(rng) < exp(-abs(1.0-cFitness[i]/fitnessP[i]));
    end
    nothing
end

function keepAll(rng::MersenneTwister,rInd::BitArray{1},cFitness::Array{Float64,1},fitnessP::Array{Float64,1})
    @inbounds for i = 1:size(rInd,1)
        rInd[i] = true;
    end
    nothing
end

end

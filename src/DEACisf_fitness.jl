module DEACisf_fitness
using LinearAlgebra
const functionNames = ["mean_isf",
                       "max_isf",
                       "sum_isf",
                       "prod_isf",
                       "beta_over_two"]

function mean_isf(isfmT::Array{Float64,1},isf_model::Array{Float64,2},isf::Array{Float64,1})
    n_isf = Float64(size(isf_model,1));
    @inbounds for i = 1:size(isf_model,2)
        _s = 0.0;
        for j = 1:size(isf_model,1)
            _s += 1 + abs(1 - isf_model[j,i] / isf[j]);
        end
        isfmT[i] = _s/n_isf;
    end
    nothing
end

function max_isf(isfmT::Array{Float64,1},isf_model::Array{Float64,2},isf::Array{Float64,1})
    @inbounds for i = 1:size(isf_model,2)
        _s = 0.0;
        for j = 1:size(isf_model,1)
            _s = max(_s,abs(1 - isf_model[j,i] / isf[j]));
        end
        isfmT[i] = 1 + _s;
    end
    nothing
end

function sum_isf(isfmT::Array{Float64,1},isf_model::Array{Float64,2},isf::Array{Float64,1})
    @inbounds for i = 1:size(isf_model,2)
        _s = 1.0;
        for j = 1:size(isf_model,1)
            _s += abs(1 - isf_model[j,i] / isf[j]);
        end
        isfmT[i] = _s;
    end
    nothing
end

function prod_isf(isfmT::Array{Float64,1},isf_model::Array{Float64,2},isf::Array{Float64,1})
    @inbounds for i = 1:size(isf_model,2)
        _s = 1.0;
        for j = 1:size(isf_model,1)
            _s *= 1 + abs(1 - isf_model[j,i] / isf[j]);
        end
        isfmT[i] = _s;
    end
    nothing
end

function beta_over_two(isfmT::Array{Float64,1},isf_model::Array{Float64,2},isf::Array{Float64,1})
    @inbounds for i = 1:size(isf_model,2)
        _s = 1 + abs(1 - isf_model[end,i] / isf[end]);
        isfmT[i] = _s;
    end
    nothing
end
end

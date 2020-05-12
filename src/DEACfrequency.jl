module DEACfrequency
using Random;
using JLD;
const functionNames = ["exponential_frequency_bins",
                       "fixed_frequency_bins",
                       "load_frequency_bins"]

function exponential_frequency_bins(start::Float64,stop::Float64,length::Int64;epsilon::Float64=1e-5)
    frequency_bins = Array{Float64,1}(undef,length + 1);
    exponential_frequency_bins!(frequency_bins,start,stop,epsilon=epsilon);
    frequency_bins
end

function exponential_frequency_bins!(frequency_bins::Array{Float64,1},start::Float64,stop::Float64;epsilon::Float64=1e-5)
    length = size(frequency_bins,1) - 1;
    frequency_bins[1] = start;
    _logspace = range(log10(start+epsilon), stop=log10(stop), length=length)
    @inbounds for i in 2:length+1
        frequency_bins[i] = exp10(_logspace[i-1]);
    end
    nothing
end

function fixed_frequency_bins(start::Float64,stop::Float64,length::Int64;epsilon::Float64=1e-5)
    frequency_bins = Array{Float64,1}(undef,length+1);
    fixed_frequency_bins!(frequency_bins,start,stop);
    frequency_bins
end

function fixed_frequency_bins!(frequency_bins::Array{Float64,1},start::Float64,stop::Float64;epsilon::Float64=1e-5)
    length = size(frequency_bins,1);
    _linspace = range(start,stop=stop,length=length);
    for i in 1:length
        frequency_bins[i]=_linspace[i];
    end
    nothing
end

function load_frequency_bins(start::Float64,stop::Float64,length::Int64;filename::String="./frequency.jld")
    frequency_bins = load_frequency_bins(filename);
    frequency_bins
end

function load_frequency_bins(filename::String="./frequency.jld")
    frequency_bins = JLD.load(filename,"frequency_bins");
    frequency_bins
end

end

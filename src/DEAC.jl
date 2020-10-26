module DEAC
include("./DEACfitness.jl")
include("./DEACisf_fitness.jl")
include("./DEACpopulation.jl")
include("./DEACreject.jl")
include("./DEACfrequency.jl")

using DelimitedFiles
using UUIDs
using Random
using Statistics
using LinearAlgebra

export deac

function mutant_indexes(rng::MersenneTwister,populationSize::Int64,idx0::Int64)
    idx1 = idx0;
    idx2 = idx0;
    idx3 = idx0;
    while idx1 == idx0
        idx1 = rand(rng,1:populationSize);
    end

    while idx2 == idx0 || idx2 == idx1
        idx2 = rand(rng,1:populationSize);
    end
    while idx3 == idx0 || idx3 == idx1 || idx3 == idx2
        idx3 = rand(rng,1:populationSize);
    end
    idx1,idx2,idx3
end

function set_differential_weights(rng::MersenneTwister,new_differential_weights::Array{Float64,1},
                              differential_weights::Array{Float64,1},SAdifferentialWeightProb::Float64,
                              SAdifferentialWeightShift::Float64,SAdifferentialWeight::Float64)
    @inbounds for i = eachindex(new_differential_weights)
        if rand(rng) < SAdifferentialWeightProb
            #new_differential_weights[i] = SAdifferentialWeightShift + SAdifferentialWeight*rand(rng);
            new_differential_weights[i] = 2.0*rand(rng);
        else
            new_differential_weights[i] = differential_weights[i];
        end
    end
    nothing
end

function set_crossover_probs(rng::MersenneTwister,new_crossover_probs::Array{Float64,1},
                                   crossover_probs::Array{Float64,1},SAcrossoverProb::Float64)
    @inbounds for i = eachindex(new_crossover_probs)
        if rand(rng) < SAcrossoverProb
            new_crossover_probs[i] = rand(rng);
        else
            new_crossover_probs[i] = crossover_probs[i];
        end
    end
    nothing
end

function set_mutateInd(rng::MersenneTwister,mutateInd::BitArray{2},new_crossover_probs::Array{Float64,1})
    @inbounds for j = 1:size(mutateInd,2)
        cR = new_crossover_probs[j];
        for i = 1:size(mutateInd,1)
            mutateInd[i,j] = (rand(rng) < cR);
        end
    end
    nothing
end

function mutate(rng::MersenneTwister,mP::Array{Float64,2},P::Array{Float64,2},
                 mIdx::BitArray{2},new_differential_weights::Array{Float64,1})
    @inbounds for j = 1:size(mP,2)
        F = new_differential_weights[j];
        mIdx1,mIdx2,mIdx3 = mutant_indexes(rng,size(mP,2),j);
        for i = 1:size(mP,1)
            if mIdx[i,j]
                mP[i,j] = abs(P[i,mIdx1] + (F * (P[i,mIdx2]-P[i,mIdx3])));
            else
                mP[i,j] = P[i,j];
            end
        end
    end
    nothing
end

function generate_pop(rng::MersenneTwister,genomeSize::Int64,populationSize::Int64,omega::Array{Float64,1},gpfn::String)
    gpfunc = getfield(DEACpopulation, Symbol(gpfn));
    gpfunc(rng,genomeSize,populationSize,omega)
end

function replace_pop(p::Array{Float64,2},c::Array{Float64,2},rInd::BitArray{1})
    @inbounds for i = 1:size(rInd,1)
        idx = rInd[i];
        if idx
            for j = 1:size(p,1)
                p[j,i] = c[j,i];
            end
        end
    end
    nothing
end

function replace_fitness(pFitness::Array{Float64,1},cFitness::Array{Float64,1},rInd::BitArray{1})
    @inbounds for i = 1:size(rInd,1)
        idx = rInd[i];
        if idx
            pFitness[i] = cFitness[i];
        end
    end
    nothing
end

function replace_crossover_probs(crossover_probs::Array{Float64,1},new_crossover_probs::Array{Float64,1},rInd::BitArray{1})
    replace_fitness(crossover_probs,new_crossover_probs,rInd);
end

function replace_differential_weights(differential_weights::Array{Float64,1},new_differential_weights::Array{Float64,1},rInd::BitArray{1})
    replace_fitness(differential_weights,new_differential_weights,rInd);
end

function set_minP(minP::Array{Float64,1},P::Array{Float64,2},minidx::Int64)
    @inbounds for i = 1:size(P,1)
        minP[i] = P[i,minidx];
    end
    nothing
end

function set_isf_term(imaginary_time::Array{Float64,1},frequency_bins::Array{Float64,1},beta::Float64)
    b = beta;
    isf_term = Array{Float64,2}(undef,size(imaginary_time,1),size(frequency_bins,1));
    for i in 1:size(imaginary_time,1)
        t = imaginary_time[i];
        bo2mt = b/2 - t;
        for j in 1:size(frequency_bins,1)
            f = frequency_bins[j];
            isf_term[i,j] = cosh(bo2mt*f);
        end
    end
    isf_term
end

function beam_distance(d::Array{Float64,1},frequency::Array{Float64,1},P::Array{Float64,2})
    #FIXME this method is slow
    for j in 1:(size(P,2))
        d[j] = 1.0;
        for i in 1:(size(frequency,1) - 2)
            d[j] += abs((P[i+2,j] - P[i,j])*frequency[i+1] - (frequency[i+2] - frequency[i])*P[i+1,j] + frequency[i+2]*P[i,j] - frequency[i]*P[i+2,j])/sqrt((frequency[i+2] - frequency[i])^2 + (P[i+2,j] - P[i,j])^2);
        end
        nothing
    end
end

function deac(  imaginary_time::Array{Float64,1},
                isf::Array{Float64,1},
                isf_error::Array{Float64,1},
                frequency::Array{Float64,1};
                smooth::Bool = false,
                normalize::Bool = false,
                use_inverse_first_moment::Bool = false,
                first_moment::Float64 = -1.0,
                third_moment::Float64 = -1.0,
                third_moment_error::Float64 = 0.01,
                temperature::Float64 = 1.2,
                number_of_generations::Int64 = 1000,
                population_size::Int64 = 512,
                crossoverProb::Float64=0.9,
                SAcrossoverProb::Float64=0.1,
                differentialWeight::Float64=0.9,
                SAdifferentialWeightProb::Float64=0.1,
                SAdifferentialWeightShift::Float64=0.1,
                SAdifferentialWeight::Float64=0.9,
                rejectFunc::String="smallerFit",
                stop_minimum_fitness::Float64 = 1.0e-8,
                seed::Int64 = 1,
                track_stats::Bool = false,
                number_of_blas_threads::Int64 = 0)

    #_bts = ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())
    #println("Number of BLAS threads: $(_bts)");
    if number_of_blas_threads > 0
        LinearAlgebra.BLAS.set_num_threads(number_of_blas_threads)
    end

    reject = getfield(DEACreject, Symbol(rejectFunc));
    
    #Set uuid
    u4 = uuid4();

    #Set RNG
    rng = MersenneTwister(seed);

    #Set constants
    ntau = size(imaginary_time,1);
    ndsf = size(frequency,1);
    npop = population_size;
    isf_r = 1.0 ./ isf;
    beta = 1/temperature;

    moment0=isf[1];
    isf_term = set_isf_term(imaginary_time,frequency,beta);
    isf_term2 = copy(isf_term);

    df = frequency[2:end] .- frequency[1:size(frequency,1) - 1];
    dfrequency = zeros(size(frequency,1));
    dfrequency2 = zeros(size(frequency,1));
    for i in 1:(size(frequency,1) - 1)
        dfrequency[i] = df[i]/2
        dfrequency2[i+1] = df[i]/2
    end
    dfrequency3 = dfrequency .+ dfrequency2;

    isf_term .*= dfrequency';
    isf_term2 .*= dfrequency2';
    isf_term .+= isf_term2;
    
    isf_m = Array{Float64}(undef,ntau,npop);

    #Generate population and set initial fitness

    P = Array{Float64,2}(undef,ndsf,npop);
    P = rand!(rng,P);

    # Normalize population
    if normalize
        normalization_factor = cosh.((beta/2) .* frequency);
        normalization_term = normalization_factor .* dfrequency3;
        normalization = Array{Float64,1}(undef,npop);
        mul!(normalization,P',normalization_term);
        normalization ./= moment0;
        P ./= normalization';
    end
    P_new = Array{Float64,2}(undef,ndsf,npop);
    fitness = zeros(npop);
    fitness_new = Array{Float64,1}(undef,npop);

    if first_moment > 0.0
        first_moments_factor = frequency .* sinh.((beta/2) .* frequency);
        first_moments_term = first_moments_factor .* dfrequency;
        first_moments_term2 = first_moments_factor .* dfrequency2;
        first_moments_term .+= first_moments_term2;

        first_moments = Array{Float64,1}(undef,npop);

        mul!(first_moments', first_moments_term', P);
        #FIXME see if faster
        #mul!(first_moments, P', first_moments_term);
    end

    if third_moment > 0.0
        third_moments_factor = (frequency .^ 3) .* sinh.((beta/2) .* frequency);
        third_moments_term = third_moments_factor .* dfrequency;
        third_moments_term2 = third_moments_factor .* dfrequency2;
        third_moments_term .+= third_moments_term2;

        third_moments = Array{Float64,1}(undef,npop);

        mul!(third_moments', third_moments_term', P);
        #FIXME see if faster
        #mul!(third_moments, P', third_moments_term);
    end
    
    #set isf_m and calculate fitness
    mul!(isf_m,isf_term,P);
    
    if use_inverse_first_moment
        dt = imaginary_time[2:end] .- imaginary_time[1:size(imaginary_time,1) - 1];
        dimaginary_time = zeros(size(imaginary_time,1));
        dimaginary_time2 = zeros(size(imaginary_time,1));
        for i in 1:(size(imaginary_time,1) - 1)
            dimaginary_time[i] = dt[i]/2
            dimaginary_time2[i+1] = dt[i]/2
        end
        dimaginary_time .+= dimaginary_time2;
        inverse_first_moment = dot(isf,dimaginary_time);
        inverse_first_moment_error = sqrt(dot(isf_error.^2,dimaginary_time.^2));
        inverse_first_moments = Array{Float64,1}(undef,npop);
        mul!(inverse_first_moments,isf_m',dimaginary_time);
    end

    broadcast!((x,y,z)->(((x-y)/z)^2),isf_m,isf,isf_m,isf_error);
    mean!(fitness',isf_m);

    if use_inverse_first_moment
        broadcast!((x,y,z)->(((x-y)/z)^2),inverse_first_moments,inverse_first_moment,inverse_first_moments,inverse_first_moment_error);
        fitness .+= inverse_first_moments;
    end

    if first_moment > 0.0
        fitness .+= (first_moments .- first_moment).^2;
    end
    if third_moment > 0.0
        broadcast!((x,y,z)->(((x-y)/z)^2),third_moments,third_moment,third_moments,third_moment_error);
        fitness .+= third_moments;
    end

    if smooth
        total_beam_distances = Array{Float64,1}(undef,npop);
        beam_distance(total_beam_distances,frequency,P);
        fitness .*= total_beam_distances.^2;
    end

    crossover_probs = ones(npop) .* crossoverProb;
    differential_weights = ones(npop) .* differentialWeight;
    new_crossover_probs = Array{Float64,1}(undef,npop);
    new_differential_weights = Array{Float64,1}(undef,npop);
    
    #Initialize statistics arrays
    
    if track_stats
        avgFitness = zeros(number_of_generations);
        minFitness = zeros(number_of_generations);
        stdFitness = zeros(number_of_generations);
    end

    #Crude Preallocation
    mIdx = trues(ndsf,npop);
    generation = 1;
    
    rInd = falses(npop); #rejection indexes

    minP = P[:,argmin(fitness)];
    minFit = minimum(fitness);
    avgFit = mean(fitness);
    stdFit = std(fitness,mean=avgFit);

    #Begin loop
    @inbounds for outer generation = 1:number_of_generations
        avgFit = mean(fitness);
        minFit = minimum(fitness);
        stdFit = std(fitness,mean=avgFit);

        #Get Statistics
        if track_stats
            avgFitness[generation] = avgFit;
            minFitness[generation] = minFit;
            stdFitness[generation] = stdFit;
        end

        #Stopping criteria
        if minFit <= stop_minimum_fitness
            break
        end

        set_crossover_probs(rng,new_crossover_probs,crossover_probs,SAcrossoverProb);
        set_differential_weights(rng,new_differential_weights,differential_weights,SAdifferentialWeightProb,
                                 SAdifferentialWeightShift,SAdifferentialWeight);

        #Set mutant population FIXME crossover built-in here do I want this?
        set_mutateInd(rng,mIdx,new_crossover_probs);
        mutate(rng,P_new,P,mIdx,new_differential_weights);

        # Normalization
        if normalize
            mul!(normalization,P_new',normalization_term);
            normalization ./= moment0;
            P_new ./= normalization';
        end

        #Rejection
        if first_moment > 0.0
            mul!(first_moments', first_moments_term', P_new);
        end

        if third_moment > 0.0
            mul!(third_moments', third_moments_term', P_new);
        end

        mul!(isf_m,isf_term,P_new);

        if use_inverse_first_moment
            mul!(inverse_first_moments,isf_m',dimaginary_time);
        end

        broadcast!((x,y,z)->(((x-y)/z)^2),isf_m,isf,isf_m,isf_error);
        mean!(fitness_new',isf_m);

        if use_inverse_first_moment
            broadcast!((x,y,z)->(((x-y)/z)^2),inverse_first_moments,inverse_first_moment,inverse_first_moments,inverse_first_moment_error);
            fitness_new .+= inverse_first_moments;
        end

        if first_moment > 0.0
            fitness_new .+= (first_moments .- first_moment).^2;
        end

        if third_moment > 0.0
            broadcast!((x,y,z)->(((x-y)/z)^2),third_moments,third_moment,third_moments,third_moment_error);
            fitness_new .+= third_moments;
        end

        if smooth
            beam_distance(total_beam_distances,frequency,P_new);
            fitness_new .*= total_beam_distances.^2;
        end

        reject(rng,rInd,fitness_new,fitness);
        replace_pop(P,P_new,rInd);
        replace_fitness(fitness,fitness_new,rInd);
        replace_crossover_probs(crossover_probs,new_crossover_probs,rInd);
        replace_differential_weights(differential_weights,new_differential_weights,rInd);

    end
    #Set best candidate solution from final generation
    minidx = argmin(fitness);
    set_minP(minP,P,minidx);

    minS = minP .* exp.((beta/2) .* frequency) ./ 2; 
    println(minimum(fitness));
    if track_stats
        return u4,frequency,minS,minP,minimum(fitness),generation,avgFitness[1:generation],minFitness[1:generation],stdFitness[1:generation],P,fitness,rng,crossover_probs,differential_weights
    end
    return u4,frequency,minS,minP,minimum(fitness),generation,zeros(2),zeros(2),zeros(2),P,fitness,rng,crossover_probs,differential_weights
end
end

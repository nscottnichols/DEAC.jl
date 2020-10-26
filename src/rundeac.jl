using ArgParse
using Random
using JLD
using NPZ
include("./DEAC.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--temperature", "-T"
            help = "Temperature of system."
            arg_type = Float64
            default = 0.0
        "--number_of_generations", "-N"
            help = "Number of generations before genetic algorithm quits."
            arg_type = Int64
            default = 100000
        "--population_size", "-P"
            help = "Size of initial population"
            arg_type = Int64
            default = 512
        "--genome_size", "-M"
            help = "Size of genome."
            arg_type = Int64
            default = 600
        "--omega_max"
            help = "Maximum frequency to explore."
            arg_type = Float64
            default = 60.0
        "--smooth"
            help = "Use smoothing factor in fitness."
            action = :store_true
        "--normalize"
            help = "Normalize spectrum to the zeroeth moment."
            action = :store_true
        "--use_inverse_first_moment"
            help = "Calculate inverse first moment from ISF data and use it in fitness."
            action = :store_true
        "--first_moment"
            help = "FIXME First moment."
            arg_type = Float64
            default = -1.0
        "--third_moment"
            help = "FIXME Third moment."
            arg_type = Float64
            default = -1.0
        "--third_moment_error"
            help = "FIXME Third moment."
            arg_type = Float64
            default = 0.01
        "--crossover_probability", "-r"
            help = "Initial probability for parent gene to become mutant vector gene."
            arg_type = Float64
            default = 0.90
        "--self_adapting_crossover_probability", "-u"
            help = "Probability for `crossover_probability` to mutate."
            arg_type = Float64
            default = 0.10
        "--differential_weight", "-F"
            help = "Initial weight factor when creating mutant vector."
            arg_type = Float64
            default = 0.90
        "--self_adapting_differential_weight_probability", "-v"
            help = "Probability for `differential_weight` to mutate."
            arg_type = Float64
            default = 0.10
        "--self_adapting_differential_weight_shift", "-l"
            help = "If `self_adapting_differential_weight_probability` mutate, new value is `l + m*rand()`."
            arg_type = Float64
            default = 0.10
        "--self_adapting_differential_weight", "-m"
            help = "If `self_adapting_differential_weight_probability` mutate, new value is `l + m*rand()`."
            arg_type = Float64
            default = 0.90
        "--reject"
            help = "Name of rejection function."
            arg_type = String
            default = "smallerFit"
        "--frequency_bins_method"
            help = "Name of function to generate frequency bins."
            arg_type = String
            default = "fixed_frequency_bins"
        "--load_frequency_bins_filename"
            help = "A JLD file containing frequency_bins to load."
            arg_type = String
            default = "./frequency.jld"
        "--stop_minimum_fitness"
            help = "Stopping criteria, if minimum fitness is below `stop_minimum_fitness` stop evolving."
            arg_type = Float64
            default = 1.0
        "--seed"
            help = "Seed to pass to random number generator."
            arg_type = Int64
            default = 1
        "--save_state"
            help = "Save state of DEAC algorithm. Saves the random number generator, population, and population fitness."
            action = :store_true
        "--save_file_dir"
            help = "Directory to save results in."
            arg_type = String
            default = "./deacresults"
        "--track_stats"
            help = "Track minimum fitness and other stats."
            action = :store_true
        "--number_of_blas_threads"
            help = "Number of BLAS threads to utilize if > 0 otherwise uses Julia default value."
            arg_type = Int64
            default = 0
        "qmc_data"
            help = "File containing quantum Monte Carlo data with columns: IMAGINARY_TIME, INTERMEDIATE_SCATTERING_FUNCTION, ERROR"
            arg_type = String
            required = true
    end

    return parse_args(s)
end

function checkMoves(argname::String,s::Array{String,1},l::Array{String,1})
    for ss in s
        checkMoves(argname,ss,l)
    end
end

function checkMoves(argname::String,s::String,l::Array{String,1})
    if !(s in l)
        print("$s is not a valid parameter for $argname. Valid parameters are: ")
        println(l)
        error("Failed to validate argument: $argname")
    end
nothing
end

function main()
    start = time();
    parsed_args = parse_commandline()

    # create data directory
    try
        mkpath(parsed_args["save_file_dir"]);
    catch
        nothing
    end
    save_dir = parsed_args["save_file_dir"];

    checkMoves("reject",parsed_args["reject"],DEAC.DEACreject.functionNames)
    checkMoves("frequency_bins_method",parsed_args["frequency_bins_method"],DEAC.DEACfrequency.functionNames)
    get_frequency_bins = getfield(DEAC.DEACfrequency, Symbol(parsed_args["frequency_bins_method"]));

    if parsed_args["frequency_bins_method"] == "load_frequency_bins"
        frequency_bins_filename = parsed_args["load_frequency_bins_filename"];
        println("loading frequency_bins from $(frequency_bins_filename) (will override genome_size and omega_max");
        frequency_bins = get_frequency_bins(frequency_bins_file);
    else
        frequency_bins = get_frequency_bins(0.0,parsed_args["omega_max"],parsed_args["genome_size"]);
    end
    #FIXME ensure frequency odd
    if size(frequency_bins,1)%2 == 0
        throw(AssertionError("frequency space must be odd FIXME"))
    end

    _ext = splitext(parsed_args["qmc_data"])[2];
    if _ext == ".npz"
        qmcdata=NPZ.npzread(parsed_args["qmc_data"]);
    elseif _ext == ".jld"
        qmcdata = load(parsed_args["qmc_data"]);
    else
        throw(AssertionError("qmc_data must be *.jld or *.npz"));
    end
    tau = qmcdata["tau"];
    isf = qmcdata["isf"];
    err = qmcdata["error"];

    u4,omega,minS,minP,minFit,total_generations,
    avgFitness,minFitness,
    stdFitness,P,
    fitnessP,rng,
    crossover_probs,differential_weights = DEAC.deac( tau,isf,err,frequency_bins,
                           use_inverse_first_moment = parsed_args["use_inverse_first_moment"],
                           smooth = parsed_args["smooth"],
                           normalize = parsed_args["normalize"],
                           first_moment = parsed_args["first_moment"],
                           third_moment = parsed_args["third_moment"],
                           third_moment_error = parsed_args["third_moment_error"],
                           temperature = parsed_args["temperature"],
                           number_of_generations = parsed_args["number_of_generations"],
                           population_size = parsed_args["population_size"],
                           crossoverProb = parsed_args["crossover_probability"],
                           SAcrossoverProb = parsed_args["self_adapting_crossover_probability"],
                           differentialWeight = parsed_args["differential_weight"],
                           SAdifferentialWeightProb = parsed_args["self_adapting_differential_weight_probability"],
                           SAdifferentialWeightShift = parsed_args["self_adapting_differential_weight_shift"],
                           SAdifferentialWeight = parsed_args["self_adapting_differential_weight"],
                           rejectFunc = parsed_args["reject"],
                           stop_minimum_fitness = parsed_args["stop_minimum_fitness"],
                           seed = parsed_args["seed"],
                           track_stats = parsed_args["track_stats"],
                           number_of_blas_threads = parsed_args["number_of_blas_threads"])
    seed = lpad(parsed_args["seed"],4,'0');
    elapsed = time() - start;
    filename = "$(save_dir)/deac_results_$(seed)_$u4.jld";
    println("Saving results to $filename");
    save(filename,
         "u4",u4,
         "frequency",omega,
         "dsf",minS,
         "minP",minP,
         "minFit",minFit,
         "total_generations",total_generations,
         "elapsed_time",elapsed);

    if parsed_args["track_stats"]
        filename = "$(save_dir)/deac_stats_$(seed)_$u4.jld";
        println("Saving stats to $filename");
        save(filename,
             "u4",u4,
             "avgFitness",avgFitness,
             "minFitness",minFitness,
             "stdFitness",stdFitness)
    end

    filename = "$(save_dir)/deac_params_$(seed)_$u4.jld";
    println("Saving parameters to $filename");
    save(filename,
         "u4",u4,
         "tau",tau,
         "isf",isf,
         "isf_error",err,
         "frequency",frequency_bins,
         "first_moment",parsed_args["first_moment"],
         "third_moment",parsed_args["third_moment"],
         "third_moment_error",parsed_args["third_moment_error"],
         "temperature",parsed_args["temperature"],
         "number_of_generations",parsed_args["number_of_generations"],
         "population_size",parsed_args["population_size"],
         "genome_size",parsed_args["genome_size"],
         "omega_max",parsed_args["omega_max"],
         "crossoverProb",parsed_args["crossover_probability"],
         "SAcrossoverProb",parsed_args["self_adapting_crossover_probability"],
         "differentialWeight",parsed_args["differential_weight"],
         "SAdifferentialWeightProb",parsed_args["self_adapting_differential_weight_probability"],
         "SAdifferentialWeightShift",parsed_args["self_adapting_differential_weight_shift"],
         "SAdifferentialWeight",parsed_args["self_adapting_differential_weight"],
         "rejectFunc",parsed_args["reject"],
         "frequency_bins_method",parsed_args["frequency_bins_method"],
         "load_frequency_bins_filename",parsed_args["load_frequency_bins_filename"],
         "stop_minimum_fitness",parsed_args["stop_minimum_fitness"],
         "seed",parsed_args["seed"])
    if parsed_args["save_state"]
        filename = "$(save_dir)/deac_state_$(seed)_$(seed_resample)_$u4.jld";
        println("Saving state to $filename");
        save(filename,
             "u4",u4,
             "P",P,
             "fitness",fitnessP,
             "rng",rng,
             "crossover_probs",crossover_probs,
             "differential_weights",differential_weights)
    end
    nothing
end

main()

module DEACfitness
using Statistics
using LinearAlgebra
const functionNames = ["isfonly",
                       "isfMn1",
                       "isfM0",
                       "isfMn1M0",
                       "isfMn1M0M1",
                       "isfMn1M0M1M3",
                       "Mn1M0M1",
                       "Mn1M0M1M3"]
const fitMethodNames = ["prodFit",
                        "sumFit",
                        "maxFit"]
function prodabsFit(fT::Float64)
    1.0 + fT;
end

function prodabsFit(fT::Float64,moment::Float64)
    1.0 + abs(moment - fT);
end


function absFit(fT::Float64,moment::Float64)
    abs(moment - fT);
end

function prodFit(f1::Float64,f2::Float64,f3::Float64,f4::Float64,f5::Float64,momentn1::Float64,moment0::Float64,moment1::Float64,moment3::Float64)
    prodabsFit(f1) * prodabsFit(f2,momentn1) * prodabsFit(f3,moment0) * prodabsFit(f4,moment1) * prodabsFit(f5,moment3) - 1.0;    
end

function sumFit(f1::Float64,f2::Float64,f3::Float64,f4::Float64,f5::Float64,momentn1::Float64,moment0::Float64,moment1::Float64,moment3::Float64)
    f1 + absFit(f2,momentn1) + absFit(f3,moment0) + absFit(f4,moment1) + absFit(f5,moment3);    
end

function sumFitDebug(f1::Float64,f2::Float64,f3::Float64,f4::Float64,f5::Float64,momentn1::Float64,moment0::Float64,moment1::Float64,moment3::Float64)
    t1 = f1;
    t2 = absFit(f2,momentn1);
    t3 = absFit(f3,moment0);
    t4 = absFit(f4,moment1);
    t5 = absFit(f5,moment3);
    println("t1: $(t1)");
    println("t2: $(t2)");
    println("t3: $(t3)");
    println("t4: $(t4)");
    println("t5: $(t5)");
    t1 + t2 + t3 + t4 + t5;    
end

function maxFit(f1::Float64,f2::Float64,f3::Float64,f4::Float64,f5::Float64,momentn1::Float64,moment0::Float64,moment1::Float64,moment3::Float64)
    max(absFit(f1),absFit(f2),absFit(f3),absFit(f4),absFit(f5));
end


function isfonly(fmfunc::Function,fitnessP::Array{Float64,1},P::Array{Float64,2},isf::Array{Float64,1},momentn1::Float64,moment0::Float64,moment1::Float64,moment3::Float64,F_term_P1::Array{Float64,2},F_term_P2::Array{Float64,2},F_term_avg::Array{Float64,1},mn1_term_P1::Array{Float64,1},mn1_term_P2::Array{Float64,1},m0_term_P1::Array{Float64,1},m0_term_P2::Array{Float64,1},m1_term_P1::Array{Float64,1},m1_term_P2::Array{Float64,1},m3_term_P1::Array{Float64,1},m3_term_P2::Array{Float64,1},isfmT_P1::Array{Float64,2},mn1T_P1::Array{Float64,1},m0T_P1::Array{Float64,1},m1T_P1::Array{Float64,1},m3T_P1::Array{Float64,1},isfmT_P2::Array{Float64,2},mn1T_P2::Array{Float64,1},m0T_P2::Array{Float64,1},m1T_P2::Array{Float64,1},m3T_P2::Array{Float64,1})
    fitInd = trues(size(P,2));
    isfonly(fmfunc,fitnessP,fitInd,P,isf,momentn1,moment0,moment1,moment3,F_term_P1,F_term_P2,F_term_avg,mn1_term_P1,mn1_term_P2,m0_term_P1,m0_term_P2,m1_term_P1,m1_term_P2,m3_term_P1,m3_term_P2,isfmT_P1,mn1T_P1,m0T_P1,m1T_P1,m3T_P1,isfmT_P2,mn1T_P2,m0T_P2,m1T_P2,m3T_P2)
end

function isfonly(fmfunc::Function,fitnessP::Array{Float64,1},fitInd::BitArray{1},P::Array{Float64,2},isf::Array{Float64,1},momentn1::Float64,moment0::Float64,moment1::Float64,moment3::Float64,F_term_P1::Array{Float64,2},F_term_P2::Array{Float64,2},F_term_avg::Array{Float64,1},mn1_term_P1::Array{Float64,1},mn1_term_P2::Array{Float64,1},m0_term_P1::Array{Float64,1},m0_term_P2::Array{Float64,1},m1_term_P1::Array{Float64,1},m1_term_P2::Array{Float64,1},m3_term_P1::Array{Float64,1},m3_term_P2::Array{Float64,1},isfmT_P1::Array{Float64,2},mn1T_P1::Array{Float64,1},m0T_P1::Array{Float64,1},m1T_P1::Array{Float64,1},m3T_P1::Array{Float64,1},isfmT_P2::Array{Float64,2},mn1T_P2::Array{Float64,1},m0T_P2::Array{Float64,1},m1T_P2::Array{Float64,1},m3T_P2::Array{Float64,1})
    mul!(F_term_P1,isfmT_P1,P[1:end-1,:]);
    mul!(F_term_P2,isfmT_P2,P[2:end,:]);
    mean!(F_term_avg,abs.(isf .- (F_term_P1 .+ F_term_P2))');
    f1 = 0.0;
    f2 = momentn1;
    f3 = moment0;
    f4 = moment1;
    f5 = moment3;
    @inbounds for j = 1:size(fitInd,1)
        if fitInd[j]
            f1 = F_term_avg[j];
            fitnessP[j] = fmfunc(f1,f2,f3,f4,f5,momentn1,moment0,moment1,moment3); #testnewfit4
        end
    end
    nothing
end

function isfMn1(fmfunc::Function,fitnessP::Array{Float64,1},P::Array{Float64,2},isf::Array{Float64,1},momentn1::Float64,moment0::Float64,moment1::Float64,moment3::Float64,F_term_P1::Array{Float64,2},F_term_P2::Array{Float64,2},F_term_avg::Array{Float64,1},mn1_term_P1::Array{Float64,1},mn1_term_P2::Array{Float64,1},m0_term_P1::Array{Float64,1},m0_term_P2::Array{Float64,1},m1_term_P1::Array{Float64,1},m1_term_P2::Array{Float64,1},m3_term_P1::Array{Float64,1},m3_term_P2::Array{Float64,1},isfmT_P1::Array{Float64,2},mn1T_P1::Array{Float64,1},m0T_P1::Array{Float64,1},m1T_P1::Array{Float64,1},m3T_P1::Array{Float64,1},isfmT_P2::Array{Float64,2},mn1T_P2::Array{Float64,1},m0T_P2::Array{Float64,1},m1T_P2::Array{Float64,1},m3T_P2::Array{Float64,1})
    fitInd = trues(size(P,2));
    isfMn1(fmfunc,fitnessP,fitInd,P,isf,momentn1,moment0,moment1,moment3,F_term_P1,F_term_P2,F_term_avg,mn1_term_P1,mn1_term_P2,m0_term_P1,m0_term_P2,m1_term_P1,m1_term_P2,m3_term_P1,m3_term_P2,isfmT_P1,mn1T_P1,m0T_P1,m1T_P1,m3T_P1,isfmT_P2,mn1T_P2,m0T_P2,m1T_P2,m3T_P2)
end

function isfMn1(fmfunc::Function,fitnessP::Array{Float64,1},fitInd::BitArray{1},P::Array{Float64,2},isf::Array{Float64,1},momentn1::Float64,moment0::Float64,moment1::Float64,moment3::Float64,F_term_P1::Array{Float64,2},F_term_P2::Array{Float64,2},F_term_avg::Array{Float64,1},mn1_term_P1::Array{Float64,1},mn1_term_P2::Array{Float64,1},m0_term_P1::Array{Float64,1},m0_term_P2::Array{Float64,1},m1_term_P1::Array{Float64,1},m1_term_P2::Array{Float64,1},m3_term_P1::Array{Float64,1},m3_term_P2::Array{Float64,1},isfmT_P1::Array{Float64,2},mn1T_P1::Array{Float64,1},m0T_P1::Array{Float64,1},m1T_P1::Array{Float64,1},m3T_P1::Array{Float64,1},isfmT_P2::Array{Float64,2},mn1T_P2::Array{Float64,1},m0T_P2::Array{Float64,1},m1T_P2::Array{Float64,1},m3T_P2::Array{Float64,1})
    mul!(F_term_P1,isfmT_P1,P[1:end-1,:]);
    mul!(F_term_P2,isfmT_P2,P[2:end,:]);
    mean!(F_term_avg,abs.(isf .- (F_term_P1 .+ F_term_P2))');
    mul!(mn1_term_P1,P[1:end-1,:]',mn1T_P1);
    mul!(mn1_term_P2,P[2:end,:]',mn1T_P2);
    f1 = 0.0;
    f2 = 0.0;
    f3 = moment0;
    f4 = moment1;
    f5 = moment3;
    @inbounds for j = 1:size(fitInd,1)
        if fitInd[j]
            f1 = F_term_avg[j];
            f2 = mn1_term_P1[j]+mn1_term_P2[j];
            fitnessP[j] = fmfunc(f1,f2,f3,f4,f5,momentn1,moment0,moment1,moment3); #testnewfit4
        end
    end
    nothing
end

function isfM0(fmfunc::Function,fitnessP::Array{Float64,1},P::Array{Float64,2},isf::Array{Float64,1},momentn1::Float64,moment0::Float64,moment1::Float64,moment3::Float64,F_term_P1::Array{Float64,2},F_term_P2::Array{Float64,2},F_term_avg::Array{Float64,1},mn1_term_P1::Array{Float64,1},mn1_term_P2::Array{Float64,1},m0_term_P1::Array{Float64,1},m0_term_P2::Array{Float64,1},m1_term_P1::Array{Float64,1},m1_term_P2::Array{Float64,1},m3_term_P1::Array{Float64,1},m3_term_P2::Array{Float64,1},isfmT_P1::Array{Float64,2},mn1T_P1::Array{Float64,1},m0T_P1::Array{Float64,1},m1T_P1::Array{Float64,1},m3T_P1::Array{Float64,1},isfmT_P2::Array{Float64,2},mn1T_P2::Array{Float64,1},m0T_P2::Array{Float64,1},m1T_P2::Array{Float64,1},m3T_P2::Array{Float64,1})
    fitInd = trues(size(P,2));
    isfM0(fmfunc,fitnessP,fitInd,P,isf,momentn1,moment0,moment1,moment3,F_term_P1,F_term_P2,F_term_avg,mn1_term_P1,mn1_term_P2,m0_term_P1,m0_term_P2,m1_term_P1,m1_term_P2,m3_term_P1,m3_term_P2,isfmT_P1,mn1T_P1,m0T_P1,m1T_P1,m3T_P1,isfmT_P2,mn1T_P2,m0T_P2,m1T_P2,m3T_P2)
end

function isfM0(fmfunc::Function,fitnessP::Array{Float64,1},fitInd::BitArray{1},P::Array{Float64,2},isf::Array{Float64,1},momentn1::Float64,moment0::Float64,moment1::Float64,moment3::Float64,F_term_P1::Array{Float64,2},F_term_P2::Array{Float64,2},F_term_avg::Array{Float64,1},mn1_term_P1::Array{Float64,1},mn1_term_P2::Array{Float64,1},m0_term_P1::Array{Float64,1},m0_term_P2::Array{Float64,1},m1_term_P1::Array{Float64,1},m1_term_P2::Array{Float64,1},m3_term_P1::Array{Float64,1},m3_term_P2::Array{Float64,1},isfmT_P1::Array{Float64,2},mn1T_P1::Array{Float64,1},m0T_P1::Array{Float64,1},m1T_P1::Array{Float64,1},m3T_P1::Array{Float64,1},isfmT_P2::Array{Float64,2},mn1T_P2::Array{Float64,1},m0T_P2::Array{Float64,1},m1T_P2::Array{Float64,1},m3T_P2::Array{Float64,1})
    mul!(F_term_P1,isfmT_P1,P[1:end-1,:]);
    mul!(F_term_P2,isfmT_P2,P[2:end,:]);
    mean!(F_term_avg,abs.(isf .- (F_term_P1 .+ F_term_P2))');
    mul!(m0_term_P1,P[1:end-1,:]',m0T_P1);
    mul!(m0_term_P2,P[2:end,:]',m0T_P2);
    f1 = 0.0;
    f2 = momentn1;
    f3 = 0.0;
    f4 = moment1;
    f5 = moment3;
    @inbounds for j = 1:size(fitInd,1)
        if fitInd[j]
            f1 = F_term_avg[j];
            f3 = m0_term_P1[j]+m0_term_P2[j];
            fitnessP[j] = fmfunc(f1,f2,f3,f4,f5,momentn1,moment0,moment1,moment3); #testnewfit4
        end
    end
    nothing
end

function isfMn1M0(fmfunc::Function,fitnessP::Array{Float64,1},P::Array{Float64,2},isf::Array{Float64,1},momentn1::Float64,moment0::Float64,moment1::Float64,moment3::Float64,F_term_P1::Array{Float64,2},F_term_P2::Array{Float64,2},F_term_avg::Array{Float64,1},mn1_term_P1::Array{Float64,1},mn1_term_P2::Array{Float64,1},m0_term_P1::Array{Float64,1},m0_term_P2::Array{Float64,1},m1_term_P1::Array{Float64,1},m1_term_P2::Array{Float64,1},m3_term_P1::Array{Float64,1},m3_term_P2::Array{Float64,1},isfmT_P1::Array{Float64,2},mn1T_P1::Array{Float64,1},m0T_P1::Array{Float64,1},m1T_P1::Array{Float64,1},m3T_P1::Array{Float64,1},isfmT_P2::Array{Float64,2},mn1T_P2::Array{Float64,1},m0T_P2::Array{Float64,1},m1T_P2::Array{Float64,1},m3T_P2::Array{Float64,1})
    fitInd = trues(size(P,2));
    isfMn1M0(fmfunc,fitnessP,fitInd,P,isf,momentn1,moment0,moment1,moment3,F_term_P1,F_term_P2,F_term_avg,mn1_term_P1,mn1_term_P2,m0_term_P1,m0_term_P2,m1_term_P1,m1_term_P2,m3_term_P1,m3_term_P2,isfmT_P1,mn1T_P1,m0T_P1,m1T_P1,m3T_P1,isfmT_P2,mn1T_P2,m0T_P2,m1T_P2,m3T_P2)
end

function isfMn1M0(fmfunc::Function,fitnessP::Array{Float64,1},fitInd::BitArray{1},P::Array{Float64,2},isf::Array{Float64,1},momentn1::Float64,moment0::Float64,moment1::Float64,moment3::Float64,F_term_P1::Array{Float64,2},F_term_P2::Array{Float64,2},F_term_avg::Array{Float64,1},mn1_term_P1::Array{Float64,1},mn1_term_P2::Array{Float64,1},m0_term_P1::Array{Float64,1},m0_term_P2::Array{Float64,1},m1_term_P1::Array{Float64,1},m1_term_P2::Array{Float64,1},m3_term_P1::Array{Float64,1},m3_term_P2::Array{Float64,1},isfmT_P1::Array{Float64,2},mn1T_P1::Array{Float64,1},m0T_P1::Array{Float64,1},m1T_P1::Array{Float64,1},m3T_P1::Array{Float64,1},isfmT_P2::Array{Float64,2},mn1T_P2::Array{Float64,1},m0T_P2::Array{Float64,1},m1T_P2::Array{Float64,1},m3T_P2::Array{Float64,1})
    mul!(F_term_P1,isfmT_P1,P[1:end-1,:]);
    mul!(F_term_P2,isfmT_P2,P[2:end,:]);
    mean!(F_term_avg,abs.(isf .- (F_term_P1 .+ F_term_P2))');
    mul!(mn1_term_P1,P[1:end-1,:]',mn1T_P1);
    mul!(mn1_term_P2,P[2:end,:]',mn1T_P2);
    mul!(m0_term_P1,P[1:end-1,:]',m0T_P1);
    mul!(m0_term_P2,P[2:end,:]',m0T_P2);
    f1 = 0.0;
    f2 = 0.0;
    f3 = 0.0;
    f4 = moment1;
    f5 = moment3;
    @inbounds for j = 1:size(fitInd,1)
        if fitInd[j]
            f1 = F_term_avg[j];
            f2 = mn1_term_P1[j]+mn1_term_P2[j];
            f3 = m0_term_P1[j]+m0_term_P2[j];
            fitnessP[j] = fmfunc(f1,f2,f3,f4,f5,momentn1,moment0,moment1,moment3); #testnewfit4
        end
    end
    nothing
end

function isfMn1M0M1(fmfunc::Function,fitnessP::Array{Float64,1},P::Array{Float64,2},isf::Array{Float64,1},momentn1::Float64,moment0::Float64,moment1::Float64,moment3::Float64,F_term_P1::Array{Float64,2},F_term_P2::Array{Float64,2},F_term_avg::Array{Float64,1},mn1_term_P1::Array{Float64,1},mn1_term_P2::Array{Float64,1},m0_term_P1::Array{Float64,1},m0_term_P2::Array{Float64,1},m1_term_P1::Array{Float64,1},m1_term_P2::Array{Float64,1},m3_term_P1::Array{Float64,1},m3_term_P2::Array{Float64,1},isfmT_P1::Array{Float64,2},mn1T_P1::Array{Float64,1},m0T_P1::Array{Float64,1},m1T_P1::Array{Float64,1},m3T_P1::Array{Float64,1},isfmT_P2::Array{Float64,2},mn1T_P2::Array{Float64,1},m0T_P2::Array{Float64,1},m1T_P2::Array{Float64,1},m3T_P2::Array{Float64,1})
    fitInd = trues(size(P,2));
    isfMn1M0M1(fmfunc,fitnessP,fitInd,P,isf,momentn1,moment0,moment1,moment3,F_term_P1,F_term_P2,F_term_avg,mn1_term_P1,mn1_term_P2,m0_term_P1,m0_term_P2,m1_term_P1,m1_term_P2,m3_term_P1,m3_term_P2,isfmT_P1,mn1T_P1,m0T_P1,m1T_P1,m3T_P1,isfmT_P2,mn1T_P2,m0T_P2,m1T_P2,m3T_P2)
end

function isfMn1M0M1(fmfunc::Function,fitnessP::Array{Float64,1},fitInd::BitArray{1},P::Array{Float64,2},isf::Array{Float64,1},momentn1::Float64,moment0::Float64,moment1::Float64,moment3::Float64,F_term_P1::Array{Float64,2},F_term_P2::Array{Float64,2},F_term_avg::Array{Float64,1},mn1_term_P1::Array{Float64,1},mn1_term_P2::Array{Float64,1},m0_term_P1::Array{Float64,1},m0_term_P2::Array{Float64,1},m1_term_P1::Array{Float64,1},m1_term_P2::Array{Float64,1},m3_term_P1::Array{Float64,1},m3_term_P2::Array{Float64,1},isfmT_P1::Array{Float64,2},mn1T_P1::Array{Float64,1},m0T_P1::Array{Float64,1},m1T_P1::Array{Float64,1},m3T_P1::Array{Float64,1},isfmT_P2::Array{Float64,2},mn1T_P2::Array{Float64,1},m0T_P2::Array{Float64,1},m1T_P2::Array{Float64,1},m3T_P2::Array{Float64,1})
    mul!(F_term_P1,isfmT_P1,P[1:end-1,:]);
    mul!(F_term_P2,isfmT_P2,P[2:end,:]);
    mean!(F_term_avg,abs.(isf .- (F_term_P1 .+ F_term_P2))');
    mul!(mn1_term_P1,P[1:end-1,:]',mn1T_P1);
    mul!(mn1_term_P2,P[2:end,:]',mn1T_P2);
    mul!(m0_term_P1,P[1:end-1,:]',m0T_P1);
    mul!(m0_term_P2,P[2:end,:]',m0T_P2);
    mul!(m1_term_P1,P[1:end-1,:]',m1T_P1);
    mul!(m1_term_P2,P[2:end,:]',m1T_P2);
    f1 = 0.0;
    f2 = 0.0;
    f3 = 0.0;
    f4 = 0.0;
    f5 = moment3;
    @inbounds for j = 1:size(fitInd,1)
        if fitInd[j]
            f1 = F_term_avg[j];
            f2 = mn1_term_P1[j]+mn1_term_P2[j];
            f3 = m0_term_P1[j]+m0_term_P2[j];
            f4 = m1_term_P1[j]+m1_term_P2[j];
            fitnessP[j] = fmfunc(f1,f2,f3,f4,f5,momentn1,moment0,moment1,moment3); #testnewfit4
        end
    end
    nothing
end

function isfMn1M0M1M3(fmfunc::Function,fitnessP::Array{Float64,1},P::Array{Float64,2},isf::Array{Float64,1},momentn1::Float64,moment0::Float64,moment1::Float64,moment3::Float64,F_term_P1::Array{Float64,2},F_term_P2::Array{Float64,2},F_term_avg::Array{Float64,1},mn1_term_P1::Array{Float64,1},mn1_term_P2::Array{Float64,1},m0_term_P1::Array{Float64,1},m0_term_P2::Array{Float64,1},m1_term_P1::Array{Float64,1},m1_term_P2::Array{Float64,1},m3_term_P1::Array{Float64,1},m3_term_P2::Array{Float64,1},isfmT_P1::Array{Float64,2},mn1T_P1::Array{Float64,1},m0T_P1::Array{Float64,1},m1T_P1::Array{Float64,1},m3T_P1::Array{Float64,1},isfmT_P2::Array{Float64,2},mn1T_P2::Array{Float64,1},m0T_P2::Array{Float64,1},m1T_P2::Array{Float64,1},m3T_P2::Array{Float64,1})
    fitInd = trues(size(P,2));
    isfMn1M0M1M3(fmfunc,fitnessP,fitInd,P,isf,momentn1,moment0,moment1,moment3,F_term_P1,F_term_P2,F_term_avg,mn1_term_P1,mn1_term_P2,m0_term_P1,m0_term_P2,m1_term_P1,m1_term_P2,m3_term_P1,m3_term_P2,isfmT_P1,mn1T_P1,m0T_P1,m1T_P1,m3T_P1,isfmT_P2,mn1T_P2,m0T_P2,m1T_P2,m3T_P2)
end

function isfMn1M0M1M3(fmfunc::Function,fitnessP::Array{Float64,1},fitInd::BitArray{1},P::Array{Float64,2},isf::Array{Float64,1},momentn1::Float64,moment0::Float64,moment1::Float64,moment3::Float64,F_term_P1::Array{Float64,2},F_term_P2::Array{Float64,2},F_term_avg::Array{Float64,1},mn1_term_P1::Array{Float64,1},mn1_term_P2::Array{Float64,1},m0_term_P1::Array{Float64,1},m0_term_P2::Array{Float64,1},m1_term_P1::Array{Float64,1},m1_term_P2::Array{Float64,1},m3_term_P1::Array{Float64,1},m3_term_P2::Array{Float64,1},isfmT_P1::Array{Float64,2},mn1T_P1::Array{Float64,1},m0T_P1::Array{Float64,1},m1T_P1::Array{Float64,1},m3T_P1::Array{Float64,1},isfmT_P2::Array{Float64,2},mn1T_P2::Array{Float64,1},m0T_P2::Array{Float64,1},m1T_P2::Array{Float64,1},m3T_P2::Array{Float64,1})
    mul!(F_term_P1,isfmT_P1,P[1:end-1,:]);
    mul!(F_term_P2,isfmT_P2,P[2:end,:]);
    mean!(F_term_avg,abs.(isf .- (F_term_P1 .+ F_term_P2))');
    mul!(mn1_term_P1,P[1:end-1,:]',mn1T_P1);
    mul!(mn1_term_P2,P[2:end,:]',mn1T_P2);
    mul!(m0_term_P1,P[1:end-1,:]',m0T_P1);
    mul!(m0_term_P2,P[2:end,:]',m0T_P2);
    mul!(m1_term_P1,P[1:end-1,:]',m1T_P1);
    mul!(m1_term_P2,P[2:end,:]',m1T_P2);
    mul!(m3_term_P1,P[1:end-1,:]',m3T_P1);
    mul!(m3_term_P2,P[2:end,:]',m3T_P2);
    f1 = 0.0;
    f2 = 0.0;
    f3 = 0.0;
    f4 = 0.0;
    f5 = 0.0;
    @inbounds for j = 1:size(fitInd,1)
        if fitInd[j]
            f1 = F_term_avg[j];
            f2 = mn1_term_P1[j]+mn1_term_P2[j];
            f3 = m0_term_P1[j]+m0_term_P2[j];
            f4 = m1_term_P1[j]+m1_term_P2[j];
            f5 = m3_term_P1[j]+m3_term_P2[j];
            fitnessP[j] = fmfunc(f1,f2,f3,f4,f5,momentn1,moment0,moment1,moment3); #testnewfit4
        end
    end
    nothing
end

function Mn1M0M1(fmfunc::Function,fitnessP::Array{Float64,1},P::Array{Float64,2},isf::Array{Float64,1},momentn1::Float64,moment0::Float64,moment1::Float64,moment3::Float64,F_term_P1::Array{Float64,2},F_term_P2::Array{Float64,2},F_term_avg::Array{Float64,1},mn1_term_P1::Array{Float64,1},mn1_term_P2::Array{Float64,1},m0_term_P1::Array{Float64,1},m0_term_P2::Array{Float64,1},m1_term_P1::Array{Float64,1},m1_term_P2::Array{Float64,1},m3_term_P1::Array{Float64,1},m3_term_P2::Array{Float64,1},isfmT_P1::Array{Float64,2},mn1T_P1::Array{Float64,1},m0T_P1::Array{Float64,1},m1T_P1::Array{Float64,1},m3T_P1::Array{Float64,1},isfmT_P2::Array{Float64,2},mn1T_P2::Array{Float64,1},m0T_P2::Array{Float64,1},m1T_P2::Array{Float64,1},m3T_P2::Array{Float64,1})
    fitInd = trues(size(P,2));
    Mn1M0M1(fmfunc,fitnessP,fitInd,P,isf,momentn1,moment0,moment1,moment3,F_term_P1,F_term_P2,F_term_avg,mn1_term_P1,mn1_term_P2,m0_term_P1,m0_term_P2,m1_term_P1,m1_term_P2,m3_term_P1,m3_term_P2,isfmT_P1,mn1T_P1,m0T_P1,m1T_P1,m3T_P1,isfmT_P2,mn1T_P2,m0T_P2,m1T_P2,m3T_P2)
end

function Mn1M0M1(fmfunc::Function,fitnessP::Array{Float64,1},fitInd::BitArray{1},P::Array{Float64,2},isf::Array{Float64,1},momentn1::Float64,moment0::Float64,moment1::Float64,moment3::Float64,F_term_P1::Array{Float64,2},F_term_P2::Array{Float64,2},F_term_avg::Array{Float64,1},mn1_term_P1::Array{Float64,1},mn1_term_P2::Array{Float64,1},m0_term_P1::Array{Float64,1},m0_term_P2::Array{Float64,1},m1_term_P1::Array{Float64,1},m1_term_P2::Array{Float64,1},m3_term_P1::Array{Float64,1},m3_term_P2::Array{Float64,1},isfmT_P1::Array{Float64,2},mn1T_P1::Array{Float64,1},m0T_P1::Array{Float64,1},m1T_P1::Array{Float64,1},m3T_P1::Array{Float64,1},isfmT_P2::Array{Float64,2},mn1T_P2::Array{Float64,1},m0T_P2::Array{Float64,1},m1T_P2::Array{Float64,1},m3T_P2::Array{Float64,1})
    mul!(mn1_term_P1,P[1:end-1,:]',mn1T_P1);
    mul!(mn1_term_P2,P[2:end,:]',mn1T_P2);
    mul!(m0_term_P1,P[1:end-1,:]',m0T_P1);
    mul!(m0_term_P2,P[2:end,:]',m0T_P2);
    mul!(m1_term_P1,P[1:end-1,:]',m1T_P1);
    mul!(m1_term_P2,P[2:end,:]',m1T_P2);
    f1 = 0.0;
    f2 = 0.0;
    f3 = 0.0;
    f4 = 0.0;
    f5 = moment3;
    @inbounds for j = 1:size(fitInd,1)
        if fitInd[j]
            f2 = mn1_term_P1[j]+mn1_term_P2[j];
            f3 = m0_term_P1[j]+m0_term_P2[j];
            f4 = m1_term_P1[j]+m1_term_P2[j];
            fitnessP[j] = fmfunc(f1,f2,f3,f4,f5,momentn1,moment0,moment1,moment3); #testnewfit4
        end
    end
end

function Mn1M0M1M3(fmfunc::Function,fitnessP::Array{Float64,1},P::Array{Float64,2},isf::Array{Float64,1},momentn1::Float64,moment0::Float64,moment1::Float64,moment3::Float64,F_term_P1::Array{Float64,2},F_term_P2::Array{Float64,2},F_term_avg::Array{Float64,1},mn1_term_P1::Array{Float64,1},mn1_term_P2::Array{Float64,1},m0_term_P1::Array{Float64,1},m0_term_P2::Array{Float64,1},m1_term_P1::Array{Float64,1},m1_term_P2::Array{Float64,1},m3_term_P1::Array{Float64,1},m3_term_P2::Array{Float64,1},isfmT_P1::Array{Float64,2},mn1T_P1::Array{Float64,1},m0T_P1::Array{Float64,1},m1T_P1::Array{Float64,1},m3T_P1::Array{Float64,1},isfmT_P2::Array{Float64,2},mn1T_P2::Array{Float64,1},m0T_P2::Array{Float64,1},m1T_P2::Array{Float64,1},m3T_P2::Array{Float64,1})
    fitInd = trues(size(P,2));
    Mn1M0M1M3(fmfunc,fitnessP,fitInd,P,isf,momentn1,moment0,moment1,moment3,F_term_P1,F_term_P2,F_term_avg,mn1_term_P1,mn1_term_P2,m0_term_P1,m0_term_P2,m1_term_P1,m1_term_P2,m3_term_P1,m3_term_P2,isfmT_P1,mn1T_P1,m0T_P1,m1T_P1,m3T_P1,isfmT_P2,mn1T_P2,m0T_P2,m1T_P2,m3T_P2)
end

function Mn1M0M1M3(fmfunc::Function,fitnessP::Array{Float64,1},fitInd::BitArray{1},P::Array{Float64,2},isf::Array{Float64,1},momentn1::Float64,moment0::Float64,moment1::Float64,moment3::Float64,F_term_P1::Array{Float64,2},F_term_P2::Array{Float64,2},F_term_avg::Array{Float64,1},mn1_term_P1::Array{Float64,1},mn1_term_P2::Array{Float64,1},m0_term_P1::Array{Float64,1},m0_term_P2::Array{Float64,1},m1_term_P1::Array{Float64,1},m1_term_P2::Array{Float64,1},m3_term_P1::Array{Float64,1},m3_term_P2::Array{Float64,1},isfmT_P1::Array{Float64,2},mn1T_P1::Array{Float64,1},m0T_P1::Array{Float64,1},m1T_P1::Array{Float64,1},m3T_P1::Array{Float64,1},isfmT_P2::Array{Float64,2},mn1T_P2::Array{Float64,1},m0T_P2::Array{Float64,1},m1T_P2::Array{Float64,1},m3T_P2::Array{Float64,1})
    mul!(mn1_term_P1,P[1:end-1,:]',mn1T_P1);
    mul!(mn1_term_P2,P[2:end,:]',mn1T_P2);
    mul!(m0_term_P1,P[1:end-1,:]',m0T_P1);
    mul!(m0_term_P2,P[2:end,:]',m0T_P2);
    mul!(m1_term_P1,P[1:end-1,:]',m1T_P1);
    mul!(m1_term_P2,P[2:end,:]',m1T_P2);
    mul!(m3_term_P1,P[1:end-1,:]',m3T_P1);
    mul!(m3_term_P2,P[2:end,:]',m3T_P2);
    f1 = 0.0;
    f2 = 0.0;
    f3 = 0.0;
    f4 = 0.0;
    f5 = 0.0;
    @inbounds for j = 1:size(fitInd,1)
        if fitInd[j]
            f2 = mn1_term_P1[j]+mn1_term_P2[j];
            f3 = m0_term_P1[j]+m0_term_P2[j];
            f4 = m1_term_P1[j]+m1_term_P2[j];
            f5 = m3_term_P1[j]+m3_term_P2[j];
            fitnessP[j] = fmfunc(f1,f2,f3,f4,f5,momentn1,moment0,moment1,moment3); #testnewfit4
        end
    end
    nothing
end
end

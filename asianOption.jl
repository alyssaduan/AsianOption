using Statistics;
using Sobol;
using Random;
include("PC.jl")

function AsianPricing_Sobol(numofPath, numofSteps, numofRuns, Sini, r, sigma, T, K)
    dt = T/numofSteps
    sobol_result = Array{Float64}(undef, numofRuns)
    sobol = SobolSeq(numofSteps);
    skip_pts= reduce(hcat, next!(sobol) for i = 1:numofPath)' # skip the first 100 points 
    for kk in 1:numofRuns
        call_price = 0
        Z = reduce(hcat, next!(sobol) for i = 1:numofPath)'
        for ii in 1:numofPath
            Ssum = Sini
            S = Sini
            for jj in 1:numofSteps
                S = S*exp((r-sigma^2/2)*dt + sigma*sqrt(dt)*invNormal(Z[ii,jj]))
                Ssum = Ssum + S
            end
            call_price += max(Ssum/(numofSteps+1) - K, 0)
        end
        sobol_result[kk] = exp(-r*T)*call_price/numofPath
    end
    sobol_result
end

function AsianPricing_MT(numofPath, numofSteps, numofRuns, Sini, r, sigma, T, K)
    dt = T/numofSteps
    mt_result = Array{Float64}(undef, numofRuns)
    for kk in 1:numofRuns
        call_price = 0
        for ii in 1:numofPath
            Ssum = Sini
            S = Sini
            for jj in 1:numofSteps
                S = S*exp((r-sigma^2/2)*dt + sigma*sqrt(dt)*invNormal(rand()))
                Ssum = Ssum + S
            end
            call_price += max(Ssum/(numofSteps+1) - K, 0)
        end
        mt_result[kk] = exp(-r*T)*call_price/numofPath
    end
    mt_result
end

function AsianPricing_MT_PC(numofPath, numofSteps, numofRuns, Sini, r, sigma, T, K)
    dt = T/numofSteps
    mt_result = Array{Float64}(undef, numofRuns)
    for kk in 1:numofRuns
        call_price = 0.
        BM_path = PCconst_MT(numofPath, numofSteps, dt)
        for ii in 1:numofPath #256
            Ssum = Sini
            S = Sini
            for jj in 1:numofSteps #20
                S = S*exp((r-sigma^2/2)*dt + sigma*(BM_path[jj+1,ii] - BM_path[jj,ii]))
                Ssum = Ssum + S
            end
            call_price += max(Ssum/(numofSteps+1) - K, 0)
        end
        mt_result[kk] = exp(-r*T)*call_price/numofPath
    end
    mt_result
end

function AsianPricing_Sobol_PC(numofPath, numofSteps, numofRuns, Sini, r, sigma, T, K)
    dt = T/numofSteps
    sobol_result = Array{Float64}(undef, numofRuns)
    sobol = SobolSeq(numofSteps);
    skip_pts= reduce(hcat, next!(sobol) for i = 1:numofPath)' 
    for kk in 1:numofRuns
        call_price = 0.
        BM_path = PCconst_Sobol(numofPath, numofSteps, dt, sobol)
        for ii in 1:numofPath #256
            Ssum = Sini
            S = Sini
            for jj in 1:numofSteps #20
                S = S*exp((r-sigma^2/2)*dt + sigma*(BM_path[jj+1,ii] - BM_path[jj,ii]))
                Ssum = Ssum + S
            end
            call_price += max(Ssum/(numofSteps+1) - K, 0)
        end
        sobol_result[kk] = exp(-r*T)*call_price/numofPath
    end
    sobol_result
end
using Sobol, Plots, Statistics;
using StatsPlots,DataFrames;
using Distributions, Random;
include("asianOption.jl")

function RMSE(exact)
end
"""
initialize parameters
"""
numofSteps = 20
numofPath = 256
numofRuns = 50
K = 90
r = 0.035
sigma = 0.2
T = 1
Sini = 100
exact_price = 12.0987254 # results for running N = 500,000 pathes
# BM_path = PCconst_MT(numofSteps, numofRuns,dt)
_mt = Array{Float64}(undef, numofRuns)
_sobol = Array{Float64}(undef, numofRuns)
_mt_PC = Array{Float64}(undef, numofRuns)
_sobol_PC = Array{Float64}(undef, numofRuns)
# _pc_mt = Array{Float64}(undef, numofRuns)
_mt = AsianPricing_MT(numofPath, numofSteps, numofRuns, Sini, r, sigma, T, K)
_sobol = AsianPricing_Sobol(numofPath, numofSteps, numofRuns, Sini, r, sigma, T, K)
_mt_PC = AsianPricing_MT_PC(numofPath, numofSteps, numofRuns, Sini, r, sigma, T, K)
_sobol_PC = AsianPricing_Sobol_PC(numofPath, numofSteps, numofRuns, Sini, r, sigma, T, K)
# _pc_mt = AsianPricing_MT_PC(numofPath,numofSteps, numofRuns, BM_path)

All_result = Array{Float64,2}(undef, (numofRuns, 4)) # 4 different Methods
All_result[:,1] = _mt
All_result[:,2] = _sobol
All_result[:,3] = _mt_PC
All_result[:,4] = _sobol_PC


df = DataFrame(All_result, :auto)
function write_result_csv()
    path = "All_result.csv"
    CSV.write(path, df)
end
# boxplot!(["MT", "Sobol", "MT+PC", "Sobol+PC"], All_result, leg=false)
# boxplot(All_result)
# legend(["MT", "Sobol", "MT+PC", "Sobol+PC"])
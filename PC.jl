using LinearAlgebra;
using Statistics, Distributions;


# convert to normal distribution using Moro algorithm
function invNormal(u::Float64)
    # Beasley-Springer-Moro algorithm
    a0=2.50662823884
    a1=-18.61500062529
    a2=41.39119773534
    a3=-25.44106049637
    b0=-8.47351093090
    b1=23.08336743743
    b2=-21.06224101826
    b3=3.13082909833
    c0=0.3374754822726147
    c1=0.9761690190917186
    c2=0.1607979714918209
    c3=0.0276438810333863
    c4=0.0038405729373609
    c5=0.0003951896511919
    c6=0.0000321767881768
    c7=0.0000002888167364
    c8=0.0000003960315187
    y=u-0.5
    if abs(y)<0.42
        r=y*y
        x=y*(((a3*r+a2)*r+a1)*r+a0)/((((b3*r+b2)*r+b1)*r+b0)*r+1)
    else
        r=u
        if(y >0)
            r=1-u
        end
        r=log(-log(r))
        x=c0+r*(c1+r*(c2+r*(c3+r*(c4+r*(c5+r*(c6+r*(c7+r*c8)))))))
        if(y<0)
            x=-x
        end
    end
    return x
end

# generate Brownian Motion - PC construction
function PCconst_MT(numofPath, numofStep, dt)
    result_W = Array{Float64,2}(undef,numofStep+1, numofPath);
    Cmat = Array{Float64, 2}(undef, numofStep, numofStep) # covariance matrix
    eigenvalues = Array{Float64,1}(undef, numofStep);
    eigenvectors = Array{Float64,2}(undef, numofStep, numofStep)
    for row in 1:numofStep
        for col in 1:numofStep
            Cmat[row,col] = min(row*dt, col*dt)
        end
    end 
    eigenvalues = eigvals(Cmat);
    eigenvectors = eigvecs(Cmat);
    for kk in 1:numofPath
        W = Array{Float64}(undef,numofStep+1);
        W[1] = 0.0
        for jj in 2:numofStep
            W[jj] = 0.0;
            for ii in 1:numofStep
                W[jj] += sqrt(eigenvalues[numofStep - ii + 1]) * invNormal(rand()) .* eigenvectors[jj-1,numofStep - ii + 1]
            end
        end
        result_W[:,kk] = W
    end
    result_W
end


function PCconst_Sobol(numofPath, numofStep, dt,sobol)
    result_W = Array{Float64,2}(undef,numofStep+1, numofPath);
    Cmat = Array{Float64, 2}(undef, numofStep, numofStep) # covariance matrix
    eigenvalues = Array{Float64,1}(undef, numofStep);
    eigenvectors = Array{Float64,2}(undef, numofStep, numofStep)
    for row in 1:numofStep
        for col in 1:numofStep
            Cmat[row,col] = min(row*dt, col*dt)
        end
    end 
    eigenvalues = eigvals(Cmat);
    eigenvectors = eigvecs(Cmat);
    for kk in 1:numofPath
        Z = reduce(hcat, next!(sobol) for i = 1:numofPath)'
        W = Array{Float64}(undef,numofStep+1);
        W[1] = 0.0
        for jj in 2:numofStep+1
            W[jj] = 0.0;
            for ii in 1:numofStep
                W[jj] += sqrt(eigenvalues[numofStep - ii + 1]) * invNormal(Z[ii,jj-1]) .* eigenvectors[jj-1,numofStep - ii + 1]
            end
        end
        result_W[:,kk] = W
    end
    result_W
end

# numofStep = 160; # 40 years x 4 month/year = 160
# numofRuns = 2
# # generate Sobol' sequence
# sobol = SobolSeq(dimofBM);
# skip_pts= reduce(hcat, next!(sobol) for i = 1:160)' # skip the first 100 points 
# Z = reduce(hcat, next!(sobol) for i = 1:2048)'
# # Z = rand(Float64, (993, 160)) # Mersenne Twister generator
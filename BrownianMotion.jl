using Sobol;
using PyPlot;
using LinearAlgebra;
using Statistics;
using Distributions;
using Random;

#make plot for eigenvalues
function plotEigenvalues(eigenvalues)
    timesteps = Array{Float64, 1}(undef, numofStep);
    for i in 1:numofStep
        timesteps[i] = i+1
    end
    subplot(212)
    subplot(211)
    sub_timesteps =[1,2,3]
    bar(sub_timesteps, reverse(eigenvalues)[1:3],align="center",alpha=0.5)
    subplot(212)
    sub_timesteps1 =[4,5,6,7,8]
    bar(sub_timesteps1, reverse(eigenvalues)[4:8],align="center",alpha=0.5)
    suptitle("eigenvalues of covariance matrix")
end

# make plot of the eigenvectors 
function plotEigenvec(eigenvectors)
    subplot(224)
    fig = figure("pyplot_subplot_mixed",figsize=(8,8)) # Create a new blank figure
    subplot(221) # Create the 1st axis of a 2x2 arrax of axes
    plot(eigenvectors[:,dimofBM])

    subplot(222) # Create a plot 
    plot(eigenvectors[:,dimofBM-1])

    ax = subplot(223) # Create a plot and make it a polar plot, 3rd axis of 2x2 axis grid
    ax.plot(eigenvectors[:,dimofBM-2])

    subplot(224) # Create the 4th axis of a 2x2 arrax of axes
    plot(eigenvectors[:,dimofBM-3])
    suptitle("First Four eigenvectors of covariance matrix")
end

function Get_W(W::Array{Float64}, Z::Array{Float64}, n::Int, W1::Float64, Wn::Float64)
    W[1] = W1;
    W[n] = Wn;
    Z_ind::Int = 1;
    Left::Array{Int} = Array{Int}(undef, n);
    Right::Array{Int} = Array{Int}(undef, n);

    cur_index::Int = 1;
    pair_num::Int = 1;
    mid::Int = 0;

    Left[1] = 1;
    Right[1] = n;

    cl::Int = 0;
    cr::Int = 0;

    while cur_index <= pair_num
        cl = Left[cur_index];
        cr = Right[cur_index];
        mid = div(cl + cr, 2); 

        if mid - cl > 1
            Left[pair_num + 1] = cl;
            Right[pair_num + 1] = mid;
            pair_num = pair_num + 1;
        end
        if cr - mid > 1
            Left[pair_num + 1] = mid;
            Right[pair_num + 1] = cr;
            pair_num = pair_num + 1;
        end
        cur_index = cur_index + 1
        # at the first loop, pair num goes from 1 to 3, cur_index goes from 1 to 2. 
    end
    for ii = 1:pair_num
        # println(Left[ii], "\t", Right[ii], "\t", div(Left[ii]+Right[ii],2))
        cl = Left[ii];
        cr = Right[ii];
        mid = div(cl + cr, 2); 
        W[mid] = Get_W_step(cl,cr, mid, W[cl], W[cr], invNormal(Z[ii]))
    end
end

function Get_W_step(l::Int, r::Int, m::Int, Wl::Float64, Wr::Float64, Zi::Float64)
    var::Float64 = 0.0
    mu::Float64 = 0.0
    mu = ((r-m)*Wl + (m-l)*Wr)/(r-l)
    var = ((m-l)*(r-m))/(r-l)
    return mu + sqrt(var) * Zi
end

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

function conveNormal(Z)
    row  =size(Z)[1]
    col = size(Z)[2]
    normedZ = Array{Float64, 2}(undef, row,col)
    for i in 1:row
        for j in 1:col
            normedZ[i,j] = invNormal(Z[i,j])
        end
    end
    return normedZ
end

# generate Brownian Motion - standard path generation
function standardPath(Z)
    myZ = conveNormal(Z)
    result_W = Array{Float64,2}(undef,dimofBM, numofRepeats);
    # Wmat = Array{Float64,2}(undef, dimofBM,dimofBM);
    # for i in 1:dimofBM
    #     for j in 1:dimofBM
    #         if i>=j
    #             Wmat[i,j] = sqrt(dt)
    #         else
    #             Wmat[i,j] = 0
    #         end
    #     end
    # end
    for j in 1:numofRepeats
        W = Array{Float64, numofFactor}(undef, dimofBM);
        W[1] = 0.0 # initial BM=3
        for i in 2:dimofBM
            # W[i] = dot(Wmat[i,:], myZ[32+j-1,:])
        end
        plot(W)
        PyPlot.title("Standard Path Generation")
        # PyPlot.ylim([-90, 90])
        result_W[:,j] = W
    end
    return result_W
end

# generate Brownian Motion - Brownian bridge
function BB(Z)
    result_W = Array{Float64,2}(undef,dimofBM, numofRepeats);
    for jj in 1:numofRepeats
        W1 = 0.0;
        Wn = sqrt(numofStep) * invNormal(Z[320+jj-1,1])
        W = Array{Float64}(undef, dimofBM)
        Get_W(W, Z[32+jj-1,2:160], numofStep, W1, Wn)
        plot(W)
        PyPlot.title("Brownian bridge Path Generation")
        result_W[:,jj] = W
    end
    return result_W
end

# generate Brownian Motion - PC construction
function PCconst(Z)
    result_W = Array{Float64,2}(undef,dimofBM, numofRepeats);
    # generate covariance matrix of BM
    Cmat = Array{Float64, 2}(undef, dimofBM, dimofBM) # covariance matrix
    eigenvalues = Array{Float64,1}(undef, dimofBM);
    eigenvectors = Array{Float64,2}(undef, dimofBM, dimofBM)
    for row in 1:dimofBM
        for col in 1:dimofBM
            Cmat[row,col] = min(row*dt, col*dt)
        end
    end 
    eigenvalues = eigvals(Cmat);
    eigenvectors = eigvecs(Cmat);
    for kk in 1:numofRepeats
        W = Array{Float64}(undef,dimofBM);
        for jj in 1:dimofBM
            W[jj] = 0.0;
            for ii in 1:dimofBM
                W[jj] += sqrt(eigenvalues[dimofBM - ii + 1]) * invNormal(Z[320+kk-1, ii]) .* eigenvectors[jj,dimofBM - ii + 1]
            end
        end
        plot(W)
        PyPlot.title("PC Path Generation")
        result_W[:,kk] = W
        # PyPlot.ylim([-90, 90])
    end
    return result_W
end

# generate Brownian Motion - PPC
function PPC(Zk)
    result_W = Array{Float64,2}(undef,dimofBM, numofRepeats);
    # generate covariance matrix of BM
    Cmat = Array{Float64, 2}(undef, dimofBM, dimofBM) # covariance matrix
    eigenvalues = Array{Float64,1}(undef, dimofBM);
    eigenvectors = Array{Float64,2}(undef, dimofBM, dimofBM)
    for row in 1:dimofBM
        for col in 1:dimofBM
            Cmat[row,col] = min(row*dt, col*dt)
        end
    end 
    eigenvalues = eigvals(Cmat);
    eigenvectors = eigvecs(Cmat);
 
    k = 4; # the number of eigenvalues would like to keep
    Amat = Array{Float64,2}(undef, dimofBM, dimofBM);
    Amat = factorize(Cmat) # need to change
 
    Imat=Diagonal(ones(dimofBM))
    Bmat = Array{Float64,2}(undef, dimofBM, dimofBM)
    Bmat = transpose(eigenvalues.*eigenvectors) # sort from small to large eigenvalues
    Bmat2 = Array{Float64, 2}(undef, dimofBM,dimofBM) # sort from large to small eigenvalues
    for row in 1:dimofBM
         for col in 1:dimofBM
             Bmat2[row,col] = Bmat[row, dimofBM-col+1]
         end
    end
    Umat = Array{Float64, 2}(undef, dimofBM,dimofBM)
    Umat = inv(Amat) * Bmat2[:,1:k]
    transUmat = transpose(Umat)
    finalMat = Array{Float64, 2}(undef, dimofBM, dimofBM)
    finalMat = inv(inv(Amat))*(Imat - Umat*transUmat)
    dimK::Int = 0
    dimK = dimofBM+k
    Z = Array{Float64,2}(undef, 65, dimK)
    for r in 1:65
        for c in 1:dimK
            Z[r,c] = invNormal(Zk[r,c])
        end
    end
    for kk in 1:numofRepeats
         finalMat2 = Array{Float64}(undef, dimofBM)
         finalMat2 = finalMat .* Z[32+kk-1,k+1:dimK]
         W = Array{Float64}(undef,dimofBM);
         for jj in 1:dimofBM # row
            W[jj] = 0.0;
            for ii in 1:k # col
                W[jj] += sqrt(eigenvalues[dimofBM - ii + 1]) * Z[320+kk-1, ii] .* eigenvectors[jj,dimofBM - ii + 1]
            end
            W[jj] += finalMat2[jj]
        end
        plot(W)
        PyPlot.title("PPC Path Generation")
        result_W[:,kk] = W
    end
    return result_W
 end

# W1 = 0.0;
# Wn = sqrt(numofStep) * invNormal(Z[32,1])
# Get_W(W, Z[32,2:160], numofStep, W1, Wn)

function Get_W_multi_know(W::Array{Float64}, Z::Array{Float64}, n::Int, known_indice::Array{Int}, known_values::Array{Float64}, known_num::Int)

    Left::Array{Int} = Array{Int}(undef, n);
    Right::Array{Int} = Array{Int}(undef, n);

    for ii = 1:known_num   
        W[known_indice[ii]] = known_values[ii] 
    end

    cur_index::Int = 1
    pair_num::Int = known_num - 1
    mid::Int = 0

    for ii = 1:(known_num - 1)
        Left[ii] = known_indice[ii];
        Right[ii] = known_indice[ii+1];
    end

    cl::Int = 0;
    cr::Int = 0;

    while cur_index <= pair_num
        cl = Left[cur_index];
        cr = Right[cur_index];
        mid = div(cl + cr, 2); 

        if mid - cl > 1
            Left[pair_num + 1] = cl;
            Right[pair_num + 1] = mid;
            pair_num = pair_num + 1;
        end
        if cr - mid > 1
            Left[pair_num + 1] = mid;
            Right[pair_num + 1] = cr;
            pair_num = pair_num + 1;
        end
        cur_index = cur_index + 1
        # at the first loop, pair num goes from 1 to 3, cur_index goes from 1 to 2. 
    end
    for ii = 1:pair_num
        # println(Left[ii], "\t", Right[ii], "\t", div(Left[ii]+Right[ii],2))
        cl = Left[ii];
        cr = Right[ii];
        mid = div(cl + cr, 2); 
        W[mid] = Get_W_step(cl,cr, mid, W[cl], W[cr], invNormal(Z[ii]))
    end
end

function getWsub(Z)
    sub_dim = 20;
    dt = 2;
    Cmat = Array{Float64, 2}(undef, sub_dim, sub_dim) # covariance matrix
    eigenvalues = Array{Float64,1}(undef, sub_dim);
    eigenvectors = Array{Float64,2}(undef, sub_dim, sub_dim)
    for row in 1:sub_dim
        for col in 1:sub_dim
            Cmat[row,col] = min(row*dt, col*dt)
        end
    end 
    eigenvalues = eigvals(Cmat);
    eigenvectors = eigvecs(Cmat);
    W = Array{Float64}(undef,sub_dim);
    for jj in 1:sub_dim
        W[jj] = 0.0;
        for ii in 1:sub_dim
            W[jj] += sqrt(eigenvalues[sub_dim - ii + 1]) * invNormal(Z[320, ii]) .* eigenvectors[jj,sub_dim - ii + 1]
        end
    end
    return W
end

function subseq(Z)
    sub_dim = 20;
    dt = 2;
    Cmat = Array{Float64, 2}(undef, sub_dim, sub_dim) # covariance matrix
    eigenvalues = Array{Float64,1}(undef, sub_dim);
    eigenvectors = Array{Float64,2}(undef, sub_dim, sub_dim)
    for row in 1:sub_dim
        for col in 1:sub_dim
            Cmat[row,col] = min(row*dt, col*dt)
        end
    end 
    eigenvalues = eigvals(Cmat);
    eigenvectors = eigvecs(Cmat);
    result_W = Array{Float64,2}(undef,dimofBM, numofRepeats);

    for kk in 1:numofRepeats
        # filled in the brownian motion which constructed using PC
        Wtemp = Array{Float64}(undef,sub_dim);
        for jj in 1:sub_dim
            Wtemp[jj] = 0.0;
            for ii in 1:sub_dim
                Wtemp[jj] += sqrt(eigenvalues[sub_dim - ii + 1]) * invNormal(Z[320+kk-1, ii]) .* eigenvectors[jj,sub_dim - ii + 1]
            end
        end 
        index_list = Array{Int64}(undef, sub_dim);
        for i in 1:sub_dim
            index_list[i] = Int(8*i)
        end

        W = Array{Float64}(undef, dimofBM);
        Get_W_multi_know(W, Z[32+kk-1,21:160], numofStep, pushfirst!(index_list,1), pushfirst!(Wtemp,0),sub_dim+1)
        plot(W)
        PyPlot.title("Subsequence Path Generation")
        result_W[:,kk] = W
    end
    return result_W
end

# call all functions
function analysis(Z, Zk)
    W1 = Array{Float64, 2}(undef, dimofBM, numofRepeats);
    W2 = Array{Float64, 2}(undef, dimofBM, numofRepeats);
    W3 = Array{Float64, 2}(undef, dimofBM, numofRepeats);
    W4 = Array{Float64, 2}(undef, dimofBM, numofRepeats);
    W5 = Array{Float64, 2}(undef, dimofBM, numofRepeats);

    W1 = standardPath(Z);
    W2 = BB(Z);
    W3 = PCconst(Z);
    W4 = PPC(Zk);
    W5 = subseq(Z);
end

function eigenratio(eigenvalues)
    total = sum(eigenvalues)
    ratio = Array{Float64}(undef, 8)
    for i in 1:8
        ratio[i] = eigenvalues[dimofBM - i + 1] / total
    end
    println(ratio)
    subplot(212)
    subplot(211)
    sub_timesteps =[1,2,3]
    bar(sub_timesteps, ratio[1:3],align="center",alpha=0.5)
    subplot(212)
    sub_timesteps1 =[4,5,6,7,8]
    bar(sub_timesteps1, ratio[4:8],align="center",alpha=0.5)
    suptitle("eigenvalues ratio of covariance matrix")
end

function calc_variance(ww)
    var_arr = Array{Float64}(undef,dimofBM)
    for i in 1:dimofBM
        var_arr[i] = var(ww[i,:])
    end
    # plot(var_arr, label="S Path Generation")
    plot(var_arr, label="Standard Path Generation")
    plot(time_step, variance_list, label="Brownian Motion Variance")
    # xlabel("time steps")
    # ylabel("Variance")
    # PyPlot.title("Variance of Standard Path Generation")
    # # PyPlot.title("Variance of Subsequence Path Generation")
    # legend(fancybox="true") # Create a legend of all the existing plots using their labels as names
    return var_arr
end


function comparison(w1,w2,w3,w4,w5)
    time_step = Array{Int64}(undef, dimofBM);
    variance_list = Array{Float64}(undef, dimofBM)
    dt = 0.25
    for i in 1:dimofBM
        time_step[i] = i
        variance_list[i] = dt*i
    end
    # plot(calc_variance(w1), label="Standard Path Generation")
    # plot(calc_variance(w2), label="BB Path Generation")
    plot(calc_variance(w3), label="PC Path Generation")
    plot(calc_variance(w4), label="PPC Path Generation")
    plot(calc_variance(w5), label="Subsequence Path Generation")
    plot(time_step, variance_list, label="Brownian Motion Variance")
    xlabel("time steps")
    ylabel("Variance")
    # PyPlot.title("Variance of Standard Path Generation")
    PyPlot.title("Variance Comparison of Path Generation Methods")
    legend(fancybox="true") # Create a legend of all the existing plots using their labels as names
end



# generate a multivariance brownian motion path

# sigmaMat = Array{Float64, 2}(undef, dimofBM, dimofBM) # covariance matrix
# for row in 1:dimofBM
#     for col in 1:dimofBM
#         sigmaMat[row,col] = min(row*dt, col*dt)
#     end
# end 
# sigmaMat
# means = fill(0,dimofBM)
# d = MvNormal(means, sigmaMat)
# xx = rand(d, 1)
# # println(xx)
# trace_A = tr(inv(inv(Amat)))
# println(trace_A)

# plain construction
# A1 = Array{Float64,2}(undef, dimofBM, dimofBM)
# for row in 1:dimofBM
#     for col in 1:dimofBM
#         if row >= col
#             A1[row,col] = dt
#         else
#             A1[row,col] = 0
#         end
#     end
# end

# function variExplain(A)
#     temp::Int = 0;
#     for i in 1:dimofBM
#         for j in 1:5
#             temp += A[i,j]
#         end
#     end
#     return temp
# end

# myZ = conveNormal(Z)
# res_W = Array{Float64,2}(undef, dimofBM, numofRepeats)
# for j in 1:numofRepeats
#     W = Array{Float64}(undef, dimofBM)
#     W[1] = 0
#     for i in 2:dimofBM
#         W[i] = W[i-1] + sqrt(dt) *  myZ[32+j-1,i-1]
#     end
#     res_W[:,j] = W
# end
# plot(res_W)
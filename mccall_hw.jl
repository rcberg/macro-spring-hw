using Pkg
ENV["GRDIR"]=""
Pkg.build("GR")
using Distributions
using LinearAlgebra
using Parameters
using Plots


@with_kw mutable struct dist_params
    mu::Float64 = 0
    sig::Float64=1
    N::Int64 = 40
end

para_dist = dist_params()

function approximateLogNormal(para_dist)
    @unpack mu, sig, N = para_dist
    wdist = LogNormal(mu,sig)
    wgrid = quantile.(wdist,LinRange(0,1,N+1))
    p = 1/N*ones(N)
    wbar = zeros(N)
    for i in 1:N-1
        wbar[i] = (wgrid[i]+wgrid[i+1])/2
    end
    wbar[N] = (exp(mu+0.5*sig^2)-dot(p[1:N-1],wbar[1:N-1]))/(1/N)
    return p,wbar
end

function mccallbellman(v,w,p,c,a,β)
    # Value of reject offer
    v_reject = c + β * dot(p,v)
    # Value of accepting the offer
    v_accept = w .+ β*(a*(v_reject) .+ (1-a)*v)

    #Compare
    v_n = max.(v_reject , v_accept)
    return v_n
end

function solvemccall(w,p,c,a,β,ϵ=1e-6)
    # "Initialize"
    v = zeros(length(w))
    diff = 1.
    # "Check if stopping criteria is reached"
    while diff > ϵ
        v_n = mccallbellman(v,w,p,c,a,β)
        # "Use supremum norm"
        diff = norm(v-v_n,Inf)
        v = v_n # Reset v
    end
    return v
end

function findResWageIndex(v)
  for i in 1:length(v)
    #test value of wage offer against value of next highest wage offer
    if v[i+1] > v[i]
      #if value function starts increasing after this wage, this wage
      #is the reservation wage.
      return i
    end
  end
  #if a reservation wage is not found, return 0: there is no wage so low
  #that it would not be accepted.
  return 0
end

p, w = approximateLogNormal(para_dist)
a = 0.03
β = 0.98
c = 3

vFinalB = solvemccall(w,p,c,a,β)

findfirst(vFinalB .> vFinalB[1])

#extract reservation wage
resWageIndex = findResWageIndex(vFinalB)

resWage = w[resWageIndex]

println("Reservation wage is ", resWage, ", at index ", resWageIndex)

#initialize multiple-reservation-wage vector as zeros
variedResWages = zeros(100)

#generate vector of alphas from 0 to 0.1
newa = LinRange(0,0.1,100)

#determine reservation wage for each alpha.
#value functions are not saved, only reservation wages.

for i in 1:100
  variedResWages[i] = w[findResWageIndex(solvemccall(w,p,c,newa[i],β))]
end

cAnswer = scatter(newa,variedResWages,color =:blue)
display(cAnswer)

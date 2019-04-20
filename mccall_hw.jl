using Pkg
ENV["GRDIR"]=""
Pkg.build("GR")
using Distributions
using LinearAlgebra
using Parameters
using Plots

# Will use μ = 0 , σ = 0.25 , N = 100

function approximatelognormal(μ,σ,N)
    wdist = LogNormal(μ,σ)
    wgrid = quantile.(wdist,LinRange(0,1,N+1))
    p = 1/N*ones(N)
    wbar = zeros(N)
    for i in 1:N-1
        wbar[i] = (wgrid[i]+wgrid[i+1])/2
    end
    wbar[N] = (exp(μ+0.5*σ^2)-dot(p[1:N-1],wbar[1:N-1]))/(1/N)
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

# This is about where part B starts

p, w = approximatelognormal(0,0.25,100)
a = 0.03
β = 0.98
c = 3

# I am going to just set up a simple simulation with plot to show this
vt = zeros(100,100)
vt[1,:] = mccallbellman(zeros(100),w,p,c,a,β)
for j in 2:100
    vt[j,:] = mccallbellman(vt[j-1,:],w,p,c,a,β)
end
plot(w,zeros(100)) # Plots a baseline
plot!(w,vt[100,:]) # Plots the final iteration of the value function (can play around with this number)

vFinalB = solvemccall(w,p,c,a,β)

findfirst(vFinalB .> vFinalB[1])

#extract reservation wage
resWageIndex = findResWageIndex(vFinalB)

resWage = w[resWageIndex]

println("Reservation wage is ", resWage, ", at index ", resWageIndex)

# This is basically where part C starts

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

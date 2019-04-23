using Pkg
ENV["GRDIR"]=""
# Pkg.build("GR")
#Pkg.add("Distributions")
using Distributions
#Pkg.add("LinearAlgebra")
using LinearAlgebra
#Pkg.add("Parameters")
using Parameters
#Pkg.add("Plots")
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

# This is basically part B I think

p, w = approximatelognormal(0,0.25,100)
a = 0.03
β = 0.99
c = 0.8
v0 = zeros(length(w))

K = 400
vt = zeros(K, length(v0))
vt[1,:] = mccallbellman(v0,w,p,c,a,β)
for k in 2:K
    vt[k,:] = mccallbellman(vt[k-1,:],w,p,c,a,β)
end
plot(w,v0)
plot!(w,vt[K,:])

vFinalB = solvemccall(w,p,c,a,β)

findfirst(vFinalB .> vFinalB[1])

#extract reservation wage
resWageIndex = findResWageIndex(vFinalB)

resWage = w[resWageIndex]

println("Reservation wage is ", resWage, ", at index ", resWageIndex)

# This is where part C starts

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

# Parts D, E, F

# Bellman Eq.: Part D

function mccallbellman_quitsE(v,J,w,p,c,a,β)
    # Value of reject offer
    v_reject = zeros(length(w), J+1)
    v_accept = zeros(length(w), J+1)
    v_reject[:,1] = β * dot(p, v[:,1]) .+ zeros(length(w))
    for j in 1:J+1
        v_accept[:,j] = w .+ (β*((a*(v[:,J+1]) + (1-a)*v[:,J+1])))
    end
    for j in 2:J+1
        v_reject[:,j] = c .+ β * dot(p, v[:,j-1]) .+ zeros(length(w))
    end
    # Value of accepting the offer
    v_n = max.(v_reject, v_accept)
    return v_n
end

# Part E: Solving Bellman

function solvemccallquits_E(w,J,p,c,a,β,ϵ=1e-6)
    # "Initialize"
    v = zeros(length(w), J+1)
    diff = 1.
    # "Check if stopping criteria is reached"
    while diff > ϵ
        v_n = mccallbellman_quitsE(v,J,w,p,c,a,β)
        # "Use supremum norm"
        diff = norm(v-v_n,Inf)
        v = v_n # Reset v
    end
    return v
end

p, w = approximatelognormal(0,0.25,100)
a = 0.03
β = 0.9
c = 0.8
v0 = zeros(length(w), J+1)

J=10
K = 400
vt_E = zeros(K, length(w), J+1)
vt_E[1,:,:] = mccallbellman_quitsE(v0,J,w,p,c,a,β)
for k in 2:K
    vt_E[k,:,:] = mccallbellman_quitsE(vt_E[k-1,:,:],J,w,p,c,a,β)
end
plot(w,vt_E[K,:,1])
plot!(w,vt_E[K,:,2])
plot!(w,vt_E[K,:,11])

vFinal_E = solvemccallquits_E(w,J,p,c,a,β)

variedResWages_E = zeros(J+1)
for j in 1:J+1
  variedResWages_E[j] = w[findfirst(vFinal_E[:,j] .> vFinal_E[1,j])]
end

# Part F

function mccallbellman_quitsF(v,J,w,p,c,a,β)
    # Value of reject offer
    v_reject = zeros(length(w), J+1)
    v_accept = zeros(length(w), J+1)
    v_reject[:,1] = β * dot(p, v[:,1]) .+ zeros(length(w))
    for j in 1:J+1
        v_accept[:,j] = w .+ (β*((a*(v[:,J+1]) + (1-a)*v[:,1])))
    end
    for j in 2:J+1
        v_reject[:,j] = c .+ β * dot(p, v[:,j-1]) .+ zeros(length(w))
    end
    # Value of accepting the offer
    v_n = max.(v_reject, v_accept)
    return v_n
end

# Part G: Solving Bellman from F

function solvemccallquits_F(w,J,p,c,a,β,ϵ=1e-6)
    # "Initialize"
    v = zeros(length(w), J+1)
    diff = 1.
    # "Check if stopping criteria is reached"
    while diff > ϵ
        v_n = mccallbellman_quitsF(v,J,w,p,c,a,β)
        # "Use supremum norm"
        diff = norm(v-v_n,Inf)
        v = v_n # Reset v
    end
    return v
end

vFinal_G = solvemccallquits_F(w,J,p,c,a,β)

variedResWages_G = zeros(J+1)
for j in 1:J+1
  variedResWages_G[j] = w[findfirst(vFinal_G[:,j] .> vFinal_G[1,j])]
end

E_Answer = scatter(variedResWages_E,color =:blue, xlabel="Unemployment Benefits Remaining", ylabel="Reservation Wage")
G_Answer = scatter!(variedResWages_G,color =:red)

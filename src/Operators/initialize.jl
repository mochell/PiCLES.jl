

#### initialize #####

using StaticArrays

""" sets node state values to S at tuple position ij to zi """
function init_z0_to_State!(S::SharedMatrix, ij::Int, zi::TT) where {TT<:Union{Vector{Float64},SVector{3,Float64}}}
        S[ ij[1], :] = zi
        nothing
end

""" sets node state values to S at tuple position ij to zi """
function init_z0_to_State!(S::StateTypeL1, ij::II, zi::TT) where {II<:Union{Tuple{Int,Int},CartesianIndex}, TT<:Union{Vector{Float64},SVector{3,Float64}}}
        S[ij[1],ij[2], :] = zi
        nothing
end


""" sets particle state values to S. position is taking from particle """
function set_u_to_shared!(S::StateTypeL1, PI::AbstractParticleInstance)
        S[ PI.position_ij[1], PI.position_ij[2] , :] = PI.ODEIntegrator.u
        nothing
end

""" takes node state values and pushes them to particle. position is taken from particle """
function set_u_to_particle!(S::StateTypeL1, PI::AbstractParticleInstance)
        set_u!(PI.ODEIntegrator, Get_u_FromShared(S, PI))
        u_modified!(PI.ODEIntegrator,true)
        nothing
end


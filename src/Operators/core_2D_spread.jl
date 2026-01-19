module core_2D_spread

using DifferentialEquations: OrdinaryDiffEq.ODEIntegrator, OrdinaryDiffEq.ODEProblem, init
using ModelingToolkit: ODESystem  ## depriciate when MTK is removed

using SharedArrays
using DocStringExtensions

# Particle-Node interaction
export GetParticleEnergyMomentum, GetVariablesAtVertex, Get_u_FromShared, ParticleDefaults, InitParticleInstance
export InitParticleValues

#include("../Utils/FetchRelations.jl")
using ...FetchRelations

using ...Architectures: AbstractParticleInstance, AbstractMarkedParticleInstance, StateTypeL1
using ...Architectures: AbstractGridStatistics

# using ..particle_waves_v3: init_vars
# t, x, y, c̄_x, c̄_y, lne, Δn, Δφ_p, r_g, C_α, C_φ, g, C_e = init_vars()

using ...custom_structures: ParticleInstance1D, StochasticParticleInstance2D, MarkedParticleInstance

export init_z0_to_State!
include("initialize.jl")

# Callbacks
export wrap_pos!, periodic_BD_single_PI!, show_pos!, periodic_condition_x
include("utils.jl")

### particle defaults ###
"""
ParticleDefaults
Structure holds default particles
# Fields  
$(DocStringExtensions.FIELDS)
"""
mutable struct ParticleDefaults
        "log energy"
        lne::Float64
        "zonal velocity"
        c̄_x::Float64
        "meridional velocity"
        c̄_y::Float64
        "x Position"
        x::Float64
        "y Position"
        y::Float64
        "angular spreading"
        angular_σ::Float64
end

Base.copy(s::ParticleDefaults) = ParticleDefaults(s.lne, s.c̄_x, s.c̄_y, s.x, s.y, s.angular_σ)
initParticleDefaults(s::ParticleDefaults) = [s.lne, s.c̄_x, s.c̄_y, s.x, s.y, s.angular_σ]

mutable struct PointSource
        "particle to launch"
        particleLaunch::ParticleDefaults
        "first launch time"
        firstTimeLaunched::Float64
end

Base.copy(s::PointSource) = PointSource(s.particleLaunch, s.lastTimeLaunched)
PointSource(s::PointSource) = [s.particleLaunch, s.lastTimeLaunched]

"""


"""

######### Particle --> Node  |  2D  ##########


"""
GetParticleEnergyMomentum(PI)

"""
function GetParticleEnergyMomentum(PI)
        ui_lne, ui_c̄_x, ui_c̄_y, ui_x, ui_y, ui_angular_σ = PI.ODEIntegrator.u

        ui_e = exp(ui_lne)
        c_speed = speed(ui_c̄_x, ui_c̄_y)
        m_x = ui_c̄_x * ui_e / c_speed^2 / 2
        m_y = ui_c̄_y * ui_e / c_speed^2 / 2

        return ui_e, m_x, m_y, ui_angular_σ
end

function GetParticleEnergyMomentum(PI::AbstractParticleInstance)
        return GetParticleEnergyMomentum(PI.ODEIntegrator.u)
end

function GetParticleEnergyMomentum(zi::Dict)
        return GetParticleEnergyMomentum([zi[lne], zi[c̄_x], zi[c̄_y], zi[x], zi[y], zi[angular_σ]])
end

function GetParticleEnergyMomentum(zi::ParticleDefaults)
        return GetParticleEnergyMomentum([zi.lne, zi.c̄_x, zi.c̄_y, zi.x, zi.y, zi.angular_σ])
end

"""
GetParticleEnergyMomentum(PI)

"""
function GetParticleEnergyMomentum(z0::Vector{Float64})

        ui_lne, ui_c̄_x, ui_c̄_y, ui_x, ui_y, ui_angular_σ =z0
        ui_e = exp(ui_lne)
        c_speed = speed(ui_c̄_x, ui_c̄_y)
        m_x = ui_c̄_x * ui_e / c_speed^2 / 2
        m_y = ui_c̄_y * ui_e / c_speed^2 / 2

        return [ui_e, m_x, m_y, ui_angular_σ]
end

speed(x::Float64, y::Float64) = sqrt(x^2 + y^2)

######### node--> Particle ##########

"""
GetVariablesAtVertex(i_State::Vector{Float64}, x::Float64, y::Float64, Δn::Float64, Δφ_p::Float64 )
i_state: [e, m_x, m_y] state vector at node
x, y: coordinates of the vertex

"""
function GetVariablesAtVertex(i_State::Vector{Float64}, x::Float64, y::Float64)
        e, m_x, m_y, sigma = i_State
        m_amp = speed(m_x, m_y)
        c_x = m_x * e / (2 * m_amp^2)
        c_y = m_y * e / (2 * m_amp^2)

    return [log(e), c_x, c_y, x, y, sigma]
end


""" returns node state from shared array, given paritcle index """
Get_u_FromShared(PI::AbstractParticleInstance, S::SharedArray,) = S[PI.position_ij[1], PI.position_ij[2], :]

"""
GetGroupVelocity(i_State::Vector{Float64})
i_state: [e, m_x, m_y] state vector at node
"""
function GetGroupVelocity(i_State)
        e = i_State[:,:,1]
        m_x = i_State[:,:,2]
        m_y = i_State[:,:,3]
        m_amp = speed.(m_x, m_y)
        c_x = m_x .* e ./ (2 .* m_amp.^2)
        c_y = m_y .* e ./ (2 .* m_amp.^2)

        return (c_x=c_x, c_y=c_y)
end


###### seed particles #####


"""
InitParticleInstance(model::WaveGrowth1D, z_initials, pars, ij, boundary_flag, particle_on ; cbSets=nothing)
wrapper function to initalize a particle instance
        inputs:
        model           is an initlized ODESytem
        z_initials      is the initial state of the ODESystem
        pars            are the parameters of the ODESystem
        ij              is the (i,j) tuple that of the initial position
        boundary_flag   is a boolean that indicates if the particle is on the boundary
        particle_on     is a boolean that indicates if the particle is on
        chSet           (optional) is the set of callbacks the ODE can have
"""
function InitParticleInstance(model, z_initials, ODE_settings, ij, boundary_flag, particle_on; cbSets=Nothing)

        # converty to ordered named tuple
        ODE_parameters = NamedTuple{Tuple(Symbol.(keys(ODE_settings.Parameters)))}(values(ODE_settings.Parameters))

        z_initials = initParticleDefaults(z_initials)
        # create ODEProblem
        problem = ODEProblem(model, z_initials, (0.0, ODE_settings.total_time), ODE_parameters)
        # inialize problem
        # works best with abstol = 1e-4,reltol=1e-3,maxiters=1e4,
        integrator = init(
                problem,
                ODE_settings.solver,
                saveat=ODE_settings.saving_step,
                abstol=ODE_settings.abstol,
                adaptive=ODE_settings.adaptive,
                dt = ODE_settings.dt,
                dtmin=ODE_settings.dtmin,
                force_dtmin=ODE_settings.force_dtmin,
                maxiters=ODE_settings.maxiters,
                reltol=ODE_settings.reltol,
                callback=ODE_settings.callbacks,
                save_everystep=ODE_settings.save_everystep)
        return StochasticParticleInstance2D(ij, (z_initials[4], z_initials[5]), integrator, boundary_flag, particle_on)
end

"""
InitParticleInstance(model::ODESystem, z_initials, pars, ij, boundary_flag, particle_on ; cbSets=nothing)
wrapper function to initalize a particle instance
        inputs:
        model           is an initlized ODESytem
        z_initials      is the initial state of the ODESystem
        pars            are the parameters of the ODESystem
        ij              is the (i,j) tuple that of the initial position
        boundary_flag   is a boolean that indicates if the particle is on the boundary
        particle_on     is a boolean that indicates if the particle is on
        chSet           (optional) is the set of callbacks the ODE can have
"""
function InitParticleInstance(model::ODESystem, z_initials, ODE_settings, ij, boundary_flag, particle_on; cbSets=Nothing)

        z_initials = initParticleDefaults(z_initials)
        # create ODEProblem
        problem = ODEProblem(model, z_initials, (0.0, ODE_settings.total_time), ODE_settings.Parameters)
        # inialize problem
        # works best with abstol = 1e-4,reltol=1e-3,maxiters=1e4,
        integrator = init(
                problem,
                ODE_settings.solver,
                saveat=ODE_settings.saving_step,
                abstol=ODE_settings.abstol,
                adaptive=ODE_settings.adaptive,
                dt=ODE_settings.dt,
                dtmin=ODE_settings.dtmin,
                force_dtmin=ODE_settings.force_dtmin,
                maxiters=ODE_settings.maxiters,
                reltol=ODE_settings.reltol,
                callback=ODE_settings.callbacks,
                save_everystep=ODE_settings.save_everystep)
        return StochasticParticleInstance2D(ij, (z_initials[4], z_initials[5]), integrator, boundary_flag, particle_on)
end




"""
InitParticleValues(defaults:: Union{Nothing,ParticleDefaults}, ij::Tuple, gridnote::OneDGridNotes, u, v, DT)

Find initial conditions for particle. Used at the beginning of the experiment.
        inputs:
        defaults        Dict with variables of the state vector, these will be replaced in this function
        ij              index of the grid point
        gridnote        grid to determine the position in the grid
        winds           NamedTuple (u,v) with interp. functions for wind values
        DT              time step of model, used to determine fetch laws
"""
function InitParticleValues(
        defaults::PP,
        xy::Tuple{Float64, Float64},
        uv::Tuple{Number, Number}, 
        DT) where {PP<:Union{Nothing,ParticleDefaults}}

        #i,j = ij
        xx, yy = xy

        if defaults == nothing

                if speed(uv[1], uv[2]) > sqrt(2)
                        # defaults are not defined and there is wind

                        # take in local wind velocities
                        WindSeaMin = FetchRelations.get_initial_windsea(uv[1], uv[2], DT)
                        # seed particle given fetch relations
                        lne = log(WindSeaMin["E"])
                        c̄_x = WindSeaMin["cg_bar_x"]
                        c̄_y = WindSeaMin["cg_bar_y"]
                        angular_σ = π/8

                        particle_on = false
                else
                        # defaults are not defined and there is no wind
                        u_min = FetchRelations.MinimalParticle(uv[1], uv[2], DT)
                        lne = u_min[1]
                        c̄_x = u_min[2]
                        c̄_y = u_min[3]
                        angular_σ = π/8
                        
                        particle_on = false
                end        
                
                # initialize particle instance based on above devfined values
                particle_defaults = ParticleDefaults(lne, c̄_x, c̄_y, xx, yy, angular_σ)
        else
                particle_defaults = defaults
                particle_on = true
        end

        #@show defaults
        return particle_defaults, particle_on
end

"""
InitParticleInstance(model::WaveGrowth1D, z_initials, pars, ij, boundary_flag, particle_on ; cbSets=nothing)
wrapper function to initalize a particle instance
        inputs:
        model           is an initlized ODESytem
        z_initials      is the initial state of the ODESystem
        pars            are the parameters of the ODESystem
        ij              is the (i,j) tuple that of the particle position
        xy              is the (x,y) tuple that of the particle position
        boundary_flag   is a boolean that indicates if the particle is on the boundary
        particle_on     is a boolean that indicates if the particle is on
        chSet           (optional) is the set of callbacks the ODE can have
"""
function InitParticleInstance(model, z_initials, ODE_settings, ij, xy , boundary_flag, particle_on; cbSets=Nothing)

        ## to do's for add the Projection:
        ## replace boundary_flag with mask that discrimiated by types [0,1,2,3]
        ## add M as input to the function and add it to the ODE_parameters
        ## modify ParticleInstance2D to deal with boundaries and mask

        # converty to ordered named tuple
        ODE_parameters = NamedTuple{Tuple(Symbol.(keys(ODE_settings.Parameters)))}(values(ODE_settings.Parameters))
        ODE_parameters = (; ODE_parameters..., x = xy[1], y = xy[2])

        z_initials = initParticleDefaults(z_initials)
        # create ODEProblem
        problem = ODEProblem(model, z_initials, (0.0, ODE_settings.total_time), ODE_parameters)
        # inialize problem
        # works best with abstol = 1e-4,reltol=1e-3,maxiters=1e4,
        integrator = init(
                problem,
                ODE_settings.solver,
                saveat=ODE_settings.saving_step,
                abstol=ODE_settings.abstol,
                adaptive=ODE_settings.adaptive,
                dt = ODE_settings.dt,
                dtmin=ODE_settings.dtmin,
                force_dtmin=ODE_settings.force_dtmin,
                maxiters=ODE_settings.maxiters,
                reltol=ODE_settings.reltol,
                callback=ODE_settings.callbacks,
                save_everystep=ODE_settings.save_everystep)

        return StochasticParticleInstance2D(ij, (xy[1], xy[2]), integrator, boundary_flag, particle_on)
end


# --------------------------- up to here for now (remove MTK)--------------------------- #

"""
ResetParticleValues(PI::AbstractParticleInstance, particle_defaults::Union{Nothing,ParticleDefaults,Vector{Float64}}, ODE_settings::ODESettings, gridnote::OneDGridNotes, winds, DT)

Reset the state of a particle instance
        inputs:
        PI                      ParticleInstance
        particle_defaults       Dict with variables of the state vector, these will be replaced in this function
        ODE_settings            ODESettings type
        gridnote                grid to determine the position in the grid
        winds                   NamedTuple (u,v) with interp. functions for wind values
        DT                      time step of model, used to determine fetch laws
returns:
        dict or vector
"""
function ResetParticleValues(
        defaults::PP,
        PI::AbstractParticleInstance,
        wind_tuple,
        DT, vector=true) where {PP<:Union{Nothing,ParticleDefaults,Vector{Float64}}}

        if defaults == nothing # this is boundary_defaults = "wind_sea"
                #@info "init particles from fetch relations: $z_i"
                #particle_defaults = Dict{Num,Float64}()
                #xx, yy = PI.position_xy[1], PI.position_xy[2]
                # take in local wind velocities
                u_init, v_init = wind_tuple[1], wind_tuple[2] 
                #winds.u(xx, yy, 0), winds.v(xx, yy, 0)

                WindSeaMin = FetchRelations.get_initial_windsea(u_init, v_init, DT)
                # seed particle given fetch relations
                particle_defaults = ParticleDefaults(log(WindSeaMin["E"]), WindSeaMin["cg_bar_x"], WindSeaMin["cg_bar_y"], PI.position_xy[1], PI.position_xy[2], π/8)
        elseif typeof(defaults) == Vector{Float64} # this is for the case of minimal wind sea
                particle_defaults = defaults
                particle_defaults[4]    = PI.position_xy[1]
                particle_defaults[5]    = PI.position_xy[2]
        else
                particle_defaults = defaults
        end

        #@show defaults
        if vector
                return initParticleDefaults(particle_defaults)
        else
                return particle_defaults
        end

end
"""
check_boundary_point(i, boundary, periodic_boundary)
checks where ever or not point is a boundary point.
returns Bool
"""
function check_boundary_point(i, boundary, periodic_boundary)
    return periodic_boundary ? false : (i in boundary)
end

function check_boundary_point(imesh, periodic_boundary)
        if periodic_boundary
                return imesh == 2 # only land points are treated as boundary points
        else
                return imesh >= 2 # land points and grid boundary points are treated as boundary points
        end
end


# """
# SeedParticle!(ParticleCollection ::Vector{Any}, State::SharedMatrix, i::Int64,
#                 particle_system::ODESystem, particle_defaults::Union{ParticleDefaults,Nothing}, ODE_settings,
#                 GridNotes, winds, DT:: Float64, Nx:: Int, boundary::Vector{Int}, periodic_boundary::Bool)

# Seed Pickles to ParticleColletion and State
# """
# function SeedParticle!(
#         ParticleCollection::Vector{Any},
#         State::SharedArray,
#         ij::Tuple{Int, Int},

#         particle_system::SS,
#         particle_defaults::PP,
#         ODE_settings, #particle_waves_v3.ODESettings type

#         GridNotes, # ad type of grid note
#         winds,     # interp winds
#         DT::Float64,

#         boundary::Vector{T},
#         periodic_boundary::Bool) where {T<:Union{Int,Any,Nothing,Int64},PP<:Union{ParticleDefaults,Nothing},SS<:Union{ODESystem,Any}}

#         xx, yy = GridNotes.x[ij[1]], GridNotes.y[ij[2]]

        
#         #uv = winds.u(xx, yy, 0)::Union{Num,Float64}, winds.v(xx, yy, 0)::Union{Num,Float64}
#         uv = winds.u(xx, yy, 0)::Float64, winds.v(xx, yy, 0)::Float64

#         # define initial condition
#         z_i, particle_on = InitParticleValues(particle_defaults, (xx,yy), uv, DT)
#         # check if point is boundary point
#         boundary_point = check_boundary_point(ij, boundary, periodic_boundary)
#         #@info "boundary?", boundary_point

#         # if boundary_point
#         #     #@info "boundary point", boundary_point, particle_on
#         #     # if boundary point, then set particle to off
#         #     particle_on = false
#         # end

#         # add initial state to State vector
#         if particle_on
#                 init_z0_to_State!(State, ij, GetParticleEnergyMomentum(z_i))
#         end

#         # Push Inital condition to collection
#         push!(ParticleCollection,
#         InitParticleInstance(
#                 particle_system,
#                 z_i,
#                 ODE_settings,
#                 ij,
#                 boundary_point,
#                 particle_on))
#         nothing
# end


"""
        function SeedParticle(State::StateTypeL1, ij:: (Int64, Int64)
        particle_system::Any, particle_defaults::Union{ParticleDefaults,Nothing}, ODE_settings,
        ij_mesh, ij_wind_tuple, DT:: Float64, boundary::Vector{Int}, periodic_boundary::Bool)

        returns ParicleInstance that can be pushed to ParticleColletion
"""
function SeedParticle(
        State::StateTypeL1,
        ij::II, # position tuple in grid
        
        particle_system::SS,
        particle_defaults::PP,
        ODE_settings, #particle_waves_v3.ODESettings type
        
        gridstats::AbstractGridStatistics,
        ProjetionKernel::Function,
        PropagationCorrection::Function,

        ij_mesh::NamedTuple, # local grid information
        ij_wind::Tuple,     # interp winds
        
        DT::Float64,
        boundary::Vector{T}, periodic_boundary::Bool) where 
        {II<:Union{Tuple{Int,Int},CartesianIndex},T<:Union{Int,Any,Nothing,Int64},PP<:Union{ParticleDefaults,Nothing},SS<:Any}

        xy = (ij_mesh.x, ij_mesh.y)
        # 1st check if particle is not in mask, Land points == 0
        if ij_mesh.mask == 0 # land point
                # init dummy instance
                return StochasticParticleInstance2D(ij, xy , nothing, false, false)
        end

        # define initial condition
        # particle initial condition is always (0,0) in relative coordinates not xy anymore
        z_i, particle_on = InitParticleValues(particle_defaults, (0.0, 0.0) , ij_wind, DT)

        # check if point is boundary point <-- replace in the future with with mask: 0 = land, 1 = ocean, 2= land boundary, 3 = domain boundary
        # boundary_point = check_boundary_point(ij, boundary, periodic_boundary) # old version that compares to list 
        boundary_point   = check_boundary_point(ij_mesh.mask, periodic_boundary)

        # add initial state to State vector
        if particle_on
                init_z0_to_State!(State, ij, GetParticleEnergyMomentum(z_i))
        end

        # check if Propgation Correction is set in gridstats
        # PropagationCorrection   = gridstats.PropagationCorrection != nothing ? PropagationCorrection(ij_mesh, gridstats) : x -> 0.0

        # set projection:
        ODE_settings.Parameters = (; ODE_settings.Parameters..., M=ProjetionKernel(ij_mesh, gridstats), PC=PropagationCorrection(ij_mesh, gridstats))

        return InitParticleInstance(
                particle_system,
                z_i,
                ODE_settings,
                ij,
                xy,
                boundary_point,
                particle_on)

end

# """
# SeedParticle!(ParticleCollection ::Vector{Any}, State::SharedMatrix, i::Int64,
#                 particle_system::ODESystem, particle_defaults::Union{ParticleDefaults,Nothing}, ODE_settings,
#                 GridNotes, winds, DT:: Float64, Nx:: Int, boundary::Vector{Int}, periodic_boundary::Bool)

# Seed Pickles to ParticleColletion and State
# """
# function SeedParticle!(
#         ParticleCollection::Vector{Any},
#         State::SharedArray,
#         ij::Tuple{Int, Int},

#         particle_system::SS,
#         particle_defaults::PP,
#         ODE_settings, #particle_waves_v3.ODESettings type

#         GridNotes, # ad type of grid note
#         winds,     # interp winds
#         DT::Float64,

#         boundary::Vector{T},
#         periodic_boundary::Bool) where {T<:Union{Int,Any,Nothing,Int64},PP<:Union{ParticleDefaults,Nothing},SS<:Union{ODESystem,Any}}

#         # Push Inital condition to collection
#         push!(ParticleCollection,
#                 SeedParticle(State, ij,
#                         particle_system, particle_defaults, ODE_settings, #particle_waves_v3.ODESettings type
#                         GridNotes, winds, DT,
#                         boundary, periodic_boundary))
#         nothing
# end



# end of module
end
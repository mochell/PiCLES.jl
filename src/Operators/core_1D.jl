module core_1D

using DifferentialEquations: OrdinaryDiffEq.ODEProblem, init

using SharedArrays
using DocStringExtensions
# Particle-Node interaction
export GetParticleEnergyMomentum, GetVariablesAtVertex, Get_u_FromShared, ParticleDefaults
export InitParticleValues, ResetParticleValues


using ..FetchRelations

using ...Architectures: AbstractParticleInstance, AbstractMarkedParticleInstance, StateTypeL1

# using ..particle_waves_v3: init_vars_1D
# t, x, c̄_x, lne, r_g, C_α, g, C_e = init_vars_1D()

using ...custom_structures: ParticleInstance1D, ParticleInstance2D, MarkedParticleInstance

export InitParticleInstance, init_z0_to_State!
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
struct ParticleDefaults
    "log energy"
    lne::Float64
    "horizontal velocity"
    c̄_x::Float64
    "x Position"
    x::Float64
end


Base.copy(s::ParticleDefaults) = ParticleDefaults(s.lne, s.c̄_x, s.x)
initParticleDefaults(s::ParticleDefaults) = [s.lne, s.c̄_x, s.x]

# """
# ParticleDefaults
# Structure holds default particles
# # Fields  
# $(DocStringExtensions.FIELDS)
# """
# struct ParticleDefaults_old
#     "x Position"
#     x::Float64
#     "horizontal velocity"
#     c̄_x::Float64
#     "log energy"
#     lne::Float64
# end

speed(x::Float64, y::Float64) = sqrt(x^2 + y^2)

######### Particle --> Node ##########

#### 1D code ####
# """
# GetParticleEnergyMomentum(PI)

# """
# function GetParticleEnergyMomentum(PI)
#     #ui_x, ui_c̄_x, ui_lne = PI.ODEIntegrator.u
#     ui_lne, ui_c̄_x, _ = PI.ODEIntegrator.u

#     ui_e = exp(ui_lne)
#     #c_speed = speed(ui_c̄_x, ui_c̄_y)
#     m_x = ui_e / ui_c̄_x / 2
#     #m_y = ui_c̄_y * ui_e  / c_speed^2 / 2

#     return ui_e, m_x, 0
# end

function GetParticleEnergyMomentum(PI::AbstractParticleInstance)
    return GetParticleEnergyMomentum(PI.ODEIntegrator.u)
end

function GetParticleEnergyMomentum(zi::Dict)
    return GetParticleEnergyMomentum([zi[lne], zi[c̄_x], zi[x]])
end

function GetParticleEnergyMomentum(zi::ParticleDefaults)
    return GetParticleEnergyMomentum([zi.lne, zi.c̄_x, zi.x])
end


"""
GetParticleEnergyMomentum(u_particle::Vector{Float64,3})
z0 is the particle state vector u with lne, cx, position
returns the state vector: energy, momentum, 0 (we are just in 1D) 
"""
function GetParticleEnergyMomentum(u_particle::Vector{Float64})

    ui_lne, ui_c̄_x, _ = u_particle
    ui_e = exp(ui_lne)
    #c_speed =  ui_c̄_x
    m_x = ui_e / ui_c̄_x / 2.0
    #m_y = ui_c̄_y * ui_e  / c_speed^2 / 2

    return [ui_e, m_x]
end




######### node--> Particle ##########
"""
GetVariablesAtVertex(i_State::Vector{Float64}, x::Float64)
i_State: [e, m_x, m_y] state vector at node
x: coordinates of the vertex
"""
function GetVariablesAtVertex(i_State::Vector{Float64}, x::Float64)
    e, m_x = i_State
    #m_amp = particle_waves_v3.speed(m_x, m_y)
    c_x = e / 2.0 / m_x
    #c_y = m_y * e / (2  * m_amp^2)

    return [log(e), c_x, x]
end


""" returns node state from shared array, given particle index """
Get_u_FromShared(PI::AbstractParticleInstance, S::SharedMatrix,) = S[PI.position_ij[1], :]


###### seed particles #####

# normal version
"""
InitParticleInstance(model::WaveGrowth1D, z_initials, pars,  ij ; cbSets=nothing)
wrapper function to initalize a particle instance
        inputs:
        model           is an initlized ODESytem
        z_initials      is the initial state of the Any
        pars            are the parameters of the Any
        ij              is the (i,j) tuple that of the initial position
        chSet           (optional) is the set of callbacks the ODE can have
"""
function InitParticleInstance(model, z_initials, ODE_settings, ij, boundary_flag, particle_on; cbSets=Nothing)

    # convert to ordered particle state
    
    #z_initials = [z_initials[lne], z_initials[c̄_x], z_initials[x]]
    z_initials = initParticleDefaults(z_initials)
    # converty to ordered named tuple
    ODE_parameters = NamedTuple{Tuple(Symbol.(keys(ODE_settings.Parameters)))}(values(ODE_settings.Parameters))

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
        dt=ODE_settings.dt,
        dtmin=ODE_settings.dtmin,
        force_dtmin=ODE_settings.force_dtmin,
        maxiters=ODE_settings.maxiters,
        reltol=ODE_settings.reltol,
        callback=ODE_settings.callbacks,
        save_everystep=ODE_settings.save_everystep)
    return ParticleInstance1D(ij, z_initials[3], integrator, boundary_flag, particle_on)
end




"""
InitParticleValues(defaults:: Dict{Number, Float64}, xx::Float64, u, DT)

Find initial conditions for particle. Used at the beginning of the experiment.
        inputs:
        defaults        Dict with variables of the state vector, these will be replaced in this function
        xx              grid to determine the position in the grid
        u               wind value
        DT              time step of model, used to determine fetch laws
"""
function InitParticleValues(
    defaults::PP,
    xx::Float64,
    uu::Number,
    DT) where {PP<:Union{Nothing,ParticleDefaults}}
    # take in local wind velocities

    if defaults == nothing
        #@info "init particles from fetch relations: $z_i"

        if abs(uu) > sqrt(2)
            # defaults are not defined and there is wind

            # take in local wind velocities
            WindSeaMin = FetchRelations.get_initial_windsea(uu, 0.0, DT)
            # seed particle given fetch relations
            lne = log(WindSeaMin["E"])
            c̄_x = WindSeaMin["cg_bar_x"]
            # particle_defaults[c̄_y] = WindSeaMin["cg_bar_y"]

            particle_on = true

        else

            # defaults are not defined and there is no wind
            u_min = FetchRelations.MinimalParticle(uu,0.0, DT) # takes u just for the sign.
            lne = u_min[1]
            c̄_x = u_min[2]
            
            particle_on = false
        end

        # initialize particle instance based on above devfined values
        particle_defaults = ParticleDefaults(lne, c̄_x, xx)

    else
        particle_defaults = defaults
        particle_on = true
    end


    #@show defaults
    return particle_defaults, particle_on
end

"""
ResetParticleValues(defaults:: Dict{Number, Float64}, PI::AbstractParticleInstance, u_rn, DT)

resets the particle state to the default values if they given, otherwise it will use the fetch relations
        inputs:
        defaults        Dict with variables of the state vector, these will be replaced in this function
        PI              ParticleInstance to be reset
        u_rn            u value at the particle position
        DT              time step of model, used to determine fetch laws
"""
function ResetParticleValues(
    defaults::PP,
    PI::AbstractParticleInstance,
    u_rn::Number, DT; vector=true) where {PP<:Union{Nothing,ParticleDefaults}}
    # take in local wind velocities

    if defaults == nothing # this is boundary_defaults = "wind_sea"
        #particle_defaults = Dict{Number,Float64}()
        #@info "init particles from fetch relations: $z_i"
        # seed particle given fetch relations
        WindSeaMin = FetchRelations.get_initial_windsea(u_rn, DT) # takes u_init just for the sign.
        particle_defaults = ParticleDefaults(log(WindSeaMin["E"]), WindSeaMin["cg_bar"], PI.position_xy[1])

    else
        particle_defaults = defaults#deepcopy(defaults)
    end
        
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




"""
SeedParticle!(ParticleCollection ::Vector{Any}, State::SharedMatrix, i::Int64,
                particle_system::Any, particle_defaults::Dict{Number, Float64}, ODE_defaults::Dict{Number, Float64},
                GridNotes, winds, DT:: Float64, Nx:: Int, boundary::Vector{Int}, periodic_boundary::Bool)

Seed Pickles to ParticleColletion and State
"""
function SeedParticle!(
    ParticleCollection::Vector{Any},
    State::SharedMatrix,
    i::Int64,

    particle_system::SS,
    particle_defaults::PP,
    ODE_settings, #particle_waves_v3.ODESettings type

    GridNotes, # ad type of grid note
    winds,     # interp winds
    DT::Float64,
    
    boundary::Vector{T},
    periodic_boundary::Bool) where {T<:Union{Int,Any,Nothing,Int64},PP<:Union{ParticleDefaults,Nothing},SS<:Any}

    # get x position
    x = GridNotes.x[i]
    u = winds(x, 0.0)::Float64

    # define initial condition
    z_i, particle_on = InitParticleValues(particle_defaults, x, u, DT)

    # check if point is boundary point
    boundary_point = check_boundary_point(i, boundary, periodic_boundary)


    # if boundary_point
    #     #@info "boundary point", boundary_point, particle_on
    #     # if boundary point, then set particle to off
    #     particle_on = false
    # end

    # add initial state to State vector
    #@info z_i
    if particle_on
        init_z0_to_State!(State, i, GetParticleEnergyMomentum(z_i))
    end

    # Push Inital condition to collection
    push!(ParticleCollection,
        InitParticleInstance(
            particle_system,
            z_i,
            ODE_settings,
            i,
            boundary_point,
            particle_on))
    nothing
end


# end of module
end
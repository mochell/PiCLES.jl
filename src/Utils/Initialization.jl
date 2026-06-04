

using PiCLES.ParticleSystems: particle_waves_v6 as PW
import PiCLES: FetchRelations
using PiCLES.Operators.core_2D: ParticleDefaults

"""
function Init_Standard(uscale, vscale, DT; r_g0=0.85)

    returns ParticleState, default_ODE_parameters, WindSeamin, Const_ID
"""
function Init_Standard(uscale, vscale, DT; r_g0=0.85)

    Const_ID = PW.get_I_D_constant()
    #@set Const_ID.γ = 0.88
    Const_Scg = PW.get_Scg_constants()

    default_ODE_parameters = (
        r_g=r_g0,
        C_α=Const_Scg.C_alpha,
        C_φ=Const_ID.c_β,
        C_e=Const_ID.C_e,
        g=9.81,
    )

    WindSeamin = FetchRelations.get_initial_windsea(uscale, vscale, DT / 2)
    ParticleState = ParticleDefaults(log(WindSeamin["E"]), WindSeamin["cg_bar_x"], WindSeamin["cg_bar_y"], 0.0, 0.0)

    return ParticleState, default_ODE_parameters, WindSeamin, Const_ID

end

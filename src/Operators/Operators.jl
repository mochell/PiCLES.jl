module Operators

export core_2D, custom_structures, mapping_2D, TimeSteppers
export init_z0_to_State!
using SharedArrays
using StaticArrays

# NOTE: the shared helpers `utils.jl` (callbacks: wrap_pos!, periodic_BD_single_PI!,
# show_pos!, periodic_condition_x) and `initialize.jl` (init_z0_to_State!) are included
# once inside `core_2D` and re-exported from there. They are intentionally NOT included
# directly here: doing so previously defined them both in `Operators` and in the
# submodules, so `using .core_2D` collided with the local copies and emitted
# "conflicts with an existing identifier" warnings on every `using PiCLES`.

include("core_2D.jl")
include("mapping_2D.jl")

include("TimeSteppers.jl")

using .core_2D
using .mapping_2D
using .TimeSteppers

end
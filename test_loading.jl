#!/usr/bin/env julia

# Script to test PiCLES loading without Revise conflicts
# Run this from the PiCLES directory

println("Testing PiCLES loading...")
println("Julia version: ", VERSION)

# Set up environment
using Pkg
Pkg.activate(".")

try
    println("Attempting to load PiCLES...")
    @time using PiCLES
    println("✅ SUCCESS: PiCLES loaded successfully!")
    
    # Test basic functionality
    println("Testing basic PiCLES functionality...")
    # Add any basic tests here if needed
    
catch e
    println("❌ ERROR: Failed to load PiCLES")
    println("Error details: ", e)
    
    # Additional debugging
    if isa(e, LoadError) && occursin("ModelingToolkit", string(e.error))
        println("\n💡 Suggestion: The ModelingToolkit dependency issue has been resolved in Project.toml")
        println("Try: julia --project=. --startup-file=no test_loading.jl")
    end
end
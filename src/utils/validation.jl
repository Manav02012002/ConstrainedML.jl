"""
    validation.jl

Utilities for validating that layers are truly equivariant.
"""

using Statistics

"""
    check_equivariance(layer, x, g::GaugeTransformation; rtol=1e-4)

Verify that layer is gauge-equivariant: layer(g·x) ≈ g·layer(x)

# Returns
- `true` if equivariant within tolerance
- `false` otherwise

# Examples
```julia
layer = GaugeEquivariantConv(1, 8, symmetry=U1())
x = rand(ComplexF64, 16, 16)
g = GaugeTransformation(U1(), π/4)

is_equivariant = check_equivariance(layer, x, g)
```
"""
function check_equivariance(layer, x, g::GaugeTransformation; rtol=1e-4)
    # Compute: layer(g·x)
    gx = transform(g, x)
    y1 = layer(gx)
    
    # Compute: g·layer(x)
    y = layer(x)
    y2 = transform(g, y)
    
    # Check if they're close
    diff = maximum(abs.(y1 .- y2))
    max_val = maximum(abs.([y1; y2]))
    
    relative_error = diff / (max_val + 1e-10)
    
    is_equivariant = relative_error < rtol
    
    if !is_equivariant
        @warn "Equivariance check failed!" relative_error rtol
    end
    
    return is_equivariant
end

"""
    equivariance_error(layer, x, g::GaugeTransformation)

Compute the equivariance error: ||layer(g·x) - g·layer(x)||

Returns the mean absolute error.
"""
function equivariance_error(layer, x, g::GaugeTransformation)
    gx = transform(g, x)
    y1 = layer(gx)
    
    y = layer(x)
    y2 = transform(g, y)
    
    return mean(abs.(y1 .- y2))
end
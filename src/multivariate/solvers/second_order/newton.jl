struct Newton{IL, L} <: SecondOrderOptimizer
    alphaguess!::IL
    linesearch!::L
end

"""
# Newton
## Constructor
```julia
Newton(; alphaguess = LineSearches.InitialStatic{T}(),
linesearch = LineSearches.HagerZhang{T}())
```

## Description
The `Newton` method implements Newton's method for optimizing a function. We use
a special factorization from the package `PositiveFactorizations.jl` to ensure
that each search direction is a direction of descent. See Wright and Nocedal and
Wright (ch. 6, 1999) for a discussion of Newton's method in practice.

## References
 - Nocedal, J. and S. J. Wright (1999), Numerical optimization. Springer Science 35.67-68: 7.
"""
function Newton(::Type{T}=Float64; alphaguess = LineSearches.InitialStatic{T}(), # Good default for Newton
                linesearch = LineSearches.HagerZhang{T}()) where T   # Good default for Newton
    Newton(alphaguess, linesearch)
end

Base.summary(::Newton) = "Newton's Method"

mutable struct NewtonState{Tx, T, F<:Base.LinAlg.Cholesky} <: AbstractOptimizerState
    x::Tx
    x_previous::Tx
    f_x_previous::T
    F::F
    s::Tx
    @add_linesearch_fields()
end

function initial_state(method::Newton, options, d, initial_x)
    T = eltype(initial_x)
    n = length(initial_x)
    # Maintain current gradient in gr
    s = similar(initial_x)

    value_gradient!!(d, initial_x)
    hessian!!(d, initial_x)
    
    NewtonState(copy(initial_x), # Maintain current state in state.x
                similar(initial_x), # Maintain previous state in state.x_previous
                T(NaN), # Store previous f in state.f_x_previous
                @static(VERSION >= v"0.7.0-DEV.393" ?
                        Base.LinAlg.Cholesky(similar(d.H, T, 0, 0), :U, BLAS.BlasInt(0)) :
                        Base.LinAlg.Cholesky(similar(d.H, T, 0, 0), :U)),
                similar(initial_x), # Maintain current search direction in state.s
                @initial_linesearch()...) # Maintain a cache for line search results in state.lsr
end

function update_state!(d, state::NewtonState, method::Newton)
    # Search direction is always the negative gradient divided by
    # a matrix encoding the absolute values of the curvatures
    # represented by H. It deviates from the usual "add a scaled
    # identity matrix" version of the modified Newton method. More
    # information can be found in the discussion at issue #153.
    T = eltype(state.x)
    
    update_h!(d, state, method)
    
    if typeof(NLSolversBase.hessian(d)) <: AbstractSparseMatrix
        state.s .= -NLSolversBase.hessian(d)\convert(Vector{T}, gradient(d))
    else
        state.F = cholfact!(Positive, NLSolversBase.hessian(d))
        if typeof(gradient(d)) <: Array
            # is this actually StridedArray?
            A_ldiv_B!(state.s, state.F, -gradient(d))
        else
            # not Array, we can't do inplace ldiv 
            gv = Vector{T}(length(gradient(d)))
            copy!(gv, -gradient(d))
            copy!(state.s, state.F\gv)
        end
    end
    # Determine the distance of movement along the search line
    lssuccess = perform_linesearch!(state, method, d)

    # Update current position # x = x + alpha * s
    @. state.x = state.x + state.alpha * state.s
    lssuccess == false # break on linesearch error
end

function assess_convergence(state::NewtonState, d, options)
  default_convergence_assessment(state, d, options)
end

function trace!(tr, d, state, iteration, method::Newton, options)
    dt = Dict()
    if options.extended_trace
        dt["x"] = copy(state.x)
        dt["g(x)"] = copy(gradient(d))
        dt["h(x)"] = copy(NLSolversBase.hessian(d))
        dt["Current step size"] = state.alpha
    end
    g_norm = maximum(abs, gradient(d))
    update!(tr,
            iteration,
            value(d),
            g_norm,
            dt,
            options.store_trace,
            options.show_trace,
            options.show_every,
            options.callback)
end

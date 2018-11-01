using Parameters, Setfield

struct Fminbox{O<:AbstractOptimizer} <: AbstractConstrainedOptimizer
    method::O
end

"""
# Fminbox
## Constructor
```julia
Fminbox(method::T)
```
"""
function Fminbox(method::O = LBFGS()) where O <: FirstOrderOptimizer
    if method isa Newton || method isa NewtonTrustRegion
        throw(ArgumentError("Newton is not supported as the Fminbox optimizer."))
    end
    @assert :alphamax ∈ fieldnames(method.linesearch!)
    Fminbox{typeof(method)}(method)
end

Base.summary(F::Fminbox) = "Fminbox with $(summary(F.method))"

function optimize(f,
                  g,
                  l::AbstractArray{T},
                  u::AbstractArray{T},
                  initial_x::AbstractArray{T},
                  F::Fminbox,
                  options = Options(x_tol=sqrt(eps(T)), f_tol=sqrt(eps(T)), g_tol=sqrt(eps(T))); inplace = true, autodiff = :finite) where T<:AbstractFloat

    g! = inplace ? g : (G, x) -> copy!(G, g(x))
    od = OnceDifferentiable(f, g!, initial_x, zero(T))

    optimize(od, l, u, initial_x, F, options)
end

function optimize(f,
                  l::AbstractArray{T},
                  u::AbstractArray{T},
                  initial_x::AbstractArray{T},
                  F::Fminbox,
                  options = Options(x_tol=sqrt(eps(T)), f_tol=sqrt(eps(T)), g_tol=sqrt(eps(T))); inplace = true, autodiff = :finite) where T<:AbstractFloat

    od = OnceDifferentiable(f, initial_x, zero(T); autodiff = autodiff)
    optimize(od, l, u, initial_x, F, options)
end

function optimize(
        df::OnceDifferentiable,
        l::AbstractArray{T},
        u::AbstractArray{T},
        initial_x::AbstractArray{T},
        F::Fminbox{<:FirstOrderOptimizer},
        options = Options(x_tol=sqrt(eps(T)), f_tol=sqrt(eps(T)), g_tol=sqrt(eps(T)))) where {T<:AbstractFloat}

    for i in eachindex(initial_x)
        thisx = initial_x[i]
        thisl = l[i]
        thisu = u[i]

        if thisx < thisl || thisx > thisu
            error("Initial x[$(ind2sub(initial_x, i))]=$thisx is outside of [$thisl, $thisu]")
        end
    end
    method = F.method
    TLS = typeof(method.linesearch!)

    state = initial_state(method, options, df, initial_x)
    state.s .= .-gradient(df)
    
    @unpack outer_iterations, iterations, allow_f_increases, 
            allow_outer_f_increases, show_trace, store_trace, extended_trace, 
            callback, successive_f_tol, time_limit, f_calls_limit, 
            g_calls_limit, h_calls_limit, x_tol, f_tol, g_tol = options

    t0 = time() # Initial time stamp used to control early stopping by options.time_limit

    tr = OptimizationTrace{typeof(value(df)), typeof(method)}()
    tracing = store_trace || show_trace || extended_trace || callback != nothing
    stopped, stopped_by_callback, stopped_by_time_limit = false, false, false
    f_limit_reached, g_limit_reached, h_limit_reached = false, false, false
    x_converged, f_converged, f_increased, counter_f_tol = false, false, false, 0

    g_converged = initial_convergence(df, state, method, initial_x, options)
    converged = g_converged
    # prepare iteration counter (used to make "initial state" trace entry)
    iteration = 0

    show_trace && print_header(method)
    trace!(tr, df, state, iteration, method, options)

    while !converged && !stopped && iteration < iterations
        iteration += 1

        # Project the search direction on the active constraints
        project_s!(state.s, state.x, l, u)
        g_converged = norm(state.s, Inf) < g_tol
        converged = converged | g_converged
        converged && break

        # Find the maximum step one can make given the box
        alphamax = get_alphamax(state.x, state.s, l, u)
        alphamax = alphamax - eps(alphamax)
        method = @set method.linesearch! = TLS(alphamax=alphamax)

        # Use optimizer and line search to go to next state within box
        # Also updates state.s
        update_state!(df, state, method) && break

        # Find new gradient and objective value
        update_fg!(df, state, method)

        x_converged, f_converged,
        g_converged, converged, f_increased = assess_convergence(state, df, options)
        # For some problems it may be useful to require `f_converged` to be hit multiple times
        # TODO: Do the same for x_tol?
        counter_f_tol = f_converged ? counter_f_tol+1 : 0
        converged = converged | (counter_f_tol > successive_f_tol)

        !converged && update_h!(df, state, method) # only relevant if not converged

        if tracing
            # update trace; callbacks can stop routine early by returning true
            stopped_by_callback = trace!(tr, d, state, iteration, method, options)
        end

        # Check time_limit; if none is provided it is NaN and the comparison
        # will always return false.
        stopped_by_time_limit = time()-t0 > time_limit
        f_limit_reached = f_calls_limit > 0 && f_calls(d) >= f_calls_limit ? true : false
        g_limit_reached = g_calls_limit > 0 && g_calls(d) >= g_calls_limit ? true : false
        h_limit_reached = h_calls_limit > 0 && h_calls(d) >= h_calls_limit ? true : false

        if (f_increased && !allow_f_increases) || stopped_by_callback ||
            stopped_by_time_limit || f_limit_reached || g_limit_reached || h_limit_reached
            stopped = true
        end
    end

    f_incr_pick = f_increased && !allow_f_increases
    O = typeof(F)
    _x_abschange = x_abschange(state)
    Tc = typeof(_x_abschange)
    bestf = pick_best_f(f_incr_pick, state, df)
    Tf = typeof(bestf)
    TTr = typeof(tr)
    Tx = typeof(initial_x)

    return pick_best_x(f_incr_pick, state)
    return MultivariateOptimizationResults{O, T, Tx, Tc, Tf}(F,
                                        initial_x,
                                        pick_best_x(f_incr_pick, state),
                                        bestf,
                                        iteration,
                                        iteration == iterations,
                                        x_converged,
                                        T(x_tol),
                                        _x_abschange,
                                        f_converged,
                                        T(f_tol),
                                        f_abschange(df, state),
                                        g_converged,
                                        T(g_tol),
                                        g_residual(df),
                                        f_increased,
                                        f_calls(df),
                                        g_calls(df),
                                        h_calls(df))
end

function project_s!(v, x, mins, maxs)
    for i in 1:length(v)
        v[i] = ifelse((x[i] <= mins[i] && v[i] < 0) || (x[i] >= maxs[i] && v[i] > 0), 0, v[i])
    end
    v
end

function get_alphamax(x, s, mins, maxs)
    m = 100.
    for i in 1:length(x)
        if s[i] < 0
            m = min((mins[i] - x[i])/s[i], m)
        elseif s[i] > 0
            m = min((maxs[i] - x[i])/s[i], m)
        end
    end
    return m
end

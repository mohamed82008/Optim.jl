const pt = promote_type

# Attempt to compute a reasonable default mu: at the starting
# position, the gradient of the input function should dominate the
# gradient of the barrier.
function initial_mu(gfunc::AbstractArray{T}, gbarrier::AbstractArray{T}; mu0::T = convert(T, NaN), mu0factor::T = 0.001) where T
    if isnan(mu0)
        gbarriernorm = sum(abs, gbarrier)
        if gbarriernorm > 0
            mu = mu0factor*sum(abs, gfunc)/gbarriernorm
        else
            # Presumably, there is no barrier function
            mu = zero(T)
        end
    else
        mu = mu0
    end
    return mu
end

function barrier_box(g, x::AbstractArray{T}, l::AbstractArray{T}, u::AbstractArray{T}) where T
    n = length(x)
    calc_g = !(g === nothing)

    v = zero(T)
    for i = 1:n
        thisl = l[i]
        if isfinite(thisl)
            dx = x[i] - thisl
            if dx <= 0
                return convert(T, Inf)
            end
            v -= log(dx)
            if calc_g
                g[i] = -one(T)/dx
            end
        else
            if calc_g
                g[i] = zero(T)
            end
        end
        thisu = u[i]
        if isfinite(thisu)
            dx = thisu - x[i]
            if dx <= 0
                return convert(T, Inf)
            end
            v -= log(dx)
            if calc_g
                g[i] += one(T)/dx
            end
        end
    end
    return v
end

function function_barrier(gfunc, gbarrier, x::AbstractArray{T}, f::F, fbarrier::FB) where {T, F<:Function, FB<:Function}
    vbarrier = fbarrier(gbarrier, x)
    if isfinite(vbarrier)
        vfunc = f(gfunc, x)
    else
        vfunc = vbarrier
    end
    return vfunc, vbarrier
end

function barrier_combined(gfunc, gbarrier, g, x::AbstractArray{T}, fb::FB, mu::T) where {T, FB<:Function}
    calc_g = !(g === nothing)
    valfunc, valbarrier = fb(gbarrier, x, gfunc)
    if calc_g
        g .= gfunc .+ mu.*gbarrier
    end
    return convert(T, valfunc + mu*valbarrier) # FIXME make this unnecessary
end

function limits_box(x::AbstractArray{T}, d::AbstractArray{T}, l::AbstractArray{T}, u::AbstractArray{T}) where T
    alphamax = convert(T, Inf)
    for i = 1:length(x)
        if d[i] < 0
            @inbounds alphamax = min(alphamax, ((l[i]-x[i])+T(eps(l[i])))/d[i])
        elseif d[i] > 0
            @inbounds alphamax = min(alphamax, ((u[i]-x[i])-T(eps(u[i])))/d[i])
        end
    end
    epsilon = T(eps(max(alphamax, one(T))))
    if !isinf(alphamax) && alphamax > epsilon
        alphamax -= epsilon
    end
    return alphamax
end

# Default preconditioner for box-constrained optimization
# This creates the inverse Hessian of the barrier penalty
function precondprepbox!(P, x, l, u, mu)
    T = eltype(x)
    @. P.diag = 1/(mu*(1/max(x-l, sqrt(realmin(T)))^2 + 1/max(u-x, sqrt(realmin(T)))^2) + 1)
end

struct Fminbox{T<:AbstractOptimizer} <: AbstractOptimizer end
Fminbox() = Fminbox{ConjugateGradient}() # default optimizer

Base.summary(::Fminbox{O}) where {O} = "Fminbox with $(summary(O()))"

function optimize(obj::AbstractObjective,
                  initial_x::AbstractArray,
                  _l::Union{Real,AbstractArray},
                  _u::Union{Real,AbstractArray},
                  F::Fminbox{O} = Fminbox(); kwargs...) where {O<:AbstractOptimizer}
    if !(typeof(_l) <: AbstractArray)
        l = fill(_l, length(initial_x))
    else
        l = _l
    end
    if !(typeof(_u) <: AbstractArray)
        u = fill(_u, length(initial_x))
    else
        u = _u
    end
    Tx = pt(typeof(initial_x), typeof(l), typeof(u))
    T = eltype(Tx)
    optimize(OnceDifferentiable(obj, Tx(initial_x), zero(T)), Tx(initial_x), Tx(l), Tx(u), F; kwargs...)
end

function optimize(f,
                  g!,
                  initial_x::AbstractArray,
                  _l::Union{Real,AbstractArray},
                  _u::Union{Real,AbstractArray},
                  F::Fminbox{O} = Fminbox(); kwargs...) where {O<:AbstractOptimizer}
    if !(typeof(_l) <: AbstractArray)
        l = fill(_l, length(initial_x))
    else
        l = _l
    end
    if !(typeof(_u) <: AbstractArray)
        u = fill(_u, length(initial_x))
    else
        u = _u
    end
    Tx = pt(typeof(initial_x), typeof(l), typeof(u))
    T = eltype(Tx)
    optimize(OnceDifferentiable(f, g!, Tx(initial_x), zero(T)), Tx(initial_x), Tx(l), Tx(u), F; kwargs...)
end

function optimize(
        df::OnceDifferentiable,
        initial_x::AbstractArray{T1},
        _l::Union{T2, AbstractArray{T2}},
        _u::Union{T3, AbstractArray{T3}},
        ::Fminbox{O} = Fminbox();
        linesearch = nothing,
        alphaguess = nothing,
        x_tol::Real = pt(T1,T2,T3)(eps(pt(T1,T2,T3))),
        f_tol::Real = sqrt(pt(T1,T2,T3)(eps(pt(T1,T2,T3)))),
        g_tol::Real = sqrt(pt(T1,T2,T3)(eps(pt(T1,T2,T3)))),
        allow_f_increases::Bool = true,
        iterations::Integer = 1_000,
        successive_f_tol::Int = 0,
        store_trace::Bool = false,
        show_trace::Bool = false,
        extended_trace::Bool = false,
        callback::TCallback = nothing,
        show_every::Integer = 1,
        eta::Real = convert(pt(T1,T2,T3),0.4),
        mu0::Real = convert(pt(T1,T2,T3), NaN),
        mufactor::Real = convert(pt(T1,T2,T3), 0.0001),
        precondprep = (P, x, l, u, mu) -> precondprepbox!(P, x, l, u, mu),
        optimizer_o::Options{TO,TCallback} = Options{pt(T1,T2,T3),Void}(x_tol, f_tol, g_tol, 0, 0, 0, allow_f_increases, successive_f_tol, iterations, store_trace, show_trace, extended_trace, show_every, callback, pt(T1,T2,T3)(NaN)),
        nargs...) where {T1<:AbstractFloat,T2<:AbstractFloat,T3<:AbstractFloat,O<:AbstractOptimizer, TO<:AbstractFloat, TCallback}

    O == Newton && warn("Newton is not supported as the inner optimizer. Defaulting to ConjugateGradient.")
    T = pt(typeof(T1(1)/1),typeof(T2(1)/1),typeof(T3(1)/1))
    if !(typeof(_l) <: AbstractArray)
        l = fill(_l, length(initial_x))
    else
        l = _l
    end
    if !(typeof(_u) <: AbstractArray)
        u = fill(_u, length(initial_x))
    else
        u = _u
    end
    Tx = pt(typeof(initial_x), typeof(l), typeof(u))
    if isnan(T(1)/T(Inf)) || isnan(T(Inf) - 1)
        for i in 1:length(l)
            l[i] = max(-sqrt(realmax(T)/2), l[i])
            u[i] = min(sqrt(realmax(T)/2), u[i])
        end
    end
    x = copy(initial_x)
    fbarrier = (gbarrier, x) -> barrier_box(gbarrier, x, l, u)
    fb = (gbarrier, x, gfunc) -> function_barrier(gfunc, gbarrier, x, df.fdf, fbarrier)
    gfunc = similar(x)
    gbarrier = similar(x)
    P = InverseDiagonal(ones(initial_x))
    # to be careful about one special case that might occur commonly
    # in practice: the initial guess x is exactly in the center of the
    # box. In that case, gbarrier is zero. But since the
    # initialization only makes use of the magnitude, we can fix this
    # by using the sum of the absolute values of the contributions
    # from each edge.
    boundaryidx = Array{Int,1}()
    for i = 1:length(gbarrier)
        thisx = x[i]
        thisl = l[i]
        thisu = u[i]

        if thisx == thisl
            thisx = T(0.99)*thisl+T(0.01)*thisu
            x[i] = thisx
            push!(boundaryidx,i)
        elseif thisx == thisu
            thisx = T(0.01)*thisl+T(0.99)*thisu
            x[i] = thisx
            push!(boundaryidx,i)
        elseif thisx < thisl || thisx > thisu
            error("Initial position must be inside the box")
        end

        xl = max(thisx - thisl, sqrt(realmin(T)))
        ux = max(thisu - thisx, sqrt(realmin(T)))
        gbarrier[i] = (isfinite(thisl) ? one(T)/xl : zero(T)) + (isfinite(thisu) ? one(T)/ux : zero(T))
    end
    if length(boundaryidx) > 0
        warn("Initial position cannot be on the boundary of the box. Moving elements to the interior.\nElement indices affected: $boundaryidx")
    end
    df.df(gfunc, x)
    mu = isnan(mu0) ? initial_mu(gfunc, gbarrier; mu0factor=mufactor) : mu0
    if show_trace > 0
        println("######## fminbox ########")
        println("Initial mu = ", mu)
    end

    g = similar(x)
    fval_all = Array{Vector{T}}(0)

    # Count the total number of outer iterations
    iteration = 0

    xold = similar(x)
    converged = false
    local results
    first = true
    fval0 = zero(T)
    while !converged && iteration < iterations
        # Increment the number of steps we've had to perform
        iteration += 1

        copy!(xold, x)
        # Optimize with current setting of mu
        funcc = (g, x) -> barrier_combined(gfunc, gbarrier,  g, x, fb, mu)
        fval0 = funcc(nothing, x)
        dfbox = OnceDifferentiable(x->funcc(nothing, x), (g, x)->(funcc(g, x); g), funcc, initial_x, zero(T))
        if show_trace > 0
            println("#### Calling optimizer with mu = ", mu, " ####")
        end
        pcp = (P, x) -> precondprep(P, x, l, u, mu)
        # TODO: Changing the default linesearch and alphaguesses
        #       in the optimization algorithms will imply a lot of extra work here
        if O == ConjugateGradient || O == Newton
            if linesearch == nothing
                linesearch = LineSearches.HagerZhang{T}()
            end
            if alphaguess == nothing
                alphaguess = LineSearches.InitialHagerZhang{T}()
            end
            _optimizer = ConjugateGradient(eta = eta, alphaguess = alphaguess,
                                           linesearch = linesearch, P = P, precondprep = pcp)
        elseif O == LBFGS
            if linesearch == nothing
                linesearch = LineSearches.HagerZhang{T}()
            end
            if alphaguess == nothing
                alphaguess = LineSearches.InitialStatic{T}()
            end
            _optimizer = O(alphaguess = alphaguess, linesearch = linesearch, P = P, precondprep = pcp)
        elseif O == BFGS
            if linesearch == nothing
                linesearch = LineSearches.HagerZhang{T}()
            end
            if alphaguess == nothing
                alphaguess = LineSearches.InitialStatic{T}()
            end
            _optimizer = O(alphaguess = alphaguess, linesearch = linesearch)
        elseif O == GradientDescent
            if linesearch == nothing
                linesearch = LineSearches.HagerZhang{T}()
            end
            if alphaguess == nothing
                alphaguess = LineSearches.InitialPrevious(alpha=one(T), alphamin=zero(T), alphamax=T(Inf))
            end
            _optimizer = O(T, alphaguess = alphaguess, linesearch = linesearch, P = P, precondprep = pcp)
        elseif O in (NelderMead, SimulatedAnnealing)
            _optimizer = O()
        else
            if linesearch == nothing
                linesearch = LineSearches.HagerZhang{T}()
            end
            if alphaguess == nothing
                alphaguess = LineSearches.InitialPrevious{T}()
            end
            _optimizer = O(alphaguess = alphaguess, linesearch = linesearch)
        end
        resultsnew = optimize(dfbox, x, _optimizer, optimizer_o)
        if first
            results = resultsnew
            first = false
        else
            append!(results, resultsnew)
        end
        copy!(x, minimizer(results))
        if show_trace > 0
            println("x: ", x)
        end

        # Decrease mu
        mu *= mufactor

        # Test for convergence
        g .= gfunc .+ mu.*gbarrier

        results.x_converged, results.f_converged, results.g_converged, converged, f_increased = assess_convergence(x, xold, minimum(results), fval0, g, x_tol, f_tol, g_tol)
        f_increased && !allow_f_increases && break
    end
    #_x_abschange = vecnorm(x - xold)
    _x_abschange = maxdiff(x,xold)
    _minimizer = minimizer(results)
    _minimum = minimum(results)
    _results::MultivariateOptimizationResults{Fminbox{O}, T, Tx, typeof(_x_abschange), typeof(_minimum), typeof(results.trace)} = MultivariateOptimizationResults{Fminbox{O}, T, Tx, typeof(_x_abschange), typeof(_minimum), typeof(results.trace)}(
        Fminbox{O}(), false, initial_x, _minimizer, _minimum,
        iteration, results.iteration_converged,
        results.x_converged, results.x_tol, _x_abschange,
        results.f_converged, results.f_tol, f_abschange(_minimum, fval0),
        results.g_converged, results.g_tol, maximum(abs, g),
        results.f_increased, results.trace, results.f_calls,
        results.g_calls, results.h_calls)
    return _results
end

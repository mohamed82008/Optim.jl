f_abschange(d::AbstractObjective, state) = f_abschange(value(d), state.f_x_previous)
f_abschange(f_x::T, f_x_previous) where T = abs(f_x - f_x_previous)
x_abschange(state) = x_abschange(state.x, state.x_previous)
x_abschange(x, x_previous) = maxdiff(x, x_previous)
g_residual(d::AbstractObjective) = g_residual(gradient(d))
g_residual(d::NonDifferentiable) = convert(typeof(value(d)), NaN)
g_residual(g) = maximum(abs, g)
gradient_convergence_assessment(state::AbstractOptimizerState, d, options) = g_residual(gradient(d)) ≤ options.g_tol
gradient_convergence_assessment(state::ZerothOrderState, d, options) = false

# Default function for convergence assessment used by
# AcceleratedGradientDescentState, BFGSState, ConjugateGradientState,
# GradientDescentState, LBFGSState, MomentumGradientDescentState and NewtonState
function default_convergence_assessment(state::AbstractOptimizerState, d, options)
    x_converged, f_converged, f_increased, g_converged = false, false, false, false

    # TODO: Create function for x_convergence_assessment
    if x_abschange(state.x, state.x_previous) ≤ options.x_tol
        x_converged = true
    end

    # Relative Tolerance
    # TODO: Create function for f_convergence_assessment
    f_x = value(d)
    if f_abschange(f_x, state.f_x_previous) ≤ options.f_tol*abs(f_x)
        f_converged = true
    end

    if f_x > state.f_x_previous
        f_increased = true
    end

    g_converged = gradient_convergence_assessment(state,d,options)

    converged = x_converged || f_converged || g_converged

    return x_converged, f_converged, g_converged, converged, f_increased
end

# Used by Fminbox and IPNewton
function assess_convergence(x,
                            x_previous,
                            f_x,
                            f_x_previous,
                            g,
                            x_tol,
                            f_tol,
                            g_tol)
    x_converged, f_converged, f_increased, g_converged = false, false, false, false

    if x_abschange(x, x_previous) ≤ x_tol
        x_converged = true
    end

    # Absolute Tolerance
    # if abs(f_x - f_x_previous) < f_tol
    # Relative Tolerance
    if f_abschange(f_x, f_x_previous) ≤ f_tol*abs(f_x)
        f_converged = true
    end

    if f_x > f_x_previous
        f_increased = true
    end

    if g_residual(g) ≤ g_tol
        g_converged = true
    end

    converged = x_converged || f_converged || g_converged

    return x_converged, f_converged, g_converged, converged, f_increased
end

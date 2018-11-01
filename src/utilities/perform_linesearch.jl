# Reset the search direction if it becomes corrupted
# return true if the direction was changed
reset_search_direction!(state, d, method) = false # no-op

function reset_search_direction!(state, d, method::BFGS)
    copy!(state.invH, method.initial_invH(state.x))
    state.s .= .-gradient(d)
    return true
end

function reset_search_direction!(state, d, method::LBFGS)
    state.pseudo_iteration = 1
    state.s .= .-gradient(d)
    return true
end

function reset_search_direction!(state, d, method::ConjugateGradient)
    state.s .= .-state.pg
    return true
end

function perform_linesearch!(state, method, d)
    # Calculate search direction dphi0
    any(isnan, gradient(d)) && (@show gradient(d); error("gradient(d) is nan"))

    dphi_0 = real(vecdot(gradient(d), state.s))
    # reset the direction if it becomes corrupted
    if dphi_0 >= zero(dphi_0) && reset_search_direction!(state, d, method)
        dphi_0 = real(vecdot(gradient(d), state.s)) # update after direction reset
    end
    phi_0  = value(d)

    # Guess an alpha
    alpha_2 = state.alpha
    method.alphaguess!(method.linesearch!, state, phi_0, dphi_0, d)
    if !isfinite(state.alpha)
        state.alpha = method.linesearch!.alphamax
        Base.warn("Linesearch initialization failed, using alpha = $(state.alpha) and exiting optimization.")
    end
    if :alphamax ∈ fieldnames(typeof(method.linesearch!))
        state.alpha = NaNMath.min(state.alpha, method.linesearch!.alphamax)
    end
    isfinite(state.alpha) || return false
    
    # Store current x and f(x) for next iteration
    state.f_x_previous = phi_0
    copy!(state.x_previous, state.x)

    any(isnan, state.s) && (@show state.s; error("state.s is nan"))

    # Perform line search; catch LineSearchException to allow graceful exit
    try
        state.alpha, ϕalpha =
            method.linesearch!(d, state.x, state.s, state.alpha,
                               state.x_ls, phi_0, dphi_0)
        isnan(ϕalpha) && (@show state.alpha, ϕalpha; println("ϕalpha is nan"))
        return true # lssuccess = true
    catch ex
        if isa(ex, LineSearches.LineSearchException)
            state.alpha = ex.alpha
            Base.warn("Linesearch failed, using alpha = $(state.alpha) and exiting optimization.\nThe linesearch exited with message:\n$(ex.message)")
            return false # lssuccess = false
        else
            rethrow(ex)
        end
    end
end

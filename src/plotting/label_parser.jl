function _string_ic_inline(ic::TOVSolvers.AbstractInitialCondition)
    "r0 = $(ic.r0), m = $(ic.m), m' = $(ic.m_prime)"
end

function _string_ic_inline_short(ic::TOVSolvers.AbstractInitialCondition)
    "$(ic.r0), $(ic.m), $(ic.m_prime)"
end

function ic_label(ic::TOVSolvers.AbstractInitialCondition, verbose=false)
    if verbose 
        return _string_ic_inline_short(ic)
    else
        return _string_ic_inline(ic)
    end
end

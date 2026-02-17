#using TOVSolvers
#using TestItemRunner

#@run_package_tests verbose=true

using Test
using TOVSolvers

@testset "TOVSolvers.jl" begin
    
    @testset "Parameters" begin
        # Valid parameters
        params = TOVParameters(0.1, 3.0)
        @test params.omega == 0.1
        @test params.x == 3.0
        
        # Test slow manifold
        r = 1.0
        m_slow = slow_manifold(r, params)
        @test m_slow ≈ 0.1/3.0 * r^3
        
        # Invalid parameters
        @test_throws ArgumentError TOVParameters(-0.1, 3.0)
        @test_throws ArgumentError TOVParameters(0.1, -3.0)
    end
    
    @testset "Initial Conditions - RegularOrigin" begin
        # Case 1: m=0, m'≠0
        ic1 = RegularOrigin(m=0.0, m_prime=1.0)
        @test ic1.m == 0.0
        @test ic1.m_prime == 1.0
        @test ic1.r0 == TOVSolvers.DEFAULT_R_ZERO
        
        # Case 2: m≠0, m'=0
        ic2 = RegularOrigin(m=0.5, m_prime=0.0)
        @test ic2.m == 0.5
        @test ic2.m_prime == 0.0
        
        # Case 3: m=0, m'=0
        ic3 = RegularOrigin(m=0.0, m_prime=0.0)
        @test ic3.m == 0.0
        @test ic3.m_prime == 0.0
        
        # Test get_initial_state
        u0 = get_initial_state(ic1)
        @test u0[1] == 0.0
        @test u0[2] == 1.0
    end
    
    @testset "Initial Conditions - AdmissibleCrossing" begin
        params = TOVParameters(0.1, 3.0)
        
        # Valid crossing
        r_h = 1.0
        ic = AdmissibleCrossing(r_h, crossing=:both, params=params)
        @test ic.r_h == 1.0
        @test ic.m_h == 0.5
        @test ic.crossing == :both
        
        # Test different crossing types
        ic_plus = AdmissibleCrossing(r_h, crossing=:plus, params=params)
        @test ic_plus.crossing == :plus
        
        ic_minus = AdmissibleCrossing(r_h, crossing=:minus, params=params)
        @test ic_minus.crossing == :minus
        
        # Invalid crossing type
        @test_throws ArgumentError AdmissibleCrossing(r_h, crossing=:invalid, params=params)
    end
    
    @testset "Admissible Crossings" begin
        params = TOVParameters(0.1, 3.0)
        r_h = 1.0
        m_h = r_h / 2.0
        
        # Test discriminant computation
        Δ = compute_discriminant(r_h, m_h, params)
        @test Δ >= 0  # Should be non-negative for admissible crossing
        
        # Test crossing slopes
        m_plus, m_minus = compute_crossing_slopes(r_h, m_h, params)
        @test !isnan(m_plus)
        @test !isnan(m_minus)
        @test m_plus != m_minus  # Should be different
        
        # Test finding admissible crossings
        crossings = find_admissible_crossings(params, 0.5:0.5:3.0, n_points=10)
        @test length(crossings) > 0
        @test all(c -> c isa AdmissibleCrossing, crossings)
    end
    
    @testset "Equations" begin
        params = TOVParameters(0.1, 3.0)
        r = 1.0
        m = 0.3
        m_prime = 0.5
        
        # Test H computation
        H = compute_H(r, m, m_prime, params)
        @test !isnan(H)
        @test !isinf(H)
        
        # Test horizon detection
        @test !is_at_horizon(1.0, 0.3)  # Not at horizon
        @test is_at_horizon(1.0, 0.5, rtol=1e-10)  # At horizon
        
        # Test slow manifold detection
        m_slow = slow_manifold(r, params)
        @test is_on_slow_manifold(r, m_slow, params)
        @test !is_on_slow_manifold(r, m, params)
    end
    
    @testset "Problem Definition" begin
        params = TOVParameters(0.1, 3.0)
        ic = RegularOrigin(m=0.0, m_prime=1.0)
        
        # Basic problem
        prob = TOVProblem(params, ic, (0.1, 10.0))
        @test prob.params === params
        @test prob.initial_condition === ic
        @test prob.r_span == (0.1, 10.0)
        @test prob.direction == :forward
        
        # Convenience constructor
        prob2 = TOVProblem(params, ic, 5.0)
        @test prob2.r_span == (TOVSolvers.DEFAULT_R_ZERO, 5.0)
        @test prob2.direction == :forward
        
        # Backward problem
        ic_horizon = AdmissibleCrossing(2.0, crossing=:plus, params=params)
        prob_back = TOVProblem(params, ic_horizon, 0.0)
        @test prob_back.direction == :backward
    end
    
    @testset "Basic Solving - RegularOrigin" begin
        params = TOVParameters(1e-3, 3.0)
        ic = RegularOrigin(m=0.0, m_prime=1.0)
        
        # Solve with default config
        config = TOVSolverConfig(maxiters=10^5)  # Limit iterations for test
        sol = solve(params, ic, 10.0, config)
        
        @test sol isa TOVSolution
        @test length(sol) > 0
        @test sol.params === params
        @test sol.initial_condition === ic
        
        # Check that solution starts at origin
        @test sol.r[1] ≈ 0.0 atol=1e-6
        @test sol.m[1] ≈ 0.0 atol=1e-6
        @test sol.m_prime[1] ≈ 1.0 atol=1e-3
    end
    
    @testset "Solution Classification" begin
        params = TOVParameters(0.1, 3.0)
        ic_origin = RegularOrigin(m=0.0, m_prime=1.0)
        
        config = TOVSolverConfig(maxiters=10^5)
        sol = solve(params, ic_origin, 5.0, config)
        
        classification = classify_solution(sol)
        @test classification in [:regular_origin_to_horizon, :regular_origin_incomplete]
    end
    
    @testset "Solver Configuration" begin
        # Default config
        config1 = TOVSolverConfig()
        @test config1.abstol == 1e-10
        @test config1.reltol == 1e-10
        @test config1.save_everystep == true
        
        # Custom config
        config2 = TOVSolverConfig(
            abstol=1e-8,
            reltol=1e-8,
            horizon_tolerance=1e-5
        )
        @test config2.abstol == 1e-8
        @test config2.reltol == 1e-8
        @test config2.horizon_tolerance == 1e-5
    end
    
end

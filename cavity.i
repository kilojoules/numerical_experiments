Simulations:
  - name: sim1
    time_integrator: ti_1
    optimizer: opt1

linear_solvers:

  - name: solve_scalar
    type: tpetra
    method: gmres
    preconditioner: sgs
    tolerance: 1e-3
    max_iterations: 50
    kspace: 50
    output_level: 0

  - name: solve_cont
    type: tpetra
    method: gmres
    preconditioner: muelu
    tolerance: 1e-5
    max_iterations: 50
    kspace: 50
    output_level: 0

realms:

  - name: realm_1
    mesh: ./grid100.exo
    use_edges: yes


    time_step_control:
     target_courant: 100.0
     time_step_change_factor: 1.2
   
    equation_systems:
      name: theEqSys
      max_iterations: 2 

      solver_system_specification:
        velocity: solve_scalar
        turbulent_ke: solve_scalar
        specific_dissipation_rate: solve_scalar
        pressure: solve_cont

      systems:

        - LowMachEOM:
            name: myLowMach
            max_iterations: 1
            convergence_tolerance: 1e-5

        - ShearStressTransport:
            name: mySST 
            max_iterations: 1
            convergence_tolerance: 1e-5

    initial_conditions:
      - constant: ic_1
        target_name: doms-QUAD
        value:
          pressure: 0
          velocity: [1.0,0.0]
          turbulent_ke: 1.125e-6
          specific_dissipation_rate: 7.5

    material_properties:
      target_name: doms-QUAD
      specifications:
        - name: density
          type: constant
          value: 1.177
        - name: viscosity
          type: constant
          value: 1.846e-5

    boundary_conditions:

    - wall_boundary_condition: bc_wall
      target_name: walls
      wall_user_data:
        velocity: [0,0]
        use_wall_function: yes

    - inflow_boundary_condition: bc_inflow
      target_name: top
      inflow_user_data:
        velocity: [95.0,0.0]
        turbulent_ke: 1.125e-6
        specific_dissipation_rate: 7.5

    solution_options:
      name: myOptions
      turbulence_model: SST_DES

      options:
        - hybrid_factor:
            velocity: 1.0 
            turbulent_ke: 1.0
            specific_dissipation_rate: 1.0

        - alpha_upw:
            velocity: 1.0 

        - limiter:
            pressure: no
            velocity: yes
            turbulent_ke: yes
            specific_dissipation_rate: yes

        - projected_nodal_gradient:
            velocity: element
            pressure: element 
            turbulent_ke: element
            specific_dissipation_rate: element
    


    post_processing:
    
    - type: surface
      physics: surface_force_and_moment_wall_function
      output_file_name: results/test.dat
      frequency: 25 
      parameters: [0,0]
      target_name: walls

    output:
      output_data_base_name: results/test.e
      output_frequency: 10
      output_node_set: no 
      output_variables:
       - velocity
       - pressure
       - pressure_force
       - tau_wall
       - turbulent_ke
       - specific_dissipation_rate
       - minimum_distance_to_wall
       - sst_f_one_blending
       - turbulent_viscosity
       - vorticity

    turbulence_averaging:
      time_filter_interval: 10.0
      specifications:
        - name: one
          target_name: doms-QUAD
          reynolds_averaged_variables:
            - velocity
            - turbulent_ke
          favre_averaged_variables:
            - velocity
            - turbulent_ke

          compute_reynolds_stress: yes
          compute_favre_stress: yes
          compute_favre_tke: yes
          compute_q_criterion: yes
          compute_vorticity: yes
          compute_lambda_ci: yes

    restart:
      restart_data_base_name: restart/S809_AoA0_SST_F.rst
      output_frequency: 2500
     
Time_Integrators:
  - StandardTimeIntegrator:
      name: ti_1
      start_time: 0
      time_step: 1.0e-10
      termination_time: 100.0 
      time_stepping_type: adaptive
      time_step_count: 0

      realms: 
        - realm_1

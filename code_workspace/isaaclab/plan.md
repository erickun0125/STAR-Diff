# STAR-Diff Future Implementation Plan

 This report outlines the detailed plan for implementing the remaining components of the STAR-Diff project, specifically focusing on the Data Collection pipeline and Policy Training/Evaluation.

## 1. Demo Collection Infrastructure

The goal is to automate the collection of expert demonstrations using a Rule-Based Expert (Finite State Machine).

### A. Trajectory Generator ([generator.py](file:///home/eric/eric_workspace/IsaacLab/source/star_diff/star_diff/demo_collector/trajectory/generator.py))
**Objective**: Generate smooth path waypoints between two poses in Cartesian space.

**Requirements**:
- **Inputs**: Start Pose $(p_0, q_0)$, End Pose $(p_1, q_1)$, Step Count $T$.
- **Logic**:
    - **Position**: Linear interpolation $p(t) = p_0 + t(p_1 - p_0)$.
    - **Orientation**: Spherical Linear Interpolation (Slerp) for quaternions.
    - **Optional**: Support for circular paths (for specific surgical maneuvers like suturing, though Peg Transfer is mostly linear).

### B. Surgical Finite State Machine ([surgical_fsm.py](file:///home/eric/eric_workspace/IsaacLab/source/star_diff/star_diff/demo_collector/fsm/surgical_fsm.py))
**Objective**: Control the robot to perform the Peg Transfer task autonomously using the [RCMAwareAction](file:///home/eric/eric_workspace/IsaacLab/source/star_diff/star_diff/envs/mdp/actions/action.py#52-239) space.

**States**:
1.  **IDLE**: Wait for initialization.
2.  **APPROACH_PEG**: Move EE to a waypoint above the target peg.
3.  **DESCEND**: Move EE vertically down to grasp the peg.
4.  **GRASP**: Activate gripper (close jaws).
5.  **LIFT**: Move EE vertically up with the peg.
6.  **MOVE_TO_TARGET**: Move EE to a waypoint above the placement target.
7.  **DESCEND_PLACE**: Lower the peg to the target.
8.  **RELEASE**: Deactivate gripper (open jaws).
9.  **RETRACT**: Move EE up and return to IDLE/Done.

**Logic Flow**:
- In each `run(obs)` step, the FSM checks the current sub-goal status.
- If a sub-trajectory is finished, it transitions to the next state and plans a new trajectory using [TrajectoryGenerator](file:///home/eric/eric_workspace/IsaacLab/source/star_diff/star_diff/demo_collector/trajectory/generator.py#4-8).
- Outputs actions compatible with [RCMAwareAction](file:///home/eric/eric_workspace/IsaacLab/source/star_diff/star_diff/envs/mdp/actions/action.py#52-239) (specifically Variant 4: $\gamma_{rel}, p_{abs}$).

### C. Collection Script ([collect_demos.py](file:///home/eric/eric_workspace/IsaacLab/scripts/star_diff/collect_demos.py))
**Objective**: Orchestrate the environment and FSM to save data.

**Implementation Steps**:
1.  **Environment Loop**:
    - Reset [FR5SurgicalEnv](file:///home/eric/eric_workspace/IsaacLab/source/star_diff/star_diff/envs/tasks/fr5_surgical_env_cfg.py#149-163).
    - Loop until `num_demos` is reached.
2.  **Data Recording**:
    - Extract observations defined in [ObservationsCfg](file:///home/eric/eric_workspace/IsaacLab/source/star_diff/star_diff/envs/tasks/fr5_surgical_env_cfg.py#89-133) (Policy State, RGB Images).
    - Store data in **HDF5** format compatible with Robomimic/Minari.
    - **Key Keys**: `data/demo_{i}/obs/agentview_image`, `data/demo_{i}/obs/robot0_eef_pos`, `data/demo_{i}/actions`.
3.  **Validation**: Ensure recorded trajectories satisfy RCM constraints (implicitly handled by Env, but good to check).

## 2. Policy Training & Evaluation

### A. Dataset Processing
- Verify HDF5 structure matches [DiffusionPolicyCfg](file:///home/eric/eric_workspace/IsaacLab/source/star_diff/star_diff/configs/policy_cfg.py#7-23) expectations.
- Create a configuration file pointing to the collected dataset.

### B. Training Script ([train_policy.py](file:///home/eric/eric_workspace/IsaacLab/scripts/star_diff/train_policy.py))
**Objective**: Train the Diffusion Policy using the collected data.

**Plan**:
1.  **Load Config**: Parse [DiffusionPolicyCfg](file:///home/eric/eric_workspace/IsaacLab/source/star_diff/star_diff/configs/policy_cfg.py#7-23).
2.  **Initialize Policy**: Instantiate `DiffusionWrapper` (Robomimic interface).
3.  **Training Loop**:
    - Load HDF5 DataLoader.
    - Feed batches to the policy.
    - Log loss (Noise prediction error).
    - Save checkpoints periodically.

### C. Evaluation Script ([eval_policy.py](file:///home/eric/eric_workspace/IsaacLab/scripts/star_diff/eval_policy.py))
**Objective**: Test the trained policy in the simulation.

**Plan**:
1.  **Load Checkpoint**: Load the best model.
2.  **Env Interaction**:
    - Reset Env.
    - Get Observation -> Feed to Policy -> Get Action string.
    - Execute Action.
3.  **Metrics**:
    - Success Rate (Peg successfully transferred).
    - RCM Violation Error (Mean distance from trocar).

## 3. Summary of Code State (Post-Fix)
the following fixes have been applied to the codebase and should be preserved:
- **[action.py](file:///home/eric/eric_workspace/IsaacLab/source/star_diff/star_diff/envs/mdp/actions/action.py)**: Fixed [RCMAwareAction](file:///home/eric/eric_workspace/IsaacLab/source/star_diff/star_diff/envs/mdp/actions/action.py#52-239) to correctly invoke [OperationalSpaceController](file:///home/eric/eric_workspace/IsaacLab/source/star_diff/star_diff/controller/operational_space_controller.py#4-53) with a 7D pose concatenated from position and quaternion.
- **[fr5_surgical_env_cfg.py](file:///home/eric/eric_workspace/IsaacLab/source/star_diff/star_diff/envs/tasks/fr5_surgical_env_cfg.py)**: Added [ObservationsCfg](file:///home/eric/eric_workspace/IsaacLab/source/star_diff/star_diff/envs/tasks/fr5_surgical_env_cfg.py#89-133) to provide necessary states (Joints, EE Pose, Trocar Pose, Camera) for the policy and FSM.
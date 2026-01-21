### Trocar-generalized & RCM-aware Generative Policy - Ver1

# RCM-aware Generative Policy for Laparoscopic Surgery Robust to Variable Trocar Points

## 1. Introduction

### 1.1 Background

- 기존의 **Laparoscopic Surgery Automation** 연구는 Robot Base와 **Trocar(Port)** 위치를 상수로 가정한 고정된 환경(Fixed Setting)에서 수행됨.
- 실제 수술 환경에서는 환자의 체형, 수술 부위, 집도의의 선호도에 따라 **Trocar Placement**가 매번 달라짐.
- 고정된 환경에서 학습된 Policy는 **Variable Trocar Layout**에서 **RCM(Remote Center of Motion)** 제약 조건을 위반하거나 **Workspace** 불일치로 인해 실패할 확률이 높음.

### 1.2 Research Objective

- **6-DoF Manipulator (without EndoWrist)** 시스템을 대상으로, 임의의 Trocar 위치에서도 재학습 없이 동작 가능한 **Single Generative Policy** 개발.
- **Joint State**를 배제하고 **Task Space** 기반으로 학습하여 **Cross-Embodiment**를 달성하고, **Simulation-based Pre-training**을 통해 **Sim-to-Real** 확장성을 확보함.
- Cross-Embodiment는 단순 서로 다른 manipulator 사이의 transfer만이 아닌, 사람이 직접 수집한 dataset으로 학습이 가능하도록 확장 가능함.

---

## 2. Problem Formulation

### 2.1 Assumptions

- **Feasible Workspace Assumption**: End effector가 기구학적으로 도달 가능한 **Feasible Workspace** 내에서만 정의됨.
- **IK Solution Assumption**: **RCM Constraint** 하에서 **Inverse Kinematics (IK)** 해가 존재함
    - 만약, IK 해가 존재하지 않으면(Unreachable), 학습 데이터 생성 시 **Fail** 처리하여 배제함.
- **Trocar position Assumption**: Fixed radius R의 Hemishpere 상에 있다고 가정
    - 필요 시, R도 variable로 취급해서 임의 위치의 trocar 위치를 다뤄야 할 수도 있음.
- **Camera fixed Assuamption**: Camera의 위치는 고정되어있다고 가정함.
    - 필요 시, Camera의 pose도 variable로 취급하여 camera port에 대한 trocar 위치를 observation에 추가해야할 수도 있음.

### 2.2 Kinematic Modeling of RCM

![image.png](attachment:9a5d8053-49cb-4cca-8853-2baab78fc305:image.png)

- **Coordinate Frames**: World Frame $\{S\}$, Hemisphere Center Frame $\{C\}$, EE Base Frames $\{l_0, r_0\}$
- **Trocar Point**: Trocar 덮개(Hemisphere) 상의 임의의 점으로 정의 → $\lambda_{Trocar} = \{\theta, \phi\}$.
- **RCM Variable**: **RCM**을 Virtual Spherical Joint (S)와 그 중심의 Virtual Prismatic Joint (P)의 결합으로 모델링
    
    $x_{RCM} = \{d, R\}$
    
    - $d$: **Insertion Depth** (Prismatic component). ($d \in \mathbb{R}$)
    - $R$: **Rotation Matrix** relative to RCM (Spherical component). ($R \in SO(3)$)

---

## 3. Proposed Methodology

### 3.1 Action Space Design: $x_{RCM}, g$

**Cartesian Space, Joint Space** 대신 **RCM-centric Action Space  $x_{RCM}$**를 정의하여 기구학적 제약 조건을 내재화함.

**Binary Gripper state $g$**를 사용함**.**

- Action Definition: $a_t = \{x_{RCM}, g\}$
    - $R$은 **6D Rotation Representation**을 사용하여 6차원 벡터로 인코딩 후 그람-슈미트 하여 출력.
- **Policy Output**:
    - Generative Policy는 Horizon $H$ 동안의 Sequence $A_t = { a_{t}, \dots, a_{t+H} }$를 생성함.

### 3.2 Observation Space Design: $I_t, x_{RCM}, \lambda_{Trocar}$

Image, Current RCM variable, Trocar position을 사용함.

- Observation Definition: $o_t = \{I_t, x_{RCM}\}$
- Condition Definition: $c_t = \{\lambda_{Trocar}\}$

### 3.3 Network Architecture: Geometry-Conditioned Policy via FiLM

**Trocar Parameters**를 단순 입력이 아닌, Visual Feature를 변조(Modulate)하는 조건으로 활용.

- **Inputs**:
    - **Observation**: Endoscope Image ($I_t$), Proprioception (Current $x_{RCM}$).
    - **Condition**: Trocar Configuration $\lambda_{Trocar}$.
- **Architecture**:
    - **Visual Encoder**: ResNet 또는 ViT Backbone 사용.
    - **FiLM Generator**: $\lambda_{Trocar}$를 MLP에 통과시켜 Affine Parameter $\gamma$ (Scale), $\beta$ (Shift) 생성.
    - FiLM Layer (Feature-wise Linear Modulation):
        
        $Feature_{new} = \gamma(\lambda_{Trocar}) \cdot Feature_{img} + \beta(\lambda_{Trocar})$
        
        이미지 Feature Map에 Trocar 정보를 주입하여, **Variable Trocar Layout**에 따른 Visual Feature의 해석을 동적으로 조절.
        
    - **Decoder**: Conditioning된 Feature를 기반으로 Action Sequence 생성 (Diffusion or Transformer).
- **Outputs**:
    - Action sequence: $A_t = { a_{t}, \dots, a_{t+H} }$

### 3.4 Data Generation Strategy

**Simulation** 환경에서 **Rule-based Planner**를 통해 데이터를 생성.

- **Procedure**:
    1. **Configuration Sampling**: Trocar 위치 및 Task Goal 무작위 샘플링.
    2. **Path Planning & IK Check**: 생성된 경로에 대해 **RCM-IK** 수행.
    3. **Feasibility Check**: **Reachability**, **Joint Limits**, **Singularity** 검증.
    4. **Filtering**: **Feasible Workspace**를 벗어나는 경우 해당 Episode를 데이터셋에서 제거.

---

## 4. Research Plan

### Phase 1: Simulation & Kinematics Engine Setup

- **Simulation**: 현재 상황에 가장 적합한 기존 Simulation Env 찾기.(MuJoCo/Isaac Sim 기반의 **Variable Trocar Environment** 구축이 가능해야 함.)
- **RCM Kinematics aware Low Level Control**: HL policy → (RCM var) → Forward/Inverse Kinematics Solver → (desired joint or pose) → LL controller
- **Trocar Environment:** Bimanual FR5, Trocar, Task related object settings, …

### Phase 2: Feasible Dataset Generation

- **Target Task**: Peg Transfer (Initial), Suturing (Advanced).
- **Data Pipeline**:
    - Rule based Trajectory Planner를 이용한 대규모 **Synthetic Data** 생성.
    - **Feasible Workspace** 검증 로직을 통과한 성공 궤적(Success Trajectory)만 선별하여 구축.

### Phase 3: Policy Learning & Evaluation

- **Implementation**: **FiLM-conditioned Generative Policy(ACT vs DP)** 모델 구현 및 학습.
- **Evaluation**: Unseen Trocar Configuration에서의 **Task Success Rate** 측정 및 **Generalization** 성능 비교 (vs. Non-conditioned Policy).
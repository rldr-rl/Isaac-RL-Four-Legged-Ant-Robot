"""Launch Isaac Sim Simulator first."""
import argparse
import sys
from isaaclab.app import AppLauncher

import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with rsl-rl.")
parser.add_argument("--video", action="store_true", default=True, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=2048, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default='PAnt-v0', help="Name of the task.")
parser.add_argument("--seed", type=int, default=0, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--distributed", action="store_true", default=False, help="Record videos during training.")

# append AppLauncher cli args
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""
import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""
import math
import torch
import gymnasium as gym
import os
from datetime import datetime

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.assets import ArticulationCfg, AssetBaseCfg

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg

import isaaclab.envs.mdp as mdp
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains import TerrainImporterCfg

# PLACEHOLDER: Extension template (do not remove this comment)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

PATH = f"./urdf/Robot.usd"
PANT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            rigid_body_enabled=True,
            max_linear_velocity=10.0,
            max_angular_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "l_leg1_joint001__to__l_leg1_link001": 0.0,
            "l_leg1_joint002__to__l_leg1_link002": 0.0/180*3.14,
            "l_leg1_joint003__to__l_leg1_link003": -90.0/180*3.14,
            "l_leg2_joint001__to__l_leg2_link001": 0.0,
            "l_leg2_joint002__to__l_leg2_link002": 0.0/180*3.14,
            "l_leg2_joint003__to__l_leg2_link003": -90.0/180*3.14,
            "l_leg3_joint001__to__l_leg3_link001": 0.0,
            "l_leg3_joint002__to__l_leg3_link002": -0.0/180*3.14,
            "l_leg3_joint003__to__l_leg3_link003": 90.0/180*3.14,
            "l_leg4_joint001__to__l_leg4_link001": 0.0,
            "l_leg4_joint002__to__l_leg4_link002": -0.0/180*3.14,
            "l_leg4_joint003__to__l_leg4_link003": 90.0/180*3.14,
        },
        pos=(0.0, 0.0, 0.5),
    ),
    actuators={
        "leg1_thigh": ImplicitActuatorCfg(
            joint_names_expr=["l_leg1_joint001__to__l_leg1_link001"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "leg1_knee": ImplicitActuatorCfg(
            joint_names_expr=["l_leg1_joint002__to__l_leg1_link002"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "leg1_calf": ImplicitActuatorCfg(
            joint_names_expr=["l_leg1_joint003__to__l_leg1_link003"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "leg2_thigh": ImplicitActuatorCfg(
            joint_names_expr=["l_leg2_joint001__to__l_leg2_link001"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "leg2_knee": ImplicitActuatorCfg(
            joint_names_expr=["l_leg2_joint002__to__l_leg2_link002"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "leg2_calf": ImplicitActuatorCfg(
            joint_names_expr=["l_leg2_joint003__to__l_leg2_link003"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "leg3_thigh": ImplicitActuatorCfg(
            joint_names_expr=["l_leg3_joint001__to__l_leg3_link001"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "leg3_knee": ImplicitActuatorCfg(
            joint_names_expr=["l_leg3_joint002__to__l_leg3_link002"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "leg3_calf": ImplicitActuatorCfg(
            joint_names_expr=["l_leg3_joint003__to__l_leg3_link003"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "leg4_thigh": ImplicitActuatorCfg(
            joint_names_expr=["l_leg4_joint001__to__l_leg4_link001"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "leg4_knee": ImplicitActuatorCfg(
            joint_names_expr=["l_leg4_joint002__to__l_leg4_link002"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "leg4_calf": ImplicitActuatorCfg(
            joint_names_expr=["l_leg4_joint003__to__l_leg4_link003"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
    },
)

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.5,
    use_cache=False,
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.05, grid_width=0.45, grid_height_range=(0.0, 0.1), platform_width=2.0
        ),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.1, noise_range=(0.02, 0.05), noise_step=0.02, border_width=0.25
        # ),
    },
)

class PAntSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""
    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # # Ground-plane
    # ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot = PANT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)


@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    joint_efforts = mdp.JointPositionActionCfg(asset_name="robot",
                                               use_default_offset=True,
                                               preserve_order=True,
       joint_names=[
        "l_leg1_joint001__to__l_leg1_link001",
        "l_leg1_joint002__to__l_leg1_link002",
        "l_leg1_joint003__to__l_leg1_link003",
        "l_leg2_joint001__to__l_leg2_link001",
        "l_leg2_joint002__to__l_leg2_link002",
        "l_leg2_joint003__to__l_leg2_link003",
        "l_leg3_joint001__to__l_leg3_link001",
        "l_leg3_joint002__to__l_leg3_link002",
        "l_leg3_joint003__to__l_leg3_link003",
        "l_leg4_joint001__to__l_leg4_link001",
        "l_leg4_joint002__to__l_leg4_link002",
        "l_leg4_joint003__to__l_leg4_link003",
    ], scale=0.1)
    # joint_efforts = mdp.JointPositionActionCfg(asset_name="robot",
    #                                            preserve_order=True,
    #    joint_names=[
    #     "l_leg1_joint001__to__l_leg1_link001",
    #     "l_leg1_joint002__to__l_leg1_link002",
    #     "l_leg1_joint003__to__l_leg1_link003",
    #     "l_leg2_joint001__to__l_leg2_link001",
    #     "l_leg2_joint002__to__l_leg2_link002",
    #     "l_leg2_joint003__to__l_leg2_link003",
    #     "l_leg3_joint001__to__l_leg3_link001",
    #     "l_leg3_joint002__to__l_leg3_link002",
    #     "l_leg3_joint003__to__l_leg3_link003",
    #     "l_leg4_joint001__to__l_leg4_link001",
    #     "l_leg4_joint002__to__l_leg4_link002",
    #     "l_leg4_joint003__to__l_leg4_link003",
    # ], scale=5)

@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_pos_prev = ObsTerm(func=mdp.last_action)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands,
                                    params={"command_name": "base_velocity"}
                                    )
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""
    pass

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # # (1) Constant running reward
    # alive = RewTerm(func=mdp.is_alive, weight=1.0)

    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-10.0)
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=5.0,
        params={"command_name": "base_velocity",
                "std": math.sqrt(0.25)}
    )

    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)  # default: -2.5e-7
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)  # default: -0.01

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # bad_root_position = DoneTerm(
    #     func=mdp.bad_orientation,
    #     params={"limit_angle": 80},
    # )
    # bad_root_height = DoneTerm(
    #     func=mdp.root_height_below_minimum,
    #     params={"minimum_height": 0.2},
    # )

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(9.0, 13.0),
        rel_standing_envs=0.01,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-0.0, 0.0),
        ),
    )

@configclass
class PAntEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Ant environment."""

    # Scene settings
    scene: PAntSceneCfg = PAntSceneCfg(num_envs=20, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    # Commands
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz

        self.episode_length_s = 5

@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 1000
    save_interval = 50
    experiment_name = "PAnt-v0"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 512, 512],
        critic_hidden_dims=[512, 512, 512],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

gym.register(
    id="PAnt-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:PAntEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}:PPORunnerCfg",
    },
)

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

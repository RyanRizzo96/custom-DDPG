import argparse
import os

from mpi4py import MPI
import gym

from baselines import logger, bench
from baselines.common.misc_util import boolean_flag
from baselines.ddpg.noise import AdaptiveParamNoiseSpec, OrnsteinUhlenbeckActionNoise,  NormalActionNoise


def run(env_id, seed, noise_type, layer_norm, evaluation, **kwargs):

    # Only rank 0 worker to report results
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)

    # Create envs
    env = gym.make(env_id)
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))

    # If evaluation on DDPG is enabled, create new environment
    if evaluation and rank == 0:
        eval_env = gym.make(env_id)
        eval_env = bench.Monitor(eval_env, os.path.join(logger.get_dir(), 'gym_eval'))

    # Parse noise type
    action_noise = None
    param_noise = None
    nb_actions = env.action_space.shape[-1]  # Example: converts (4,) to 4

    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initia)


def parse_args():
    """
    parse arguments for DDPG training

    :return: (dict) the arguments
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env-id', type=str, default='FetchReach-v1')
    boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'layer-norm', default=True)
    boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--actor-lr', type=float, default=1e-3)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'enable-popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--nb-train-steps', type=int, default=50)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-eval-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-rollout-steps', type=int, default=100)  # per epoch cycle and MPI worker
    # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--noise-type', type=str, default='adaptive-param_0.2')
    parser.add_argument('--num-timesteps', type=int, default=int(10e4))
    boolean_flag(parser, 'evaluation', default=False)
    args = parser.parse_args()
    dict_args = vars(args)

    return dict_args


if __name__ == '__main__':
    args = parse_args()  # parse arguments presented by user
    if MPI.COMM_WORLD.Get_rank() == 0:  # Return the rank of this process in a communicator
        logger.configure()  # run logger only for 0 rank process
    # Run actual script
    run(**args)
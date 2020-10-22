import numpy as np
import torch
import os
from glob import glob
import pickle
import iape.iape.backend.core as core
from iape.iape.backend import make_coin_env, make_procgen_env
from iape.iape.backend.utils import ac_pruner, inner_prunner, ac_prune_remover
import warnings
warnings.filterwarnings('ignore')
import pathlib
from pathlib import Path

def print_fork(x, path):
    print(x)
    with open(path, 'a+') as f:
        f.write(x+'\n')


def inputs_to_tensor(o_t, a_oh_tm1, h_tm1, c_tm1):
    if device is not None:
        tensor_o_t = torch.as_tensor(o_t, dtype=torch.float32).to(device)
        tensor_a_oh_tm1 = torch.as_tensor(a_oh_tm1, dtype=torch.float32).to(device)
        tensor_h_tm1 = torch.as_tensor(h_tm1, dtype=torch.float32).to(device)
        tensor_c_tm1 = torch.as_tensor(c_tm1, dtype=torch.float32).to(device)
    else:
        tensor_o_t = torch.as_tensor(o_t, dtype=torch.float32)
        tensor_a_oh_tm1 = torch.as_tensor(a_oh_tm1, dtype=torch.float32)
        tensor_h_tm1 = torch.as_tensor(h_tm1, dtype=torch.float32)
        tensor_c_tm1 = torch.as_tensor(c_tm1, dtype=torch.float32)
    return tensor_o_t, tensor_a_oh_tm1, tensor_h_tm1, tensor_c_tm1

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--steps', type=int, default=4000000)
    parser.add_argument('--n_env', type=int, default=256)
    parser.add_argument('--overwrite', type=int, default=0)
    parser.add_argument('--env_aug', action='store', type=int, default=0, help='use data augmentation (cutout)')
    parser.add_argument('--env_force_paint_velocity', type=int, default=0, help='force paint_velocity_info flag')
    parser.add_argument('--env', type=str, default='Coinrun-standard')
    parser.add_argument('--env_monotheme', action='store_true', default=False, help='use single env theme')
    parser.add_argument('--exp_name', type=str, default='dev_baseline')
    parser.add_argument('--output_dir', type=str, default='/workdisk/nosnap/procgen/', help='number of training levels')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--difficulty', type=int, default=0)


    args = parser.parse_args()


    root_path = args.output_dir
    base_path = os.path.join(root_path, '{:s}'.format(args.exp_name), '{:s}_s{:d}'.format(args.exp_name, args.seed))
    base_path = os.path.join(base_path, 'param_save/')


    #Instantiate device
    if args.gpu<0 or not torch.cuda.is_available():
        device=torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % (args.gpu))

    #Get actor critic kwargs
    with open(os.path.join(base_path,'ac_kwarg.pkl'), 'rb') as handle:
        ac_kwargs = pickle.load(handle)

    # Get latest model parameters
    model_param_path_candidates = glob(os.path.join(base_path, 'model_vars_*.pth'))
    epochs=[]
    model_param_paths=[]
    for m in model_param_path_candidates:
        try:
            epochs.append(int(m.split('_')[-1].split('.pth')[0]))
            model_param_paths.append(m)
        except ValueError:
            continue
    sort_idx =np.argsort(epochs)
    epochs=[epochs[i] for i in sort_idx]
    model_param_paths=[model_param_paths[i] for i in sort_idx]

    if args.env =='Coinrun-standard':

        env = make_coin_env(args.n_env, game_type='standard', set_seed=args.seed+7777,
                          num_levels=500, use_data_augmentation=bool(args.env_aug),
                          single_theme=args.env_monotheme, )
    else:
        env = make_procgen_env(args.n_env, start_level=5000, game_type=args.env.rstrip(),
                                       num_levels=500, use_data_augmentation=bool(args.env_aug),
                                       restrict_themes=args.env_monotheme,paint_vel_info = bool(args.env_force_paint_velocity),difficulty =args.difficulty)
    if args.difficulty ==0:
        eval_filepath = 'eval_file_aug_{}.pkl'.format(bool(args.env_aug))
        print_path = os.path.join(base_path, 'sentinel_std_aug_{}'.format(bool(args.env_aug)))
    else:
        eval_filepath = 'eval_file_hard_aug_{}.pkl'.format(bool(args.env_aug))
        print_path = os.path.join(base_path, 'sentinel_std_hard_aug_{}'.format(bool(args.env_aug)))
    printf = lambda x: print_fork(x, print_path)
    if args.overwrite:
        save_dict = {}
        with open(print_path, 'w+') as f:
            f.write('')
    else:
        try:
            with open(os.path.join(base_path, eval_filepath), 'rb') as handle:
                save_dict=pickle.load(handle)
        except FileNotFoundError:
            save_dict={}
            with open(print_path, 'w+') as f:
                f.write('')
    printf('SAVING TO {}'.format(os.path.join(base_path, eval_filepath)))
    #Set up buffer and ac
    if args.old_arch:
        actor_critic = core.RNNAWSEnsembleActorCritic
    else:
        actor_critic = core.RNNAbridgedActorCritic
    # param_filepath = model_param_path
    n_timesteps = args.steps


    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    n_act = env.action_space.n
    num_envs = 1
    if hasattr(env, 'num_envs'):
        num_envs = env.num_envs
    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    if device is not None:
        ac = ac.to(device)

    ac.load_state_dict(torch.load(model_param_paths[0], map_location=device), strict=False)

    n_ensemble = ac.n_ens
    n_rnn = ac.n_layers
    n_feat = ac.n_hid
    o_t= env.reset()
    o_t = o_t[np.newaxis, ...]
    printf("NUMBER OF ENSEMBLES  {:d}".format( n_ensemble))
    printf("MAX EPOCH {:d}".format(np.max(epochs)))
    d_limit=2500
    for epoch, model_path in zip(epochs,model_param_paths):
        if epoch< args.ini_epoch:
            continue
        try:
            epoch_return_dict=save_dict[epoch]
        except KeyError:
            epoch_return_dict={0:{}}


        if 'ep_rets' in epoch_return_dict[0].keys():
            continue
        ac.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        ac.eval()

        # for ens in range(n_ensemble):
        for ens in [0]:
            is_reset = np.zeros(num_envs)
            features, ep_rets, ep_lens, hc_list, val_list = [],[],[],[],[]

            ep_ret, ep_len = np.zeros(num_envs), np.zeros(num_envs)
            # persistent variables
            head_idx = (np.ones([1, num_envs, 1])*ens).astype('int')  # ensemble head index
            h_tm1 = np.zeros([n_rnn, num_envs, n_feat])  # initial head and cell state
            c_tm1 = np.zeros([n_rnn, num_envs, n_feat])  # initial head and cell state
            a_oh_tm1 = np.zeros([1, num_envs, n_act])  # one_hot encoding of previous action
            d_count=0
            for t in range(n_timesteps):
                hc_list.append((h_tm1, c_tm1))
                # step input
                tensor_o_t, tensor_a_oh_tm1, tensor_h_tm1, tensor_c_tm1 = inputs_to_tensor(o_t, a_oh_tm1, h_tm1, c_tm1)
                # compute step
                a_t, v_t, logp_t, logits_t, hc_t, extras_t = ac.step(obs=tensor_o_t, prev_act=tensor_a_oh_tm1,
                                                                     hc=(tensor_h_tm1, tensor_c_tm1),
                                                                     average_ens=True)#,detached_ens=False)
                h_t, c_t = hc_t
                # take step in environment
                o_tp1, r_tp1, d_tp1, _ = env.step(a_t[0])
                o_tp1 = o_tp1[np.newaxis, ...]
                ep_ret += r_tp1
                ep_len += 1  # ones
                # reset important variables for dones
                for idx_env in np.where(d_tp1)[0]:
                    # if is_reset[idx_env]:
                    ep_rets.append(ep_ret[idx_env])
                    ep_lens.append(ep_len[idx_env])
                    d_count += 1
                    # else:
                    #     is_reset[idx_env]=1
                    ep_ret[idx_env], ep_len[idx_env] = 0, 0
                val_list.append(v_t.mean())
                # Update obs, states and previous actions (critical!)
                o_t = o_tp1
                a_oh_tm1 = np.eye(n_act)[a_t]
                h_tm1 = h_t * (1-d_tp1[np.newaxis, :, np.newaxis])
                c_tm1 = c_t * (1-d_tp1[np.newaxis, :, np.newaxis])
                if d_count>=d_limit:
                    break
        old_ep_rets =ep_rets
        epoch_return_dict[ens]={'ep_rets':ep_rets,'ep_lens':ep_lens,'av_vals':np.mean(val_list), 'd_limit':d_limit}
        printf('EPOCH {:d} \nENS {:d} \nAverage Return {:.1f} \nAverage length {:.0f} \nAverage Vals {:.1f}'.format(epoch, ens, np.mean(ep_rets), np.mean(ep_lens), np.mean(val_list)))

        success_idx = np.where(np.array(ep_rets)>0)[0]
        success_len, success_len_m, success_len_s = 0,0,0
        if len(success_idx)>0:
            success_len = np.array(ep_lens)[success_idx]
            success_len_m = np.mean(success_len)
            success_len_s = np.std(success_len)
        printf('Average success length {:.0f} \nSTD success length {:.1f}'.format(
        success_len_m, success_len_s))


        save_dict[epoch] = epoch_return_dict
        with open(os.path.join(base_path, eval_filepath), 'wb') as handle:
            printf('SAVING TO {}'.format(os.path.join(base_path, eval_filepath)))
            pickle.dump(save_dict,handle)





import numpy as np
import torch ,os, pickle, time, warnings, random

from torch.optim import Adam
import gym
from pathlib import Path
import iape.iape.backend.core as core
import iape.iape.backend.detrace as detrace
import iape.iape.backend.vtrace as vtrace
from iape.iape.backend.utils import RnnDeviantBufferLong as DeviantBuffer
from iape.iape.backend.utils import inputs_to_tensor
from iape.iape.backend import make_coin_env, make_atari_env, make_procgen_env, EpochLogger
from iape.iape.backend.logger import setup_logger_kwargs
from torch.distributions.categorical import Categorical

from glob import glob
import torch.nn as nn
warnings.filterwarnings('ignore')


def print_fork(x, path):
    print(x)
    with open(path, 'a+') as f:
        f.write(x+'\n')

def compute_param_norm(parameters):
    total_norm = 0
    for p in parameters:
        try:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        except AttributeError:
            continue
    total_norm = total_norm ** (1. / 2)
    return total_norm


def deviant(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
            steps_per_epoch=3200, n_rollout=100, n_bootstrap=100, epochs=50, gamma=0.99,
            lr_v=1e-3, train_iters=1, logger_kwargs=dict(), save_freq=20, clip_rho_threshold=1.0, pg_mode='one_step', mode='deviant',
            clip_pg_rho_threshold=1.0, clip_beta_threshold=2.0,
            ent_coef=0.01, n_minibatch=8, multi_buff=1, n_ensemble=1,
            param_path='',
            device=None, verbose=True, reload=False, weight_decay=0,
            average_ens=False,
            individual_training=True,
            baseline_ensemble=False,
            zero_hidden=False,

            ):
    """

    """

    # # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    # setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    printf = lambda x: print_fork(x, logger.output_file_std_path)

    # Random seed
    # seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    n_act = env.action_space.n

    num_envs = 1
    if hasattr(env,'num_envs'):
        num_envs=env.num_envs

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    if device is not torch.device('cpu'):
        ac = ac.to(device)

    # Set up optimizers for policy and value function
    optimizer = Adam(ac.parameters(), lr=lr_v, weight_decay=weight_decay)

    # Initialize or reload parameters
    ini_epoch = 0
    if reload:
        model_param_path_candidates = glob(os.path.join(logger.output_dir, 'param_save', 'model_vars_*.pth'))
        epochs_paths,  model_param_paths= [],[]
        for m in model_param_path_candidates:
            try:
                epochs_paths.append(int(m.split('_')[-1].split('.pth')[0]))
                model_param_paths.append(m)
            except ValueError:
                continue
        if len(model_param_path_candidates)>0:
            last_idx = np.argmax(epochs_paths)
            param_path = model_param_paths[last_idx]
            ini_epoch = epochs_paths[last_idx]

    if param_path !='' :
        printf('LOADING FROM'+param_path)
        ac.load_state_dict(torch.load(param_path,map_location=device),strict=False)
        optim_param_path = param_path.replace('model_vars_', 'model_opt_')
        try:
            optimizer.load_state_dict(torch.load(optim_param_path,map_location=device))
        except ValueError:
            pass

    ac.train()
    # # Sync params across processes
    # sync_params(ac)
    n_rnn , n_feat= ac.n_layers, ac.n_hid

    # Count variables
    logger.log('\nNumber of parameters: %d\n' % core.count_vars(ac))
    # Set up model saving
    logger.setup_pytorch_saver(ac)

    fpath =os.path.join(logger.output_dir, 'param_save')
    os.makedirs(fpath, exist_ok=True)
    with open(os.path.join(fpath, 'ac_kwarg.pkl'), 'wb') as handle:
        pickle.dump(ac.ac_kwargs, handle)

    # Set up experience buffer
    # local_steps_per_epoch = int(steps_per_epoch / num_procs())
    local_steps_per_epoch = int(steps_per_epoch)
    buf = DeviantBuffer(size= local_steps_per_epoch, n_env=num_envs,n_t=n_rollout,obs_dim=obs_dim,
                        n_actions=n_act, n_minibatch=n_minibatch, n_rnn=n_rnn, n_feat=n_feat)




    def compute_detrace_loss(data, zero_hidden=False):
        obs, actions, dones, rewards,rollout_logp,prev_oh_acts, hidden_h, hidden_c, breakpoints, lvls = data['obs'], \
                                            data['act'], data['dones'], data['rew'], data['logp'], data['prev_oh_acts'], \
                                            data['prev_h_buf'], data['prev_c_buf'], data['breakpoints'] , data['lvl']

        hidden_h = hidden_h[0].transpose(0,1) #n_layers x batch x hidden_size
        hidden_c = hidden_c[0].transpose(0,1)
        if zero_hidden:
            hidden_h = hidden_h*0
            hidden_c = hidden_c*0

        #Goal, compute bootstrap values and everything else....
        policy, values, _, _, extras = ac(obs, prev_act=prev_oh_acts,hc=(hidden_h, hidden_c), dones=dones, breakpoints=breakpoints)
        policy_logits = policy.logits

        if individual_training:
            lvl_ens_assignment= torch.remainder(lvls,n_ensemble).unsqueeze(-1).long()
            rep = [1 for s in policy_logits.shape[:-1]] + [policy_logits.shape[-1]]
            policy_logits = torch.gather(input=policy_logits, index=lvl_ens_assignment.unsqueeze(-1).repeat(rep), dim=-2).squeeze(-2)
            values = torch.gather(input=values, index=lvl_ens_assignment, dim=-1).squeeze(-1)
        else:
            #add ensemble dims as needed
            rep  =[1 for s in actions.shape ] +[n_ensemble]
            actions = actions.unsqueeze(-1).repeat(rep)
            rewards = rewards.unsqueeze(-1).repeat(rep)
            rollout_logp = rollout_logp.unsqueeze(-1).repeat(rep)
            dones = dones.unsqueeze(-1).repeat(rep)


        bootstrap_value = values[-1,...].detach()
        values = values[:-1,...]
        policy = Categorical(logits=policy_logits[:-1,...])
        dones = dones[:-1]
        actions=actions[:-1]
        rollout_logp=rollout_logp[:-1]
        rewards=rewards[:-1]

        discounts =  (1-dones)*gamma
        policy_logp = policy.log_prob(actions)
        approxkl = (rollout_logp - policy_logp.detach().cpu()).mean().numpy()

        if mode =='deviant':
            detrace_returns = detrace.from_logp(
                rollout_action_log_probs=rollout_logp,
                target_action_log_probs=policy_logp.detach().cpu(),
                dones_tp1=dones.detach().cpu(),
                reward_tp1=rewards,
                values_t=values.detach().cpu(),
                bootstrap_values=bootstrap_value.cpu(),
                n=n_bootstrap,
                gamma_value=gamma,
                pg_mode=pg_mode,
                clip_traj_threshold_u=2,
                clip_traj_threshold_d=0.5,#None,
            )
            pg_advantages = detrace_returns.pg_advantages
            vs = detrace_returns.vs
        elif mode=='impala':
            vtrace_returns = vtrace.from_logp(
                rollout_action_log_probs=rollout_logp,
                target_action_log_probs=policy_logp.detach().cpu(),
                discounts=discounts.detach().cpu(),
                rewards=rewards,
                values=values.detach().cpu(),
                bootstrap_value=bootstrap_value.cpu(),
                clip_rho_threshold=clip_rho_threshold,
                clip_pg_rho_threshold=clip_pg_rho_threshold,
                clip_beta_threshold=clip_beta_threshold,
                actions=actions.detach().cpu(),
                dones=dones.detach().cpu(),
                name='vtrace_from_logits')
            pg_advantages = vtrace_returns.pg_advantages
            vs = vtrace_returns.vs
        else:
            raise NotImplementedError

        pg_advantages=pg_advantages.to(policy_logp.device)
        vs=vs.to(policy_logp.device)

        # Compute loss as a weighted sum of the baseline loss, the policy gradient
        # loss and an entropy regularization term.
        actor_loss= -torch.mean(policy_logp * pg_advantages)
        entropy_loss = -torch.mean(policy.entropy())
        critic_loss = .5 * torch.mean((vs - values)**2)


        #Useful extra info
        extra_info={'average_value_target':vs.mean().cpu().numpy(),
                    'approxkl':approxkl,}

        return actor_loss, entropy_loss, critic_loss, extra_info

    def update():
        batch_t_idx = range(0, buf.size - n_rollout, n_rollout // 2)
        batch_e_idx = range(0, buf.n_env - n_minibatch, n_minibatch)
        batch_pairs = [(t,e) for t in batch_t_idx for e in batch_e_idx]
        random.shuffle(batch_pairs)
        # Train policy with multiple steps of gradient descent
        for train_iter in range(train_iters):
            # for batch_counter in range(0, buf.max_size*buf.n_env//n_rollout//n_minibatch*2):
            #     data=buf.sample_minibatch(n_minibatch=n_minibatch, device=device)
            for batch_t, batch_e in batch_pairs:
                data=buf.get_indexed_minibatch(start_t=batch_t, start_e=batch_e,lenght=n_rollout,n_minibatch=n_minibatch,device=device)
                optimizer.zero_grad()

                # actor_loss, entropy_loss, critic_loss, extra_info = compute_detrace_loss(data,device=device, zero_hidden=zero_hidden)
                actor_loss, entropy_loss, critic_loss, extra_info = compute_detrace_loss(data,zero_hidden=zero_hidden)
                loss =  0.5 * critic_loss + actor_loss + ent_coef * entropy_loss
                loss.backward()

                # pre_clip_norm = compute_param_norm(ac.parameters())
                torch.nn.utils.clip_grad_norm_(ac.parameters(), 0.25)
                optimizer.step()
                # buf.update_kl(kl=extra_info['approxkl'], idx=mb_indicies)
                # Log changes from update
                logger.store(LossPi=actor_loss.item(), LossV=critic_loss.item(), Entropy=-entropy_loss.item(),
                             ValueTargets=extra_info['average_value_target'].item(),
                             # LossGrad=pre_clip_norm,
                             )
                if train_iter==train_iters-1:
                    logger.store(KL=extra_info['approxkl'].mean())



    def dump_log():
        # Log info about epoch
        if verbose:
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('ValueTargets', with_min_and_max=True)
            logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch*num_envs)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            # logger.log_tabular('LossGrad', average_only=True)
            logger.log_tabular('Entropy', average_only=True)
            logger.log_tabular('KL', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.log_tabular('Update Time', time.time()-update_time)
            logger.log_tabular('AC Step Time', step_ac_time)
            logger.log_tabular('Env Step Time', step_env_time)
            logger.dump_tabular()



    # Prepare for interaction with environment
    start_time = time.time()
    o_t, ep_ret, ep_len = env.reset(), np.zeros(num_envs), np.zeros(num_envs)
    o_t=o_t[np.newaxis,...]
    #persistent variables
    head_idx = np.random.randint(n_ensemble, size=[1,num_envs,1]) #ensemble head index
    h_tm1 , c_tm1= np.zeros([n_rnn,num_envs,n_feat]), np.zeros([n_rnn,num_envs,n_feat]) #initial head and cell state
    a_oh_tm1 = np.zeros([1, num_envs, n_act]) #one_hot encoding of previous action
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(ini_epoch,epochs):
        step_ac_time = 0
        step_env_time = 0
        # initialize logger just in case episodes never end
        logger.store(EpRet=0, EpLen=0)
        ac.eval()
        for epoch_time_idx in range(buf.size):
            aux_t=time.time()
            #step input
            tensor_o_t, tensor_a_oh_tm1, tensor_h_tm1, tensor_c_tm1 = inputs_to_tensor(o_t, a_oh_tm1, h_tm1, c_tm1, device=device)
            #compute step
            a_t, v_t, logp_t, logits_t, hc_t, extras_t = ac.step(obs=tensor_o_t, prev_act=tensor_a_oh_tm1, hc=(tensor_h_tm1, tensor_c_tm1), average_ens=average_ens)
            step_ac_time+=time.time()-aux_t

            if (not average_ens):# and (not detached_ens):
                #take action along current head
                a_t = np.take_along_axis(a_t,head_idx, axis=-1)[...,0]
                v_t = np.take_along_axis(v_t,head_idx, axis=-1)[...,0]
                logp_t = np.take_along_axis(logp_t,head_idx, axis=-1)[...,0]
                logits_t = np.take_along_axis(logits_t,head_idx[...,np.newaxis], axis=-2)[...,0,:]
            h_t, c_t = hc_t
            #take step in environment
            aux_t = time.time()
            o_tp1, r_tp1, d_tp1, env_info = env.step(a_t[0])
            o_tp1 = o_tp1[np.newaxis, ...]
            ep_ret += r_tp1
            ep_len += 1#ones
            step_env_time += time.time() - aux_t
            try:
                lvl = np.array([l['level'] for l in env_info])
            except KeyError:
                try:
                    lvl = np.array([l['level_seed'] for l in env_info])
                except KeyError:
                    lvl = None
            # save and log
            buf.store(obs=o_t, act=a_t, rew=r_tp1, logp=logp_t,
                      lvl=lvl, done=d_tp1, prev_oh_act=a_oh_tm1, h=h_tm1, c=c_tm1)
            logger.store(VVals=v_t.mean())

            # reset important variables for dones
            for idx_env in np.where(d_tp1)[0]:
                logger.store(EpRet=ep_ret[idx_env], EpLen=ep_len[idx_env])
                ep_ret[idx_env] , ep_len[idx_env]= 0, 0
                if lvl is not None and not baseline_ensemble:
                    head_idx[0,idx_env] = np.mod(lvl[idx_env], n_ensemble)
                else:
                    head_idx[0, idx_env] = np.random.randint(n_ensemble)
            # Update obs, states and previous actions (critical!)
            o_t = o_tp1
            a_oh_tm1 = np.eye(n_act)[a_t] * (1-d_tp1[np.newaxis,:,np.newaxis])
            h_tm1=h_t* (1-d_tp1[np.newaxis,:,np.newaxis])
            c_tm1=c_t* (1-d_tp1[np.newaxis,:,np.newaxis])


        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            fpath = os.path.join(logger.output_dir, 'param_save')
            fname = os.path.join(fpath, 'model_vars_{:d}.pth'.format(epoch))
            fopt  = os.path.join(fpath, 'model_opt_{:d}.pth'.format(epoch))
            os.makedirs(fpath, exist_ok=True)
            torch.save(ac.state_dict(), fname)
            torch.save(optimizer.state_dict(), fopt)

        # Perform update!
        ac.train()
        update_time = time.time()
        update()
        dump_log()





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # resource usage
    # parser.add_argument('--cpu', type=int, default=1, help='number of cpu cores for traininig')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='aws_debug')
    parser.add_argument('--output_dir', type=str, default='/workdisk/nosnap/procgen/', help='dir to store results')

    #Network configurations
    parser.add_argument('--n_hid', type=int, default=256, help='rnn hidden dim') #256
    parser.add_argument('--n_layers', type=int, default=1, help='rnn hidden number') #256
    parser.add_argument('--n_ens', type=int, default=1, help='number of ensembles')

    # Training configrations
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--wd', type=float, default=2e-5, help='weight decay')
    parser.add_argument('--epochs', type=int, default=2000,  help='number of environment interaction epochs')#int(256000000/256/1024)); 2000;122
    parser.add_argument('--train_iters', type=int, default=1, help='number of iterations through buffer')
    parser.add_argument('--steps', type=int, default=526, help='number of env interactions before SGD') #256*256
    parser.add_argument('--n_minibatch', type=int, default=8, help='minibatch size')
    parser.add_argument('--n_rollout', type=int, default=256, help='number of timesteps per rollout')#256
    parser.add_argument('--n_bootstrap', type=int, default=256, help='time horizon for bootstrapping')#256
    parser.add_argument('--ent_coef', type=float, default=0, help='entropy bonus')#0.01
    parser.add_argument('--lr_v', type=float, default=2e-4)
    parser.add_argument('--pg_mode', type=str, default='impala_pg',  help='policy gradient type {one_step, n_step, simple_n, short_n, impala_pg}') #one_step, n_step, simple_n, short_n, impala_pg
    parser.add_argument('--mode', type=str, default='deviant', help='value estimator {impala, deviant}') #impala, deviant
    parser.add_argument('--multi_buff', type=int, default=1, help='how many old buffers to keep')
    parser.add_argument('--save_freq', type=int, default=20, help='checkpoint frequency')
    parser.add_argument('--zero_hidden', type=int, default=0, help='zero out initial hidden state from buffer during training?')
    parser.add_argument('--small_nw', type=int, default=1, help='use small cnn?')

    #Pretrain or reload options
    parser.add_argument('--model_param_path', type=str, default='', help='specific reload path') #single, split
    parser.add_argument('--reload', type=int, default=1, help='continue from previous save?')


    #Environment options
    parser.add_argument('--env', type=str, default='procgen:procgen-coinrun-v0') #CartPole-v1, Coinrun-standard; special
    # parser.add_argument('--env_aug', action='store_true', default=False, help='use data augmentation (cutout)')
    parser.add_argument('--env_aug', action='store', type=int, default=0, help='use data augmentation (cutout)')
    parser.add_argument('--env_monotheme', action='store_true', default=False, help='use single env theme')
    parser.add_argument('--env_force_paint_velocity', type=int, default=0, help='force paint_velocity_info flag')
    parser.add_argument('--env_difficulty', type=int, default=0, help='force paint_velocity_info flag')
    parser.add_argument('--env_maze', action='store_true', default=False, help='use maze env')
    parser.add_argument('--n_env_levels', type=int, default=500, help='number of training levels')
    parser.add_argument('--env_seed', type=int, default=0, help='seed offset for environment')
    parser.add_argument('--n_env', type=int, default=256, help='simultaneous environment number') #256
    parser.add_argument('--seed', '-s', type=int, default=0)

    #Confusing ensemble options,all of them OFF for IAPE
    parser.add_argument('--joint', action='store_true', default=False, help='Train all ensembles on all experience traces')
    parser.add_argument('--baseline_ensemble', action='store_true', default=False, help='Collect episodes from random ensemble data')
    parser.add_argument('--average_ens', action='store', type=int, default=0, help='use ensemble average as rollout')

    args = parser.parse_args()

    if args.gpu<0 or not torch.cuda.is_available():
        device=torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % (args.gpu))



    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir=args.output_dir)
    ac_kwargs={'n_ens':args.n_ens, 'n_layers' : args.n_layers, 'n_hid': args.n_hid}
    # actor_critic = core.RNNAWSEnsembleActorCritic
    actor_critic = core.RNNAbridgedActorCritic
    if args.small_nw:
        actor_critic = core.RNNAbridgedSimplifiedActorCritic

    if args.env =='Coinrun-standard':
        game_type ='standard'
        if args.env_maze:
            game_type = 'maze'
        env_fn = lambda: make_coin_env(args.n_env, game_type=game_type, set_seed=args.env_seed,
                                       num_levels=args.n_env_levels,use_data_augmentation=bool(args.env_aug), single_theme=args.env_monotheme, )
    elif args.env.startswith('procgen'):
        env_fn = lambda: make_procgen_env(args.n_env, start_level=args.env_seed, game_type=args.env.rstrip(),
                                       num_levels=args.n_env_levels, use_data_augmentation=bool(args.env_aug),
                                       restrict_themes=args.env_monotheme,paint_vel_info = bool(args.env_force_paint_velocity),
                                          difficulty= bool(args.env_difficulty))
        # env_fn = lambda: make_procgen_env(args.n_env, set_seed=-1 + args.env_seed, game_type=args.env.rstrip(),
        #                                num_levels=args.n_env_levels, use_data_augmentation=bool(args.env_aug),
        #                                single_theme=args.env_monotheme, )
    else: #review
        env_fn =lambda: make_atari_env(num_env=args.n_env, env_name=args.env)


    deviant(env_fn,
            actor_critic=actor_critic,
            ac_kwargs=ac_kwargs, gamma=args.gamma,
            seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs, save_freq=args.save_freq,
            train_iters=args.train_iters,
            n_minibatch=args.n_minibatch, n_rollout=args.n_rollout, n_bootstrap=args.n_bootstrap,
            ent_coef=args.ent_coef, lr_v=args.lr_v, device=device, pg_mode=args.pg_mode,
            mode=args.mode,n_ensemble = args.n_ens, multi_buff=args.multi_buff,
            param_path=args.model_param_path, reload = bool(args.reload),
            logger_kwargs=logger_kwargs, weight_decay=args.wd,
            average_ens = bool(args.average_ens), #detached_ens = args.detached_ens,
            individual_training=not args.joint,
            baseline_ensemble=args.baseline_ensemble,
            zero_hidden= bool(args.zero_hidden)
            )
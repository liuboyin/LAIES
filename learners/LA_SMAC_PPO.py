import copy

import torch, os, random, wandb, csv

from components.episode_buffer import EpisodeBatch
from controllers.n_controller import NMAC
from components.action_selectors import categorical_entropy
from utils.rl_utils import build_gae_targets
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np
from utils.th_utils import get_parameters_num
from torch import nn as nn
from numpy import add
import torch.optim as optim
from torch import nn as nn
from torch.nn import functional as F
from components.episode_buffer import EpisodeBatch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from utils.value_norm import ValueNorm

class LA_SMAC_PPO:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_episode = 0
        self.last_target_update_step = 0
        self.critic_training_steps = 0
        np.random.seed(args.seed)
        th.manual_seed(args.seed)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.time_limit = args.env_args['time_limit']
        # a trick to reuse mac
        dummy_args = copy.deepcopy(args)
        dummy_args.n_actions = 1
        self.critic = NMAC(scheme, None, dummy_args)
        self.params = list(mac.parameters()) + list(self.critic.parameters())
        self.train_t = 0
        self.csv_dir = f'./csv_file/{args.env}/reward/{args.label}'
        self.csv_path = f'{self.csv_dir}/seed_{args.seed}_{args.label}.csv'
        if not os.path.exists(self.csv_dir):
            os.makedirs(self.csv_dir)
        self.optimiser = Adam(params=self.params, lr=args.lr)
        self.last_lr = args.lr
        self.start_anneal_time=5e6
        self.init_anneal_time=False
        self.n_agents = args.n_agents
        self.n_enemies = args.n_enemies

        self.eval_model_env = Predict_Network(
            args.state_shape + args.n_actions * args.n_agents, 128, args.enemy_shape * args.n_enemies)
        self.target_model_env = Predict_Network(
            args.state_shape + args.n_actions * args.n_agents, 128, args.enemy_shape * args.n_enemies)


        self.target_model_env.load_state_dict(self.eval_model_env.state_dict())
        self.Target_update = False
        if args.use_cuda:
            self.eval_model_env.cuda()
            self.target_model_env.cuda()

        self.use_value_norm = getattr(self.args, "use_value_norm", False)
        if self.use_value_norm:
            self.value_norm = ValueNorm(1, device=self.args.device)
        
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        state=batch["state"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        old_avail_actions = batch["avail_actions"][:, :-1]

        avail_actions = batch["avail_actions"]
        visible = batch['visible_matrix'][:, :-1]

        # self.train_model(copy.deepcopy(batch))

        b, t, a, _ = batch["obs"][:, :-1].shape
        actions_onehot = (batch["actions_onehot"][:, :-1])
        model_s = th.cat((state,actions_onehot.reshape(b, t, -1)), dim=-1)
        model_opp_s = batch['extrinsic_state'][:, 1:]
        intrinsic_mask = mask.clone()


        loss_model_list = []
        for _ in range(self.args.predict_epoch):
            loss_model = self.eval_model_env.update(
                model_s, model_opp_s, mask)
            loss_model_list.append(loss_model)


        old_probs = batch["probs"][:, :-1]
        old_probs[old_avail_actions == 0] = 1e-10
        old_logprob = th.log(th.gather(old_probs, dim=3, index=actions)).detach()
        mask_agent = mask.unsqueeze(2).repeat(1, 1, self.n_agents, 1)
        
        # targets and advantages
        with th.no_grad():
            old_values = []
            self.critic.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = self.critic.forward(batch, t=t)
                old_values.append(agent_outs)
            old_values = th.stack(old_values, dim=1) 

            if self.use_value_norm:
                value_shape = old_values.shape
                values = self.value_norm.denormalize(old_values.view(-1)).view(value_shape)

            agent_visible = visible[..., :self.n_agents]
            enemies_visible = visible[..., self.n_agents:]
            agent_alive = (agent_visible * (torch.eye(self.n_agents).to(agent_visible.device))).sum(dim=-1)
            agent_alive_mask = torch.bmm(agent_alive.reshape(-1, self.n_agents, 1),
                                         agent_alive.reshape(-1, 1, self.n_agents)).reshape(b, t, self.n_agents,
                                                                                            self.n_agents)
            enemies_visible = enemies_visible.unsqueeze(-1).repeat(1, 1, 1, 1, self.args.enemy_shape)
            enemies_visible = enemies_visible.reshape(b, t, self.n_agents, -1)
            mask_env = mask.clone().roll(dims=-2, shifts=-1)
            mask_env[:, -1, :] = 0
            # Opp_mse_Exp = self.target_model_env.get_log_pi(model_s, model_opp_s) * mask_env
            ac = avail_actions[:, :-1]
            # ac = (1 - actions_onehot) * avail_actions[:, :-1]
            lazy_avoid_intrinsic, enemy_ate = self.target_model_env.get_opp_intrinsic(model_s.clone(), state.clone(),
                                                                                        actions_onehot,
                                                                                        enemies_visible, ac)

            lazy_avoid_intrinsic = lazy_avoid_intrinsic.clamp(max=0.006)
            mean_rewards = rewards.sum() / mask.sum()
            lazy_avoid_intrinsic = lazy_avoid_intrinsic * agent_alive

            old_extrin_s = batch['extrinsic_state'][:, :-1]
            new_extrin_s = batch['extrinsic_state'][:, 1:]
            s_transition = (old_extrin_s - new_extrin_s) ** 2
            CDI = s_transition.sum(dim=-1).clamp(max=0.15).unsqueeze(-1) / 100

            IDI = (lazy_avoid_intrinsic.sum(dim=-1).unsqueeze(-1))
            CDI, IDI = CDI * intrinsic_mask, IDI * intrinsic_mask  #
            # intrinsic_1=intrinsic_1.clamp(max=0.06)
            intrinsic = self.args.beta2 * CDI + self.args.beta1 * IDI
            intrinsic=intrinsic.clamp(max=self.args.itrin_two_clip)
            decay=((self.time_limit-(mask.sum(dim=-2)).unsqueeze(-1))/self.time_limit).repeat(1,intrinsic.shape[-2],1)
            decay=decay.clamp(min=0).unsqueeze(-1)
            if not self.init_anneal_time and mean_rewards > 0.00:
                self.init_anneal_time = True
                self.start_anneal_time = t_env
            if t_env > self.start_anneal_time and self.args.env_args['reward_sparse'] and self.args.anneal_intrin:
                intrinsic = max(1 - 0.5 * (
                        t_env - self.start_anneal_time) / self.args.anneal_speed, 0) * intrinsic
            intrinsic = intrinsic.unsqueeze(-1)
            # if self.args.time_decay:intrinsic=intrinsic*decay
            rewards_new = rewards.unsqueeze(2).repeat(1, 1, self.n_agents, 1) + intrinsic.repeat(1, 1, self.n_agents, 1) # +intrinsic
            rewards_new=20*rewards_new
            advantages, targets = build_gae_targets(rewards_new,
                    mask_agent, values, self.args.gamma, self.args.gae_lambda)


            if self.use_value_norm:
                targets_shape = targets.shape
                targets = targets.reshape(-1)
                self.value_norm.update(targets)
                targets = self.value_norm.normalize(targets).view(targets_shape)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        
        # PPO Loss
        for _ in range(self.args.mini_epochs):
            # Critic
            values = []
            self.critic.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length-1):
                agent_outs = self.critic.forward(batch, t=t)
                values.append(agent_outs)
            values = th.stack(values, dim=1) 

            # value clip
            values_clipped = old_values[:,:-1] + (values - old_values[:,:-1]).clamp(-self.args.eps_clip,
                                                                                self.args.eps_clip)

            # 0-out the targets that came from padded data
            td_error = th.max((values - targets.detach())** 2, (values_clipped - targets.detach())** 2)
            masked_td_error = td_error * mask_agent
            critic_loss = 0.5 * masked_td_error.sum() / mask_agent.sum()

            # Actor
            pi = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length-1):
                agent_outs = self.mac.forward(batch, t=t)
                pi.append(agent_outs)
            pi = th.stack(pi, dim=1)  # Concat over time

            pi[old_avail_actions == 0] = 1e-10
            pi_taken = th.gather(pi, dim=3, index=actions)
            log_pi_taken = th.log(pi_taken)
            
            ratios = th.exp(log_pi_taken - old_logprob)
            surr1 = ratios * advantages
            surr2 = th.clamp(ratios, 1-self.args.eps_clip, 1+self.args.eps_clip) * advantages
            actor_loss = -(th.min(surr1, surr2) * mask_agent).sum() / mask_agent.sum()
            
            # entropy
            entropy_loss = categorical_entropy(pi).mean(-1, keepdim=True) # mean over agents
            entropy_loss[mask == 0] = 0 # fill nan
            entropy_loss = (entropy_loss * mask).sum() / mask.sum()
            loss = actor_loss + self.args.critic_coef * critic_loss - self.args.entropy * entropy_loss / entropy_loss.item()

            # Optimise agents
            self.optimiser.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
            self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num
        if random.random() < 0.01 and self.Target_update:
            self.writereward(self.csv_path, rewards,rewards_new, CDI, IDI, mask, actor_loss, critic_loss, np.array(
                loss_model_list).mean(), intrinsic, t_env)
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            mask_elems = mask_agent.sum().item()
            self.logger.log_stat("advantage_mean", (advantages * mask_agent).sum().item() / mask_elems, t_env)
            self.logger.log_stat("actor_loss", actor_loss.item(), t_env)
            self.logger.log_stat("entropy_loss", entropy_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("lr", self.last_lr, t_env)
            self.logger.log_stat("critic_loss", critic_loss.item(), t_env)
            self.logger.log_stat("target_mean", (targets * mask_agent).sum().item() / mask_elems, t_env)
            self.log_stats_t = t_env


    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()
    def _update_targets(self):
        self.Target_update = True
        self.target_model_env.load_state_dict(self.eval_model_env.state_dict())
        self.logger.console_logger.info("Updated target network")
    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.optimiser.state_dict(), "{}/agent_opt.th".format(path))
    def writereward(self, path, reward, rewards_new,intrinsic_1, intrinsic_2, mask, actor_loss, critic_loss, OPP_eval_Loss, intrinsic, step):
        reward = reward.sum() / mask.sum()
        rewards_new = rewards_new.sum() / mask.shape[0]

        intrinsic = intrinsic.sum() / mask.sum()
        intrinsic_2 = intrinsic_2.sum() / mask.sum()
        intrinsic_1 = intrinsic_1.sum() / mask.sum()
        if self.args.wandb:
            # wandb.log({f"{phase} MSE loss": epoch_mse_loss})
            wandb.log({'step': step, " Actor_Loss": actor_loss,'rewards_sum':rewards_new, "Training reward": reward, 'OPP_eval_Loss': OPP_eval_Loss,
                       'intrinsic_2': intrinsic_2, 'intrinsic_1': intrinsic_1, 'intrinsic': intrinsic,'Critic_loss':critic_loss})
        if os.path.isfile(path):
            with open(path, 'a+') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(
                    [step, reward.item(), reward.item(), actor_loss.item(), critic_loss.item()])
        else:
            with open(path, 'w') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(
                    ['step', 'reward', 'Actor_Loss', 'Critic Loss'])
                csv_write.writerow(
                    [step, reward.item(), actor_loss.item(), critic_loss.item()])

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))



class Predict_Network(nn.Module):

    def __init__(self, num_inputs, hidden_dim, num_outputs, lr=3e-4):
        super(Predict_Network, self).__init__()

        def weights_init_(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                torch.nn.init.constant_(m.bias, 0)
        self.hideen_dim= hidden_dim
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            num_layers=1,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.last_fc = nn.Linear(hidden_dim, num_outputs)

        self.apply(weights_init_)
        self.lr = lr

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input):
        b,t,_=input.shape
        hidden = torch.zeros((1,b,self.hideen_dim)).to(input.device)
        h1 = F.relu(self.linear1(input))
        hrnn ,_= self.rnn(h1,hidden)
        x = self.last_fc(hrnn)
        return x,hrnn
    def counterfactual(self, input,h):
        b,t,n_a,_=input.shape
        input=input.reshape(b*t*n_a,1,-1)
        h=h.reshape(1,b*t*n_a,-1)
        h1 = F.relu(self.linear1(input))
        hrnn ,_= self.rnn(h1,h)
        x = self.last_fc(hrnn)
        return x.reshape(b,t,n_a,-1)
    def get_log_pi(self, own_variable, other_variable):
        predict_variable,_ = self.forward(own_variable)
        log_prob =  F.mse_loss(predict_variable,
                                   other_variable, reduction='none')
        log_prob = torch.sum(log_prob, -1, keepdim=True)
        return log_prob
    def get_opp_intrinsic(self, s_a,s, a,enemies_visible,avail_u=None):
        b,t,n_agents,n_actions=a.shape

        p_s_a,h = self.forward(s_a)

        h_new=torch.zeros_like(h).to(h.device)
        h_new[:,1:]=h[:,:-1]
        full_actions=torch.ones((b,t,n_agents,n_actions,n_actions))*torch.eye(n_actions)
        full_actions=full_actions.type_as(s).to(a.device)
        full_s=s.unsqueeze(-2).repeat(1,1,n_actions,1)
        full_a=a.unsqueeze(-2).repeat(1,1,1,n_actions,1)
        full_h=h_new.unsqueeze(-2).repeat(1,1,n_actions,1)
        intrinsic_2=torch.zeros((b,t,n_agents)).to(a.device)
        Enemy = torch.zeros((b, t, n_agents,p_s_a.shape[-1])).to(a.device)
        if avail_u==None:
            avail_u=torch.ones_like(a).type_as(a)
        for i in range(n_agents):
            ATE_a=(full_a.clone())
            ATE_a[...,i,:,:]=full_actions[...,i,:,:]
            ATE_a=ATE_a.transpose(-2,-3).reshape(b,t,n_actions,-1)
            s_a_noi=torch.cat((full_s,ATE_a),dim=-1)
            p_s_a_noi=self.counterfactual(s_a_noi,full_h)
            p_s_a_noi=p_s_a_noi*(avail_u[...,i,:].unsqueeze(-1))
            p_s_a_mean_noi=p_s_a_noi.sum(dim=-2)/(avail_u[...,i,:].sum(dim=-1).unsqueeze(-1)+1e-6)
            ATE_enemy_i=enemies_visible[...,i,:]*F.mse_loss(p_s_a_mean_noi,p_s_a,reduction='none')
            ATE_i=ATE_enemy_i.sum(dim=-1)
            intrinsic_2[...,i]=ATE_i
            Enemy[...,i,:]=ATE_enemy_i
        return intrinsic_2,Enemy
    def update(self, own_variable, other_variable, mask):
        if mask.sum() > 0:
            predict_variable,_ = self.forward(own_variable)
            loss = F.mse_loss(predict_variable,
                              other_variable, reduction='none')
            loss = loss.sum(dim=-1, keepdim=True)
            loss = (loss * mask).sum() / mask.sum()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
            self.optimizer.step()

            return loss.to('cpu').detach().item()

        return None


"""有限时域训练、独立场景评估与可重载检查点。"""
from dataclasses import asdict
from pathlib import Path
import csv
import json
import random
import time
import platform
import numpy as np
import torch
from .config import Config
from .environment import VPPAdapter
from .algorithms import OfficialPPO, UPSTREAM_COMMIT, to_numpy


def seed_all(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_csv(path, rows):
    if not rows:
        raise ValueError('没有可写入的数据')
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_device(config):
    if config.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('请求了 CUDA，但当前 PyTorch 无法使用 CUDA；请改用 --device cpu')


def train(config, output, env_factory=VPPAdapter):
    config.validate()
    check_device(config)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('输出目录非空，请使用新目录，避免覆盖已有实验')
    seed_all(config.seed, config.threads)
    env = env_factory(config, config.train_csv)
    agent = OfficialPPO(config, env)
    config.save(out / 'config.json')
    metadata = dict(upstream_commit=UPSTREAM_COMMIT, torch=torch.__version__, numpy=np.__version__,
                    python=platform.python_version(), device=config.device,
                    data_sha256=env.data.sha256 if env.data else None,
                    data_source='csv' if env.data else 'synthetic',
                    resolved_official_args=vars(agent.args))
    metadata['dispatch_spec'] = env.spec.record() if hasattr(env, 'spec') else None
    metadata['cyber_spec'] = asdict(env.cyber_spec) if hasattr(env,'cyber_spec') else None
    if config.cyber_mode!='off':
        from .cyber import CONTRACT_VERSION as CYBER_VERSION
        metadata['cyber_contract']=CYBER_VERSION
        metadata['policy_information']='completed_telemetry_and_candidate_command_DT_only'
        metadata['safety_information']='local_current_sensors'
    else: metadata['cyber_contract']=None
    metadata['resource_contract'] = config.resource_model
    metadata['flex_spec'] = asdict(env.flex) if hasattr(env,'flex') else None
    metadata['ev_bundle'] = env.bundle if hasattr(env,'bundle') else None
    metadata['data_provenance'] = env.data.provenance if env.data else None
    from .objectives import CONTRACT_VERSION, aggregate_metrics
    if config.resource_model=='sessions_v1': CONTRACT_VERSION += '+sessions-dr-curtail-v1'
    metadata['objective_contract'] = CONTRACT_VERSION if config.metrics_enabled else None
    if config.network_model == 'ieee33':
        from importlib.metadata import version
        metadata['grid_dependencies'] = {k: version(k) for k in ('pandapower', 'scipy', 'pandas')}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    history = []
    for ep in range(config.episodes):
        obs, share = env.reset(config.seed + ep, ep)
        buffer = agent.buffer()
        buffer.obs[0, 0] = obs
        buffer.share_obs[0, 0] = agent.critic_obs(obs, share)
        total_cost = total_reward = 0.0
        violations = 0
        objective_steps = []
        start = time.perf_counter()
        for step in range(config.horizon):
            agent.trainer.prep_rollout()
            with torch.no_grad():
                values, actions, logprobs, ra, rc = agent.policy.get_actions(
                    buffer.share_obs[step, 0], buffer.obs[step, 0],
                    buffer.rnn_states[step, 0], buffer.rnn_states_critic[step, 0], buffer.masks[step, 0])
            action_np = to_numpy(actions)
            next_obs, next_share, rewards, done, info = env.step(action_np)
            masks = np.full((1, env.num_agents, 1), 0.0 if done else 1.0, np.float32)
            buffer.insert(agent.critic_obs(next_obs, next_share)[None], next_obs[None],
                          to_numpy(ra)[None] * masks[..., None], to_numpy(rc)[None] * masks[..., None],
                          action_np[None], to_numpy(logprobs)[None], to_numpy(values)[None], rewards[None], masks)
            total_cost += info['cost']
            total_reward += info['reward']
            violations += info['constraint_violations']
            if config.metrics_enabled or config.resource_model=='sessions_v1':
                objective_steps.append(info)
        # 每个 rollout 完整包含一个任务，终端掩码阻断跨回合 bootstrap。
        buffer.compute_returns(np.zeros((1, env.num_agents, 1), np.float32), agent.trainer.value_normalizer)
        agent.trainer.prep_training()
        before = torch.cat([p.detach().flatten().cpu() for p in agent.policy.actor.parameters()])
        losses = agent.trainer.train(buffer)
        after = torch.cat([p.detach().flatten().cpu() for p in agent.policy.actor.parameters()])
        metrics = {k: float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v) for k, v in losses.items()}
        if not all(np.isfinite(v) for v in metrics.values()):
            raise RuntimeError('训练出现非有限损失，请检查环境和配置')
        row = dict(episode=ep + 1, env_steps=(ep + 1) * config.horizon, scenario_seed=config.seed + ep,
                   cost=total_cost, reward=total_reward, violations=violations,
                   actor_parameter_delta=float(torch.linalg.vector_norm(after-before)),
                   seconds=time.perf_counter()-start, **metrics)
        history.append(row)
        if config.metrics_enabled:
            row.update(aggregate_metrics(objective_steps))
        if config.resource_model=='sessions_v1':
            from .flex_environment import resource_totals
            row.update(resource_totals(objective_steps))
        if config.cyber_mode!='off':
            from .cyber_environment import cyber_totals
            row.update(cyber_totals(objective_steps))
        write_csv(out / 'training.csv', history)
        checkpoint = dict(format_version=1, upstream_commit=UPSTREAM_COMMIT, config=asdict(config),
                          episode=ep + 1, data_sha256=env.data.sha256 if env.data else None, **agent.state())
        for key in ('resource_contract','flex_spec','ev_bundle','cyber_spec','cyber_contract'):
            checkpoint[key]=metadata[key]
        checkpoint['dispatch_spec'] = metadata['dispatch_spec']
        checkpoint['objective_contract'] = metadata['objective_contract']
        checkpoint['data_provenance'] = metadata['data_provenance']
        checkpoint['train_scenarios'] = env.data.scenario_names if env.data else []
        checkpoint['train_fingerprints'] = env.data.fingerprints if env.data else []
        torch.save(checkpoint, out / 'latest.tmp')
        (out / 'latest.tmp').replace(out / 'latest.pt')
        print(f"episode={ep+1}/{config.episodes} cost={total_cost:.3f} reward={total_reward:.3f} violations={violations}", flush=True)
    return history


def evaluate(checkpoint, output, episodes=3, seed=100000, device='cpu', csv_path=None, env_factory=VPPAdapter, ev_sessions_path=None):
    if episodes < 1 or seed < 0:
        raise ValueError('评估回合数必须为正数，种子不得为负数')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if state.get('format_version') != 1 or state.get('upstream_commit') != UPSTREAM_COMMIT:
        raise ValueError('检查点格式或官方版本不匹配')
    config = Config(**state['config'])
    from .objectives import CONTRACT_VERSION, aggregate_metrics
    if config.resource_model=='sessions_v1': CONTRACT_VERSION += '+sessions-dr-curtail-v1'
    if config.metrics_enabled and state.get('objective_contract') != CONTRACT_VERSION:
        raise ValueError('多目标检查点的指标定义版本不匹配')
    config._dispatch_record = state.get('dispatch_spec')
    config._cyber_record = state.get('cyber_spec')
    if config.cyber_mode!='off':
        from .cyber import CONTRACT_VERSION as CYBER_VERSION
        if state.get('cyber_contract')!=CYBER_VERSION: raise ValueError('C3 检查点版本不匹配')
    config._flex_record = state.get('flex_spec')
    config._ev_bundle = state.get('ev_bundle')
    if ev_sessions_path:
        from .flex_resources import read_bundle
        config._ev_bundle=read_bundle(ev_sessions_path)
    if config.resource_model=='sessions_v1' and state.get('resource_contract')!='sessions_v1':
        raise ValueError('资源模型检查点版本不匹配')
    train_csv = config.train_csv
    config.train_csv = None
    config.device = device
    config.validate()
    check_device(config)
    if csv_path and train_csv and Path(csv_path).resolve() == Path(train_csv).resolve():
        raise ValueError('评估 CSV 必须与训练 CSV 分开')
    train_seeds = range(config.seed, config.seed + config.episodes)
    if not csv_path and any(seed + i in train_seeds for i in range(episodes)):
        raise ValueError('评估场景种子与训练场景重叠')
    seed_all(seed, config.threads)
    env = env_factory(config, csv_path)
    if env.data and env.data.sha256 == state.get("data_sha256"):
        raise ValueError("评估数据与训练数据内容相同，请提供独立测试集")
    if env.data and (set(env.data.scenario_names)&set(state.get('train_scenarios', [])) or
                     set(env.data.fingerprints)&set(state.get('train_fingerprints', []))):
        raise ValueError('评估场景日期或曲线与训练集部分重叠')
    if env.data and episodes > len(env.data.profiles):
        raise ValueError("评估回合数超过 CSV 独立场景数，拒绝循环重复计数")
    agent = OfficialPPO(config, env)
    agent.load(state)
    agent.trainer.prep_rollout()
    rows, trajectories = [], []
    for ep in range(episodes):
        obs, share = env.reset(seed + ep, ep)
        rnn = np.zeros((env.num_agents, 1, config.hidden_size), np.float32)
        masks = np.ones((env.num_agents, 1), np.float32)
        cost = reward = 0.0
        violations = 0
        for step in range(config.horizon):
            with torch.no_grad():
                actions, rnn_new = agent.policy.act(obs, rnn, masks, deterministic=True)
            obs, share, rewards, done, info = env.step(to_numpy(actions))
            rnn = to_numpy(rnn_new)
            trajectories.append(dict(episode=ep+1, scenario_seed=seed+ep, step=step, **info))
            cost += info['cost']
            reward += info['reward']
            violations += info['constraint_violations']
        rows.append(dict(episode=ep+1, scenario_seed=seed+ep, cost=cost, reward=reward, violations=violations))
        if config.network_model == 'ieee33':
            episode_steps = trajectories[-config.horizon:]
            rows[-1].update(objective=cost+sum(r['terminal_penalty'] for r in episode_steps),
                ac_violations=sum(r['ac_violations'] for r in episode_steps),
                ac_failed_steps=sum(not r['ac_converged'] for r in episode_steps),
                ac_cost=sum(r['ac_cost'] for r in episode_steps) if all(r['ac_converged'] for r in episode_steps) else None)
            if config.resource_model=='sessions_v1':
                from .flex_environment import resource_totals
                rows[-1].update(resource_totals(episode_steps))
            if config.cyber_mode!='off':
                from .cyber_environment import cyber_totals
                rows[-1].update(cyber_totals(episode_steps))
            if config.metrics_enabled:
                rows[-1].update(aggregate_metrics(episode_steps))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('评估输出目录非空，请使用新目录')
    write_csv(out / 'episodes.csv', rows)
    write_csv(out / 'trajectory.csv', trajectories)
    summary = dict(algorithm=config.algorithm, episodes=episodes, checkpoint_episode=state['episode'],
                   data_source='csv' if env.data else 'synthetic', data_sha256=env.data.sha256 if env.data else None,
                   mean_cost=float(np.mean([r['cost'] for r in rows])),
                   std_cost=float(np.std([r['cost'] for r in rows], ddof=1)) if episodes > 1 else None,
                   total_violations=sum(r['violations'] for r in rows))
    if config.network_model == 'ieee33':
        summary.update(total_ac_violations=sum(r['ac_violations'] for r in rows),
            ac_failed_steps=sum(r['ac_failed_steps'] for r in rows),
            mean_objective=float(np.mean([r['objective'] for r in rows])))
    if config.cyber_mode!='off':
        summary.update(cyber_contract=state['cyber_contract'],cyber_spec=state['cyber_spec'],
            mean_aoi_seconds=float(np.mean([r['mean_aoi_seconds'] for r in rows])),
            total_dt_updates=sum(r['dt_updates'] for r in rows),safety_information='local_current_sensors')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if config.metrics_enabled:
        from .objectives import OBJECTIVE_NAMES
        summary['objective_contract'] = CONTRACT_VERSION
        summary['mean_objective_vector'] = [float(np.mean([r[k] for r in rows])) for k in OBJECTIVE_NAMES]
        summary['carbon_source'] = 'csv_column' if csv_path else 'explicit_synthetic_assumption'
        (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return rows

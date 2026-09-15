"""统一命令入口。"""
import argparse
from .config import Config


def main():
    parser = argparse.ArgumentParser(description='VPP 官方 MAPPO 训练与评估')
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--config', default='configs/mappo_smoke.json')
    t.add_argument('--output', required=True)
    t.add_argument('--device')
    t.add_argument('--algorithm', choices=['mappo', 'ippo', 'weighted_mappo'])
    t.add_argument('--episodes', type=int)
    t.add_argument('--seed', type=int)
    t.add_argument('--csv', help='训练场景 CSV，覆盖配置中的 train_csv')
    e = sub.add_parser('evaluate')
    e.add_argument('--checkpoint', required=True)
    e.add_argument('--output', required=True)
    e.add_argument('--episodes', type=int, default=3)
    e.add_argument('--seed', type=int, default=100000)
    e.add_argument('--device', default='cpu')
    e.add_argument('--csv')
    e.add_argument('--ev-sessions',help='独立评估 EV 会话 JSON')
    sub.add_parser('doctor')
    args = parser.parse_args()
    if args.command == 'doctor':
        import torch
        import numpy
        import gymnasium
        from .algorithms import UPSTREAM_COMMIT
        print(f'PyTorch={torch.__version__}, NumPy={numpy.__version__}, Gymnasium={gymnasium.__version__}')
        print(f'CUDA 可用={torch.cuda.is_available()}, 官方 MAPPO={UPSTREAM_COMMIT}')
    elif args.command == 'train':
        from .runner import train
        cfg = Config.load(args.config)
        if args.csv:
            cfg.train_csv = args.csv
        for key in ('device', 'algorithm', 'episodes', 'seed'):
            if getattr(args, key) is not None:
                setattr(cfg, key, getattr(args, key))
        train(cfg, args.output)
    else:
        from .runner import evaluate
        evaluate(args.checkpoint, args.output, args.episodes, args.seed, args.device, args.csv, ev_sessions_path=args.ev_sessions)


if __name__ == '__main__':
    main()

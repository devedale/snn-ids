#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SNN-IDS Unified Benchmark Runner
"""
import argparse
import sys
import os

sys.path.append(os.path.abspath('src'))
from snn_ids import workflows

def main():
    parser = argparse.ArgumentParser(
        description="SNN-IDS Unified Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.epilog = '''\
Examples of use:
# Run a standard centralized smoke test
   python3 benchmark.py centralized --smoke-test
# Run the new MLP deep analysis
   python3 benchmark.py mlp-analysis --sample-size 10000
'''
    subparsers = parser.add_subparsers(dest='mode', required=True, help="Select the execution mode.")

    p_centralized = subparsers.add_parser('centralized', help='Standard centralized training.')
    p_centralized.add_argument('--smoke-test', action='store_true')
    p_centralized.add_argument('--full', action='store_true')
    p_centralized.add_argument('--sample-size', type=int)
    p_centralized.add_argument('--models', nargs='+', choices=['dense', 'gru', 'lstm'])

    p_mlp = subparsers.add_parser('mlp-analysis', help='Run the deep MLP analysis.')
    p_mlp.add_argument('--sample-size', type=int)

    p_progressive = subparsers.add_parser('progressive', help='Run benchmark with progressive data samples.')
    p_progressive.add_argument('--sample-size', type=int)
    p_progressive.add_argument('--models', nargs='+', choices=['dense', 'gru', 'lstm'])

    # Placeholders for other modes
    subparsers.add_parser('best-config', help='(Not yet implemented)')
    subparsers.add_parser('federated', help='(Not yet implemented)')
    
    args = parser.parse_args()

    if args.mode == 'centralized':
        workflows.run_centralized(
            smoke_test=args.smoke_test,
            full_benchmark=args.full,
            sample_size=args.sample_size,
            models_to_test=args.models
        )
    elif args.mode == 'mlp-analysis':
        workflows.run_mlp_deep_analysis(
            sample_size=args.sample_size
        )
    elif args.mode == 'progressive':
        workflows.run_progressive(
            sample_size=args.sample_size,
            models_to_test=args.models
        )
    else:
        print(f"Mode '{args.mode}' is not yet implemented.")

if __name__ == '__main__':
    main()

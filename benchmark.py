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
# Run the progressive optimization benchmark
   python3 benchmark.py progressive --models gru
# Run the best-config fine-tuning workflow
   python3 benchmark.py best-config --sample-size 20000
# Run a federated experiment with Homomorphic Encryption
   python3 benchmark.py federated --he
# Run a federated sweep to compare HE vs non-HE
   python3 benchmark.py federated --sweep
'''
    subparsers = parser.add_subparsers(dest='mode', required=True, help="Select the execution mode.")

    p_centralized = subparsers.add_parser('centralized', help='Standard centralized training.')
    p_centralized.add_argument('--smoke-test', action='store_true')
    p_centralized.add_argument('--full', action='store_true')
    p_centralized.add_argument('--sample-size', type=int)
    p_centralized.add_argument('--models', nargs='+', choices=['dense', 'gru', 'lstm'])

    p_progressive = subparsers.add_parser('progressive', help='Run the progressive optimization benchmark.')
    p_progressive.add_argument('--sample-size', type=int)
    p_progressive.add_argument('--models', nargs='+', choices=['dense', 'gru', 'lstm'])

    p_best_config = subparsers.add_parser('best-config', help='Run the fine-tuning workflow.')
    p_best_config.add_argument('--sample-size', type=int)

    p_federated = subparsers.add_parser('federated', help='Run Federated Learning experiments.')
    p_federated.add_argument('--he', action='store_true')
    p_federated.add_argument('--dp', action='store_true')
    p_federated.add_argument('--sweep', action='store_true')
    p_federated.add_argument('--sample-size', type=int)
    
    args = parser.parse_args()

    if args.mode == 'centralized':
        workflows.run_centralized(
            smoke_test=args.smoke_test,
            full_benchmark=args.full,
            sample_size=args.sample_size,
            models_to_test=args.models
        )
    elif args.mode == 'progressive':
        workflows.run_progressive(
            sample_size=args.sample_size,
            models_to_test=args.models
        )
    elif args.mode == 'best-config':
        workflows.run_best_config_tuning(
            sample_size=args.sample_size
        )
    elif args.mode == 'federated':
        workflows.run_federated(
            use_he=args.he,
            use_dp=args.dp,
            sweep=args.sweep,
            sample_size=args.sample_size
        )

if __name__ == '__main__':
    main()

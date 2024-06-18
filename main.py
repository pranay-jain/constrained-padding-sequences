import argparse

from pfs import run_pfs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for PFS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-d",
        "--dataset",
        help="Dataset",
        choices=['wikipedia', 'linode_from_index', 'autocomplete', 'synthetic'])
    parser.add_argument("-c", "--pad_factor", help="Maximum allowed padding factor for PFS or PWOD pad scheme.", default=1.25)
    parser.add_argument("-k", "--stride", help="", default=2)
    parser.add_argument("--prefix_closed", help="increase verbosity", default=True)
    
    args = parser.parse_args()
    config = vars(args)
    print(config['prefix_closed'])

    run_pfs(
        config['dataset'],
        float(config['pad_factor']), 
        int(config['stride']),
        config['prefix_closed'])

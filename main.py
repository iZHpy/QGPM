"""Run Base, QGPM or Oracle on NoLiMa or RULER.

    python3 main.py <NoLiMa|RULER> <Base|QGPM|Oracle> --model_name_or_path <model> --context_length <length> [flags]
"""
from config import parse_args
from runners import run_nolima, run_ruler


def main() -> None:
    args = parse_args()
    if args.dataset == "NoLiMa":
        run_nolima(args)
    else:
        run_ruler(args)


if __name__ == "__main__":
    main()

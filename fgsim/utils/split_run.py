#!/dev/shm/mscham/fgsim/bin/python
import subprocess
from copy import deepcopy
from datetime import datetime

from rich.console import Console

from fgsim.config import parse_arg_conf
from fgsim.utils.cli import parser

console = Console()


def run_args(args):
    if args.remote:
        if args.command in ["setup", "overwrite", "dump"]:
            raise Exception("Command not allowed remote")
        if args.hash is None:
            args.hash = parse_arg_conf(args)[0].hash

    cmd = []
    if args.hash is not None:
        cmd.append(f"--hash")
        cmd.append(args.hash)
    elif args.tag is not None:
        cmd.append(f"--tag")
        cmd.append(args.tag)
    else:
        raise Exception
    for option in ["debug", "ray"]:
        if vars(args)[option]:
            cmd.append(f"--{option}")
    cmd.append(args.command)

    if args.remote:
        cmd = [
            "sbatch",
            "--partition=allgpu",
            "--time=48:00:00",
            "--nodes=1",
            "--constraint=P100|V100|A100",
            f"--output=wd/slurm-train-{args.hash}-%j.out",
            f"--job-name=train.{args.hash}",
            "run.sh",
        ] + cmd
    else:
        cmd = ["python", "-m", "fgsim"] + cmd
    cmd_str = " ".join(cmd)
    # cmd = ["python", "/home/mscham/test.py"]

    color_start = "color(11)"
    color_end = "color(190)"
    color_fail = "color(1)"

    style = f"bold {color_start}"
    console.rule(
        f"[{style}]"
        f" {args.command} {args.hash if args.hash is not None else args.tag}",
        style=style,
    )

    console.log(f"[bold color(0) on {color_start}]  Start [/] :rocket: {cmd_str} ")
    start = datetime.now()
    process = subprocess.run(cmd)
    end = datetime.now()
    dur_str = f":four_o’clock: {(end-start).seconds}sec"
    if process.returncode == 0:
        style = f"bold {color_end}"
        console.log(f"[bold color(0) on {color_end}] End [/] {cmd_str} {dur_str}")

    else:
        style = f"bold {color_fail}"
        console.log(
            f"[bold color(0) on {color_fail}] Fail [/] :x: {process.returncode}"
            f" {cmd_str} {dur_str}"
        )
    console.rule(
        f"[{style}]"
        f" {args.command} {args.hash if args.hash is not None else args.tag}",
        style=style,
    )
    if process.returncode != 0:
        exit(process.returncode)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.hash is None:
        if "," not in args.tag:
            run_args(args)
        else:
            for tag in args.tag.split(","):
                eargs = deepcopy(args)
                eargs.tag = tag
                run_args(eargs)
    else:
        if "," not in args.hash:
            run_args(args)
        else:
            for hash in args.hash.split(","):
                eargs = deepcopy(args)
                eargs.hash = hash
                run_args(eargs)

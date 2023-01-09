#!/dev/shm/mscham/fgsim/bin/python
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from rich.console import Console

from fgsim.config import parse_arg_conf
from fgsim.utils.cli import parser

console = Console()


def run_args(args):
    if args.remote:
        if args.command in ["setup", "overwrite", "dump"]:
            raise Exception("Command not allowed remote")
        if args.hash is None or args.tag is None:
            conf = parse_arg_conf(args)[0]
            if args.hash is not None:
                assert args.hash == conf.hash
            if args.tag is not None:
                assert args.tag == conf.tag
            args.hash = conf.hash
            args.tag = conf.tag

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
        jobstr = f"{args.tag}.{args.hash}.{args.command}"
        slurm_log_path = Path(f"~/slurm/{jobstr}-%j.out").expanduser()
        cmd = [
            "sbatch",
            "--partition=maxgpu",
            "--time=48:00:00",
            "--nodes=1",
            "--constraint=V100|A100",
            f"--output={slurm_log_path}",
            f"--job-name={jobstr}",
            "scripts/run.sh",
        ] + cmd
    else:
        cmd = ["python", "-m", "fgsim"] + cmd
    cmd_str = "<[underline]" + " ".join(cmd) + "[/]>"
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

    start = datetime.now()
    console.print(
        "[bold color(0) on"
        f" {color_start}]Start[/] [{start.strftime('%H:%M')}] :rocket: {cmd_str} "
    )

    process = subprocess.run(cmd)
    end = datetime.now()
    dur_str = f":four_o’clock: {(end-start).seconds}sec"
    if process.returncode == 0:
        style = f"bold {color_end}"
        console.print(
            f"[bold color(0) on {color_end}]End  [/]"
            f" [{end.strftime('%H:%M')}] :white_check_mark:{cmd_str} {dur_str}"
        )

    else:
        style = f"bold {color_fail}"
        console.print(
            f"[bold color(0) on {color_fail}]Fail [/] [{end.strftime('%H:%M')}] :x:"
            f" {process.returncode} {cmd_str} {dur_str}"
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

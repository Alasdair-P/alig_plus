import os
import time
import yaml

def create_jobs():
    jobs = []

    template = "python main.py --no_visdom --no_tqdm --no_tb --jade "
    rn_opts = " --width 1  "

    with open("reproduce/hparams/aalig_cifar10_da.yaml", "r") as f:
        hparams = yaml.safe_load(f)

    for hparam in hparams:
        command = template + " ".join("--{} {}".format(key, value) for key, value in hparam.items())
        if hparam['model'] == "wrn":
            command += wrn_opts
        elif hparam['model'] == "rn":
            command += rn_opts
        elif hparam['model'] == "dn":
            command += dn_opts
        elif hparam['model'] == "gcn":
            command += gcn_opts
        else:
            raise ValueError("Model {} not recognized".format(hparam["model"]))
        jobs.append(command)

    template = "python main.py --no_visdom --no_tqdm --no_tb --jade --no_data_augmentation "
    rn_opts = " --width 1  "

    with open("reproduce/hparams/aalig_cifar10_nda.yaml", "r") as f:
        hparams = yaml.safe_load(f)

    for hparam in hparams:
        command = template + " ".join("--{} {}".format(key, value) for key, value in hparam.items())
        if hparam['model'] == "wrn":
            command += wrn_opts
        elif hparam['model'] == "rn":
            command += rn_opts
        elif hparam['model'] == "dn":
            command += dn_opts
        elif hparam['model'] == "gcn":
            command += gcn_opts
        else:
            raise ValueError("Model {} not recognized".format(hparam["model"]))
        jobs.append(command)
    return jobs


def run_command(command, noprint=True):
    command = " ".join(command.split())
    print(command)
    os.system(command)

def launch(jobs, interval):
    for i, job in enumerate(jobs):
        print("\nJob {} out of {}".format(i + 1, len(jobs)))
        run_command(job)
        time.sleep(interval)

if __name__ == "__main__":
    jobs = create_jobs()
    for job in jobs:
        print(job)
    print("Total of {} jobs to launch".format(len(jobs)))
    launch(jobs, 5)




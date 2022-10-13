import os
import time
import yaml

def create_jobs():

    jobs = []

    run = "_3 "
    GPU = "CUDA_VISIBLE_DEVICES=0, taskset -c 0-2 "

    template = "python main.py --no_tb --no_visdom --batch_size 128 --epochs 200 --depth 10 "
    no_data_aug = " --no_data_augmentation --tag nda"
    data_aug = " --tag da"
    data_set = " --dataset svhn "

    t_nda = GPU + template + data_set + no_data_aug + run

    list_jobs(t_nda, jobs)

    template = "python main.py --no_tb --no_visdom --batch_size 128 --epochs 200 --depth 10 "
    no_data_aug = " --no_data_augmentation --tag nda"
    data_aug = " --tag da"
    data_set = " --dataset cifar10 "

    t_nda = GPU + template + data_set + no_data_aug + run
    t_da = GPU + template + data_set + data_aug + run

    list_jobs(t_nda, jobs)
    list_jobs(t_da,  jobs)

    template = "python main.py --no_tb --no_visdom --batch_size 128 --epochs 200 --depth 10 "
    no_data_aug = " --no_data_augmentation --tag nda"
    data_aug = " --tag da"
    data_set = " --dataset cifar100 "

    t_nda = GPU + template + data_set + no_data_aug + run
    t_da = GPU + template + data_set + data_aug + run

    list_jobs(t_nda, jobs)
    list_jobs(t_da,  jobs)

    template = "python main.py --no_tb --no_visdom --batch_size 128 --epochs 200 --depth 10 "
    no_data_aug = " --no_data_augmentation --tag nda"
    data_aug = " --tag da"
    data_set = "--dataset tiny_imagenet "

    t_nda = GPU + template + data_set + no_data_aug + run
    t_da = GPU + template + data_set + data_aug + run

    list_jobs(t_nda, jobs)
    list_jobs(t_da,  jobs)

    return jobs


def list_jobs(template, jobs):

    wrn_opts = " --depth 40 --width 4 --epochs 200 "
    rn_opts = " --width 1 "
    dn_opts = " --depth 40 --growth 40 --epochs 300 "
    mlp_opts = " "

    #with open("reproduce/opts.yaml", "r") as f:
    with open("reproduce/galig_plus.yaml", "r") as f:
        hparams = yaml.safe_load(f)
    for hparam in hparams:
        command = template + " ".join("--{} {}".format(key, value) for key, value in hparam.items())
        if hparam['model'] == "wrn":
            command += wrn_opts
        elif hparam['model'] == "rn":
            command += rn_opts
        elif hparam['model'] == "mlp":
            command += mlp_opts
        elif hparam['model'] == "dn":
            command += dn_opts
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


import yaml
from scheduling import launch

def create_jobs():

    jobs = [

            """python main.py --dataset imagenet --model resnet18 --opt coin
               --batch_size 256 --epochs 90  """,

            """python main.py --dataset imagenet --model resnet18 --opt adam --eta 0.01
               --batch_size 256 --epochs 90 """,

            """python main.py --dataset imagenet --model resnet18 --opt sgd --eta 0.1
               --batch_size 256 --epochs 90 --l2_reg 1e-4 --T 30 60 --momentum 0.9 """,

            """python main.py --dataset imagenet --model resnet18 --opt pal --eta 1.0
               --batch_size 256 --epochs 90 --l2_reg 1e-5 """,

            """python main.py --dataset imagenet --model resnet18 --opt sgd_armijo
               --batch_size 256 --epochs 90 --l2_reg 1e-5 """,

            """python main.py --dataset imagenet --model resnet18 --opt sgd_goldstein
               --batch_size 256 --epochs 90 """,

            """python main.py --dataset imagenet --model resnet18 --opt sgd_polyak
               --batch_size 256 --epochs 90 """,

            """python main.py --dataset imagenet --model resnet18 --opt alig+ --eta 1.0
               --batch_size 256 --epochs 90 --weight_decay 0.0001 """,

            """python main.py --dataset imagenet --model resnet18 --opt alig --eta 0.1
               --batch_size 256 --epochs 90 --weight_decay 0.0001 """,

            """python main.py --dataset imagenet --model resnet18 --opt sps --eta 0.1
               --batch_size 256 --epochs 90 --l2_reg 0.0 """,

           ]



    return jobs


if __name__ == "__main__":
    jobs = create_jobs()
    launch(jobs)

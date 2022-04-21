
import sys
from train import learn_mle
from em_train import run_em_5times

def main():
    uai_file = sys.argv[1]
    task_id = int(sys.argv[2])
    train_data = sys.argv[3]
    test_data = sys.argv[4]

    if task_id == 1:
        dif = learn_mle(uai_file, train_data, test_data)
        print(f"log likelihood difference = {round(dif, 4)}. ")
    else:
        avg_dif, std_dif = run_em_5times(uai_file, train_data, test_data)
        print(f"The average log likelihood difference = {round(avg_dif, 4)}. ")
        print(f"Its standard deviation is {round(std_dif, 4)}.")

if __name__ == "__main__":
    main()
import random
import subprocess
import pandas as pd

if __name__ == '__main__':
    embedding_size = 768  # base embedding_size
    number_values = 100  # 50 values for training, 50 values to set threshold

    randoms_list = []
    half_randoms_list = []
    dataFrames_to_save = pd.DataFrame()
    i = 0

    while i < number_values:
        r = random.randint(0, embedding_size - 1)
        if r not in randoms_list:
            randoms_list.append(r)
            half_randoms_list.append(r)
            i += 1
            if i % (number_values // 2) == 0:
                df = pd.DataFrame(half_randoms_list)
                dataFrames_to_save = pd.concat([dataFrames_to_save, df], axis=1)
                half_randoms_list = []

    # remove Excel if exists
    random_values_excel_path = '/home/gilnetanel/Desktop/random_values/random_values.xlsx'
    cmd = 'rm ' + random_values_excel_path
    subprocess.run(cmd, shell=True)

    dataFrames_to_save.to_excel(random_values_excel_path, index=None, header=["training", "threshold"])
    print("Saved random_values to Excel")

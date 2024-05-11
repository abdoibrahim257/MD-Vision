from utils import *

def main():
    print('AMERICA YA, HALOOO')
    caption_gen = initialize()
    caption_gen.caption_generation()
    

def initialize():
    n = 3 #ngrams default = 3
    parents = 5 #parents default = 5
    hops = 1
    nlogP = n #n used in loss function
    optimiser = 4 #which cost fucntion to use
    max_iters = 150 #max iterations default = 150
    configs = [n, parents, hops, nlogP, optimiser, max_iters]
    start_end_tokens = ['<t>', '</t>']
    out_folder = 'output_captions'
    input_csv_file_path = './input.csv' #CNN output will be here

    caption_gen = kengic(configs, out_folder, input_csv_file_path)
    return caption_gen

main()
    
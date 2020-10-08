import os
import sys
import hashlib
import json
from PIL import Image
import argparse 
import time
import base64
from collections import defaultdict

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir',
                        help = 'Directory in which the data is located')
    parser.add_argument('output_dir',
                        help = 'Directory in which the output files will be stored')

    arguments = parser.parse_args()

    print("Data Directory:      {}".format(arguments.data_dir))
    print("Output Directory:    {}\n".format(arguments.output_dir))

    return arguments

def check_arguments(args):
    if not os.path.exists(args.data_dir):
        return False
    if not os.path.exists(args.output_dir):
        return False
    return True


def find_invalid(path):
    print("Finding corrupted or non-image files...")
    invalid_files = {}

    for filename in os.listdir(path):
        
        if filename.endswith('.png'):
            try:
                img = Image.open(os.path.join(path, filename))
                img.verify()
            except (IOError, SyntaxError) as e:
                invalid_files.update( {filename : "Corrupted .png file"} )
        else:
            invalid_files.update( {filename : "Not a .png file"} )
    print("Done.")
    return invalid_files

def find_mislabeled(path):
    print("Finding mislabeled files...")
    
    mislabeled_labels_csv = {}
    mislabeled_list = []
    
    for filename in os.listdir(path):
        
        file_label = filename.split('_')[0]
        if file_label not in labels and filename.endswith('.png'):
            mislabeled_list.append(filename)
            if file_label not in mislabeled_labels_csv.keys():
                mislabeled_labels_csv.update({file_label: 1})
            else:
                mislabeled_labels_csv[file_label]+=1
        else:
            pass
    print("Done.")
    return mislabeled_labels_csv, mislabeled_list
    
def encode_img(img_path):
    with open(img_path, 'rb') as file:
        encoding = base64.b64encode(file.read())
    return encoding


def find_dups(directory):
    print("Finding duplicated images...")

    file_list = list(filter(lambda file_filter: file_filter.endswith('.png'), os.listdir(directory))) #list of all the image file in the directory

    seen_hashes = set() #data structure that will keep track of hashes already seen

    root_files = {} #data structure that will map the hash to its respective files

    filecount = 0 

    label_counts = {} #dict that will map the number of duplicates per label

    dups = defaultdict(list) #dict-like data structure that will map the filenames and its duplicates

    for file in file_list: 
        filecount += 1

        if filecount % 10000 == 0:
            print ('{} files processed.'.format(filecount))


        encoded_img = encode_img(os.path.join(directory, file))

        if encoded_img not in seen_hashes:
            seen_hashes.add(encoded_img)
            root_files[encoded_img] = file
        else:
            root_file = root_files[encoded_img]
            dups[root_file].append(file)

            
            label = file.split('_')[0] 

            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 0
    
    print("Done")
    return dups, label_counts


def write_to_csv(dictionary, output_filename, output_dir):
    out_file = os.path.join(output_dir, output_filename)
    with open(out_file, 'w') as f:
        for key in dictionary.keys():
            f.write("%s,%s\n"%(key, dictionary[key]))
    print("{} created".format(output_filename))

def write_to_txt(list_ds, output_filename, output_dir):
    out_file = os.path.join(output_dir, output_filename)
    with open(out_file, 'w') as f:
        for element in list_ds:
            f.write("%s\n"%(element))
    print("{} created".format(output_filename))

def write_to_json(dictionary, output_filename, output_dir):
    out_file = os.path.join(output_dir, output_filename)
    with open(out_file, 'w') as jsonfile:
        json.dump(dictionary, jsonfile, indent = 4)
    print("{} created".format(output_filename))

def count_and_remove(data_path, output_path, dups_dictionary):
    print("Counting and removing duplicated, corrupted and non-image files...")
    dict_keys = ["OriginalDataSetCount", "InvalidCount", "MislabeledCount", "DuplicatedCount", "NewDataSetCount", "LabelData"]
    json_dict = {keys:0 for keys in dict_keys}

    for filename in os.listdir(data_path):
        json_dict["OriginalDataSetCount"] += 1
        if filename.endswith('.png'):
            try:
                img = Image.open(os.path.join(data_path, filename))
                img.verify()
            except (IOError, SyntaxError) as e:
                json_dict["InvalidCount"] += 1
        elif not filename.endswith('.png'):
            json_dict["InvalidCount"] += 1
    
    with open(os.path.join(output_path, 'Mislabeled_files_count.csv'), 'r') as f:
        for line in f:
            splitline = line.split(",")
            json_dict["MislabeledCount"] += int(splitline[-1])

    with open(os.path.join(output_path, "Duplicates_count.csv"), 'r') as f2:
        for line2 in f2:
            splitline2 = line2.split(",")
            json_dict["DuplicatedCount"] += int(splitline2[-1])

    json_dict["NewDataSetCount"] = json_dict["OriginalDataSetCount"] - json_dict["InvalidCount"]  - json_dict["MislabeledCount"] - (json_dict["DuplicatedCount"]  - json_dict["MislabeledCount"])

    # remove invalid files
    for filename in os.listdir(data_path):
        if filename.endswith('.png'):
            try:
                img = Image.open(os.path.join(data_path, filename))
                img.verify()
            except (IOError, SyntaxError) as e:
                os.remove(os.path.join(data_path, filename))               
        elif not filename.endswith('.png'):
            os.remove(os.path.join(data_path, filename))

    #remove duplicates
    for (k, v) in dups_dictionary.items():
        if k.split('\\')[-1].split('_')[0] in labels:
            for element in v:
                os.remove(element)
        else:
            if len(v) > 1:
                os.remove(v[-1])
            os.remove(k)

    # remove mislabeled data
    with open(os.path.join(output_path, 'Mislabeled_names.txt'), 'r') as textfile:
        for line in textfile:
            for filename in os.listdir(data_path):
                if filename == line:
                    os.remove(os.path.join(data_path, filename))
                else:
                    pass

    label_dict = {label : {"Count" : 0, "Fraction" : 0} for label in labels}

    #first, calculate the count of each label
    for filename in os.listdir(data_path):
        if filename.split('_')[0] in label_dict.keys():
            label_dict[filename.split('_')[0]]["Count"] += 1

    #now, calculate the fraction
    for key in label_dict.keys():
        label_dict[key]["Fraction"] = round(float(label_dict[key]["Count"] / json_dict["NewDataSetCount"]), 2)

    json_dict["LabelData"] = label_dict

    write_to_json(json_dict,'Report.json', args.output_dir)
    
    print("Done.")


if __name__ == '__main__':
    
    args = arg_parser()

    if not check_arguments(args):
        print("Invalid argument(s)")
        sys.exit(-1)
    
    print('Beginning process...\n')

    start_time = time.time()

    invalid_csv = find_invalid(args.data_dir)
    print("Finding corrupted and non-image files took %s seconds\n"%(time.time() - start_time))

    mislabeled_time = time.time()

    mislabeled_csv, mislabeled_text = find_mislabeled(args.data_dir)
    print("Finding mislabeled files took %s seconds\n"%(time.time() - mislabeled_time))

    dups_time = time.time()
    
    duplicates_dict, duplicates_csv = find_dups(args.data_dir)
    print("Finding duplicated files took %s seconds\n"%(time.time() - dups_time))

    writing_files_time = time.time()

    write_to_csv(invalid_csv, 'Invalid_file.csv', args.output_dir)
    write_to_csv(mislabeled_csv, 'Mislabeled_files_count.csv', args.output_dir)
    write_to_csv(duplicates_csv, 'Duplicates_count.csv', args.output_dir)
    write_to_txt(mislabeled_text, 'Mislabeled_names.txt', args.output_dir)
    write_to_json(duplicates_dict, 'Duplicate_files.json', args.output_dir)
    print("File writing took %s seconds\n"%(time.time() - writing_files_time))

    count_and_remove_time = time.time()

    count_and_remove(args.data_dir, args.output_dir, duplicates_dict)
    print("Counting and removing took %s seconds\n"%(time.time() - count_and_remove_time))

    print('Data processing done. Script took %s seconds to run'%(time.time() - start_time))    
    
    
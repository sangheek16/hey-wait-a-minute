"""
Just some util functions that I use quite often...
"""

import csv
import json
import re
import unicodedata


def read_jsonl(path):
    with open(path) as f:
        data = f.readlines()
    data = [json.loads(line) for line in data]
    return data


def read_csv_dict(path):
    data = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for line in reader:
            data.append(line)
    return data


def roundup(x):
    return x if x % 1000 == 0 else x + 1000 - x % 1000


def read_file(path):
    """TODO: make read all"""
    return [
        unicodedata.normalize("NFKD", i.strip())
        for i in open(path, encoding="utf-8").readlines()
        if i.strip() != ""
    ]


def write_file(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for line in data:
            f.write(line + "\n")


def belongingness(tup1, tup2):
    """is tup1 contained in tup2?"""
    assert tup1[0] <= tup1[1] and tup2[0] <= tup2[1]

    if tup2[0] <= tup1[0] and tup2[1] >= tup1[1]:
        return True
    else:
        return False


def write_dict_list_to_csv(dict_list, csv_file):
    fieldnames = dict_list[0].keys()  # Assuming all dictionaries have the same keys

    with open(csv_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Write the data
        writer.writerows(dict_list)


def write_jsonl(data, path):
    with open(path, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")


def write_json(data, path):
    json_object = json.dumps(data, indent=4)

    # Writing to sample.json
    with open(path, "w") as f:
        f.write(json_object)


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def write_csv(data, path, header=None):
    with open(path, "w") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(data)


def read_json(file_path):
    """
    Reads a JSON file and returns the parsed Python object.

    Parameters:
        file_path (str): The path to the JSON file.

    Returns:
        dict or list: The data loaded from the JSON file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
    except json.JSONDecodeError:
        print(f"Error: File is not a valid JSON - {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
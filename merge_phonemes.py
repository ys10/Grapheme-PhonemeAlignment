# coding=UTF-8
import re

input_file_name = "input_data.txt"
output_file_name = "output_data.txt"
output_lines = list()
with open(input_file_name, "r") as input_file:
    while 1:
        lines = input_file.readlines(10000)
        if not lines:
            break
        # for line in lines:
        #     trans_list = line.split("\t")
        #     if len(trans_list[0]) < len(trans_list[1].split(" ")):
        #         print(line)
        #     pass
        for line in lines:
            pattern = re.compile("c.*s iy2")
            match = re.match(pattern, line)
            if match:
                print(match.group())
            else:
                output_lines.append(line)
            pass
        pass
    pass

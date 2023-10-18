#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_file> <output_file>"
    exit 1
fi

# Assign the arguments to variables for better readability
input_file=$1
output_file=$2

# Get the header line
head -n 1 $input_file > $output_file

# Get random lines from the file (excluding the header)
tail -n +2 $input_file | shuf -n 5000 >> $output_file

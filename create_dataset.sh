#!/bin/bash
# Check if the output directory argument is provided
if [ $# -lt 2 ]; then
    echo "Please provide both a directory and an output directory as arguments."
    exit 1
fi

# Check if the directory exists
if [ ! -d "$1" ]; then
    echo "Directory '$1' does not exist."
    exit 1
fi

# Check if the output directory exists
if [ ! -d "$2" ]; then
    echo "Output directory '$2' does not exist."
    exit 1
fi

# Check if a third argument is provided and use it to automatically confirm the move
if [ $# -eq 3 ] && [ "$3" == "-y" ]; then
    auto_response="y"
else
    auto_response="n"
fi

# Count the number of files in the subdirectories
file_count=$(find "$1" -type f ! -name "*.json" | wc -l)

# Prompt the user to confirm the move
echo "Are you sure you want to move all files ($file_count) from subdirectories of '$1' to '$2'? (y/n)"
if [ "$auto_response" == "y" ]; then
    echo "Move automatically confirmed."
    response="y"
else
    read -r response
    if [ "$response" != "y" ]; then
        echo "Operation cancelled."
        exit 1
    fi
fi
# Move files from subdirectories to the output directory, excluding JSON files
find "$1" -type f ! -name "*.json" -exec mv {} "$2" \;

echo "All files from subdirectories, excluding JSON files, have been moved to '$2'."
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

# Prompt the user to confirm the move
echo "Are you sure you want to move all files from subdirectories of '$1' to '$2'? (y/n)"
read -r response
if [ "$response" != "y" ]; then
    echo "Operation cancelled."
    exit 1
fi

# Move files from subdirectories to the output directory
find "$1" -type f -exec mv {} "$2" \;

echo "All files from subdirectories have been moved to '$2'."
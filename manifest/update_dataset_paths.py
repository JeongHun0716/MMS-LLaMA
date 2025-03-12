import argparse
from pathlib import Path

def modify_paths(file_path, vox2_root, muavic_root, lrs3_root, avs_root):
    """
    Reads a TSV file with placeholders and replaces them with the provided absolute dataset paths.
    The processed content is written back to the same file.
    """
    # Mapping of placeholders to absolute paths, removing any trailing slash.
    placeholder_map = {
        '{VOX2_ROOT}': vox2_root.rstrip('/'),
        '{LRS3_ROOT}': lrs3_root.rstrip('/'),
    }
    
    # Read all lines from the input TSV file.
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    processed_lines = []
    for line in lines:
        # Replace each placeholder in the line with the provided absolute path.
        for placeholder, absolute in placeholder_map.items():
            line = line.replace(placeholder, absolute)
        processed_lines.append(line.rstrip('\n'))
    
    # Write the updated lines back to the same file.
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(processed_lines) + "\n")

def process_all_tsvs(input_dir, vox2_root, muavic_root, lrs3_root, avs_root):
    """
    Recursively finds all TSV files in the input directory and applies the modify_paths function.
    """
    input_path = Path(input_dir)
    tsv_files = list(input_path.rglob("*.tsv"))
    if not tsv_files:
        print(f"No TSV files found in {input_dir}")
        return
    for tsv_file in tsv_files:
        print(f"Processing file: {tsv_file}")
        modify_paths(tsv_file, vox2_root, muavic_root, lrs3_root, avs_root)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Recursively replace placeholders in TSV files with user-provided absolute dataset paths. "
                    "The input files will be overwritten."
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help="Input directory containing TSV files with placeholders.")
    parser.add_argument('--vox2', type=str, required=True,
                        help="Absolute path for the VOX2 dataset (e.g., /Dataset/vox2).")
    parser.add_argument('--lrs3', type=str, required=True,
                        help="Absolute path for the LRS3 dataset (e.g., /Dataset/lrs3).")
    args = parser.parse_args()

    process_all_tsvs(args.input_dir, args.vox2, args.lrs3)
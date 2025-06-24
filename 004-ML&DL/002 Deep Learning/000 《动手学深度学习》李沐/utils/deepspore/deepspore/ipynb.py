import json


def ipynb2(ipynb_file:str, label:str, output_file:str) -> None:
    '''
    Collect code blocks started with label, such as "#@save" in ipynb format file to one file named with outputfile.
    Args:
        ipynb_file: str
        label: str, such as %%bash, #@save and etc.
        output_file: str
    
    # Demo:
    >>> extract_save_blocks(ipynb_file= 'learn_PyTorch.ipynb', label= '#@save', output_file= 'utils/utils.py')
    '''
    
    with open(ipynb_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    saved_code_blocks = []

    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source_lines = cell.get('source', [])
            if source_lines and source_lines[0].lstrip().startswith(label):
                code_block = ''.join(source_lines)
                saved_code_blocks.append(code_block)

    if saved_code_blocks:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            f_out.write("# This file is generated from saved notebook code blocks\n\n")
            f_out.write("\n\n\n".join(saved_code_blocks))
        print(f"Saved {len(saved_code_blocks)} block(s) to {output_file}")
    else:
        print(f"No {label} blocks found.")
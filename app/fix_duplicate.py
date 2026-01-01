with open('chess_detection.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Remove lines 1020-1235 (0-indexed: 1019-1234)
new_lines = lines[:1019] + lines[1235:]

with open('chess_detection.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"Removed {1235-1019} duplicate lines. New file has {len(new_lines)} lines.")

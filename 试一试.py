import os

# 定义要运行的文件路径
script_folder = r"D:\Dosktop\all code\5883getdata\GLEAM"
scripts = ["E.py", "Et.py", "SMrz.py", "SMs.py"]

# 依次运行每个脚本
for script in scripts:
    script_path = os.path.join(script_folder, script)
    print(f"Running: {script_path}")
    os.system(f'python "{script_path}"')  # 调用系统命令运行脚本
    print(f"Finished: {script_path}")

print("All scripts executed.")

import subprocess

def run_program(command):
    """
    运行一个命令并检查是否成功执行。

    Args:
        command (str): 要执行的命令。

    Returns:
        bool: 如果执行成功返回 True，否则返回 False。
    """
    try:
        # 使用 subprocess 执行命令
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        # 输出命令执行的结果
        print("Command Output:")
        print(result.stdout)
        
        # 检查执行状态码
        if result.returncode == 0:
            print("Program executed successfully.")
            return True
        else:
            print("Program execution failed.")
            print("Error Output:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

if __name__ == "__main__":
    # 要检查的命令
    command = input("Enter the command to execute: ")
    
    # 检查程序是否执行成功
    if run_program(command):
        print("The program ran without errors.")
    else:
        print("The program encountered an error.")

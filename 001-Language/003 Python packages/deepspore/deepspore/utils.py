from multiprocessing import Pool, cpu_count
import subprocess 
import os 


def run_cmd(cmd: str) -> str:
    '''Run cmd with subprocess module.
    args:
        cmd: str.
    return：
        The state of the command.
    >>>run_cmd(cmd='ls -alh')
    '''
    try:
        subprocess.run(cmd, shell=True, check=True)
        return f"Success: {cmd}"
    except subprocess.CalledProcessError as e:
        return f"Failed: {cmd} (Error: {e})"


def num_thread() -> int:
    '''Return the thread numbers.'''
    from multiprocessing import cpu_count 
    return cpu_count()


def on_error(error: str) -> None:
    '''Print the error informations.'''
    print(error)


def parallel(tasks, num_threads= cpu_count()):
    '''Paralles with Multiprocessing module of Python
    Args:
        tasks: list of tuples, each tuple contains a function and its parameters
        num_threads: number of threads to use
    '''
    print(f"Running {len(tasks)} tasks in parallel with {num_threads} threads.")
    try:
        pool = Pool(processes=num_threads)
        for func, params in tasks:
            # 立马执行并获取错误结果
            pool.apply_async(func, params, error_callback= on_error)
    except:
        print("An error occurred while executing tasks.")
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    
    print("All tasks completed.")
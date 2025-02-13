import os
# try to compute a suggested max number of worker based on system's resource
max_num_worker_suggest = None
cpuset_checked = False
if hasattr(os, 'sched_getaffinity'):
    try:
        max_num_worker_suggest = len(os.sched_getaffinity(0))
        cpuset_checked = True
    except Exception:
        pass
cpu_count = os.cpu_count()


print("max_num_worker_suggest based on sched_getaffinity", max_num_worker_suggest)
print("cpu_count", cpu_count)
print("cpuset_checked: ", cpuset_checked)


from datasets import load_dataset, load_from_disk
# 'sst2',"mnli", "mnli_mismatched", "mnli_matched", "qnli", "rte", "wnli and so on"
glue_tsk="mrpc"
dataset = load_dataset("glue", glue_tsk)
dataset.save_to_disk(f"./workspace/dataset/{glue_tsk}") # 保存到该目录下
print("success")
# dataset = load_from_disk(f"/home/wangyukun/workspace/new_kd/dataset/wnli")

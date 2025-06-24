#!/bin/bash

# 读取输入参数
BASE_DIR="$1"   # subject 目录
SUBJECT_ID="$2" # subject ID

# 如果没有传参，则默认处理100610
if [ -z "$BASE_DIR" ] || [ -z "$SUBJECT_ID" ]; then
  echo "Usage: $0 <subject_directory> <subject_id>"
  exit 1
fi

# 需要处理的标签列表
LABELS=("9" "10" "11" "12" "13" "14" "15")

# 遍历每个标签
for label in "${LABELS[@]}"; do
  LABEL_REL_PATH="label/label_${label}.nii.gz"  # 当前标签的标签文件路径
  OUTPUT_ROOT="/media/UG1/wzh/MyData/Pacels_7Networks/result/${SUBJECT_ID}/mask_results_${label}" # 对应的输出目录

  # 创建输出目录（如果不存在）
  mkdir -p "$OUTPUT_ROOT"

  # 查找所有名称包含 _X 或 X_ 的目录（例如 1_9, 9_10）
  find "$BASE_DIR" -maxdepth 1 -type d \( -name "*_${label}" -o -name "${label}_*" \) | while read -r dir; do
    # 提取目录基名 (例如 1_9)
    dir_name=$(basename "$dir")

    # 定义当前目录的路径
    tck_dir="$dir/Cluster_clean_in_yeo_space"
    label_file="$LABEL_REL_PATH"
    output_file="$OUTPUT_ROOT/${dir_name}.nii.gz"

    # 跳过不存在的目录
    if [ ! -d "$tck_dir" ]; then
      echo "Skipping $dir_name: Cluster_clean_in_yeo_space not found"
      continue
    fi
    if [ ! -f "$label_file" ]; then
      echo "Skipping $dir_name: label file $label_file not found"
      continue
    fi

    # 初始化总和文件
    sum_file="$OUTPUT_ROOT/sum_mask_tmp.nii.gz"
    mrcalc "$label_file" 0 -mul "$sum_file" -quiet

    # 遍历当前目录下的所有 .tck 文件
    echo "==== Processing directory: $dir_name for label ${label} ===="
    for tck_file in "$tck_dir"/cluster_*.tck; do
      # 跳过无匹配文件的情况
      [ ! -f "$tck_file" ] && continue

      # 提取文件名基名
      base_name=$(basename "$tck_file" .tck)
      echo "Processing: $base_name"

      # Step 1: 生成纤维密度图
      density_file="$OUTPUT_ROOT/${base_name}_density.nii.gz"
      overlap_file="$OUTPUT_ROOT/${base_name}_overlap.nii.gz"
      mask_file="$OUTPUT_ROOT/${base_name}_mask.nii.gz"

      tckmap -template "$label_file" -contrast tdi "$tck_file" "$density_file"
      mrcalc "$density_file" "$label_file" -mult "$overlap_file"
      mrcalc "$overlap_file" 0 -gt "$mask_file"

      # Step 2: 累加到总和文件
      mrcalc "$sum_file" "$mask_file" -add "$OUTPUT_ROOT/sum_mask_tmp2.nii.gz"
      mv "$OUTPUT_ROOT/sum_mask_tmp2.nii.gz" "$sum_file"

      # 清理中间文件
      rm "$density_file" "$overlap_file" "$mask_file"
    done

    # 重命名最终文件
    mv "$sum_file" "$output_file"
    echo "Saved: $output_file"
  done
done

#!/bin/bash

# 사용법 안내
usage() {
    echo "Usage: $0 -m <model_path> -v <video_directory>"
    exit 1
}

# 모델 경로와 비디오 디렉토리를 파라미터로 받기
while getopts ":m:v:" opt; do
  case $opt in
    m) model_path="$OPTARG"
    ;;
    v) video_directory="$OPTARG"
    ;;
    *) usage
    ;;
  esac
done

# 모델 경로 또는 비디오 디렉토리가 설정되지 않은 경우 사용법 출력
if [ -z "$model_path" ] || [ -z "$video_directory" ]; then
    usage
fi

# 비디오 디렉토리 내의 모든 mp4 파일에 대해 파이썬 스크립트 실행
for video_file in "$video_directory"/*.mp4; do
    if [ -f "$video_file" ]; then
        echo "Processing $video_file"
        python src/pred.py -m "$model_path" -v "$video_file"
    fi
done

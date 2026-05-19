import supervision as sv

# Load YOLO dataset
dataset = sv.DetectionDataset.from_yolo(
    images_directory_path="/mnt/hdd10tb/Users/thuongnh/datasets/test_sat_personhand/images",
    annotations_directory_path="/mnt/hdd10tb/Users/thuongnh/datasets/test_sat_personhand/labels",
    data_yaml_path="data_hand.yaml"
)

# Save as COCO
dataset.as_coco(
    images_directory_path="/mnt/hdd10tb/Users/thuongnh/datasets/test_sat_personhand/coco_sat/images",
    annotations_path="/mnt/hdd10tb/Users/thuongnh/datasets/test_sat_personhand/coco_sat/annotations.json"
)
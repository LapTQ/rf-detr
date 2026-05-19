from rfdetr import RFDETRMedium

model = RFDETRMedium()

model.train(
    dataset_dir="/home/laptq/laptq-fs26-shoplifting-detection/data/satudora_mix_kita8jo_19k_v2",
    epochs=200,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir="outputs/train/fs26/v1.medium.person_hand",
    early_stopping=True,
    early_stopping_patience=15,
    early_stopping_min_delta=0.005,
    device="cuda:5",
    # warmup_epochs
    # lr_scheduler
    # accelerator
    # seed
    progress_bar=True
)
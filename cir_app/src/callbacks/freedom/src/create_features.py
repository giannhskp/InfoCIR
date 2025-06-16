import os
import numpy as np
import torch
import pickle
import csv
import argparse
import open_clip
from PIL import Image
from torch.utils.data import DataLoader
from utils_features import *
from utils import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
from src import config


def parse_args():
    parser = argparse.ArgumentParser(description="Frature extraction parameters")
    parser.add_argument(
        "--dataset",
        choices=["corpus", "imagenet_r", "nico", "minidn", "ltll"],
        type=str,
        help="define dataset",
    )
    parser.add_argument(
        "--backbone",
        choices=["clip", "siglip"],
        default="clip",
        type=str,
        help="choose the backbone",
    )
    parser.add_argument("--batch", default=512, type=int, help="choose a batch size")
    parser.add_argument(
        "--gpu", default=0, type=int, metavar="gpu", help="Choose a GPU id"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.backbone == "siglip":
        model, preprocess = open_clip.create_model_from_pretrained(
            "hf-hub:timm/ViT-L-16-SigLIP-256"
        )
        tokenizer = open_clip.get_tokenizer("hf-hub:timm/ViT-L-16-SigLIP-256")
    elif args.backbone == "clip":
        model, preprocess = open_clip.create_model_from_pretrained("ViT-L/14", "openai")
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
    device = setup_device(gpu_id=args.gpu)
    model.to(device)
    model.eval()

    save_dir = os.path.join(config.WORK_DIR, f"{args.backbone}_features", args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    if args.dataset == "corpus":
        corpus_path = os.path.join(config.WORK_DIR, "open_image_v7_class_names.csv")
        save_file = os.path.join(save_dir, "open_image_v7_class_names.pkl")
        save_corpus_features(
            model=model,
            tokenizer=tokenizer,
            corpus_path=corpus_path,
            save_file=save_file,
            device=device,
        )
        print("Corpus features saved successfully:", save_file)
    else:
        dataset_types = ["query", "database"]
        if args.dataset in ["imagenet_r", "ltll"]:
            dataset_types = ["full"]
        for dataset_type in dataset_types:
            csv_dir = os.path.dirname(config.DATASET_ROOT_PATH)
            path = os.path.join(csv_dir, f"{dataset_type}_files.csv")
            dataset = ImageDomainLabels(path, root=config.DATASET_ROOT_PATH, csv_with_names_path=config.AUGMENTED_DATASET_PATH, preprocess=preprocess)
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
            )
            save_file = os.path.join(
                save_dir, f"{dataset_type}_{args.dataset}_features.pkl"
            )
            save_dataset_features(
                model=model, dataloader=dataloader, save_file=save_file, device=device
            )
            print("Dataset features saved successfully:", save_file)

            # NEW: Save image names to separate file
            name_list = [dataset.image_ids[i] for i in range(len(dataset))]
            name_file = os.path.join(
                save_dir, f"{dataset_type}_{args.dataset}_names.pkl"
            )
            with open(name_file, "wb") as f:
                pickle.dump(name_list, f)
            print(f"Saved image names to: {name_file}")


if __name__ == "__main__":
    main()
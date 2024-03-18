import os
import shutil
from collections import defaultdict

import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import (
    file_exists,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
)
from tqdm import tqdm

import src.settings as s


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # Possible structure for bbox case. Feel free to modify as you needs.

    train_path = "/home/alex/DATASETS/TODO/VisDrone_video/train/sequences"
    val_path = "/home/alex/DATASETS/TODO/VisDrone_video/val/sequences"
    test_dev_path = "/home/alex/DATASETS/TODO/VisDrone_video/test-dev/sequences"
    test_challenge_path = "/home/alex/DATASETS/TODO/VisDrone_video/challenge/sequences"

    batch_size = 30
    images_ext = ".jpg"
    anns_ext = ".txt"

    # ds_name_to_data = {"train": train_path, "val": val_path, "test": test_dev_path}
    ds_name_to_data = {"test": test_dev_path}

    def create_ann(image_path):
        labels = []
        tags = [seq]

        if ds_name == "test":
            if image_path.split("/")[-3] == "test-dev":
                test_tag = sly.Tag(dev)
                tags.append(test_tag)
            else:
                test_tag = sly.Tag(challenge)
                tags.append(test_tag)

        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        ann_data = frame_to_data.get(get_file_name(image_path))
        if ann_data is not None:

            for curr_data in ann_data:
                l_tags = []
                obj_class = idx_to_class[int(curr_data[6])]

                target_value = int(curr_data[0])
                target = sly.Tag(target_meta, value=target_value)
                l_tags.append(target)

                occlusion_meta = idx_to_occlusion[int(curr_data[8])]
                occlusion = sly.Tag(occlusion_meta)
                l_tags.append(occlusion)

                truncation_meta = idx_to_truncation[int(curr_data[7])]
                truncation = sly.Tag(truncation_meta)
                l_tags.append(truncation)

                left = int(curr_data[1])
                top = int(curr_data[2])
                right = left + int(curr_data[3])
                bottom = top + int(curr_data[4])
                rectangle = sly.Rectangle(top=top, left=left, bottom=bottom, right=right)
                label = sly.Label(rectangle, obj_class, tags=l_tags)
                labels.append(label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=tags)

    idx_to_class = {
        0: sly.ObjClass("ignored region", sly.Rectangle),
        1: sly.ObjClass("pedestrian", sly.Rectangle),
        2: sly.ObjClass("person", sly.Rectangle),
        3: sly.ObjClass("bicycle", sly.Rectangle),
        4: sly.ObjClass("car", sly.Rectangle),
        5: sly.ObjClass("van", sly.Rectangle),
        6: sly.ObjClass("truck", sly.Rectangle),
        7: sly.ObjClass("tricycle", sly.Rectangle),
        8: sly.ObjClass("awning tricycle", sly.Rectangle),
        9: sly.ObjClass("bus", sly.Rectangle),
        10: sly.ObjClass("motor", sly.Rectangle),
        11: sly.ObjClass("other", sly.Rectangle),
    }

    challenge = sly.TagMeta("challenge", sly.TagValueType.NONE)
    dev = sly.TagMeta("dev", sly.TagValueType.NONE)
    seq_meta = sly.TagMeta("sequence", sly.TagValueType.ANY_STRING)
    target_meta = sly.TagMeta("target id", sly.TagValueType.ANY_NUMBER)
    no_occlusion = sly.TagMeta("no occlusion", sly.TagValueType.NONE)
    partial_occlusion = sly.TagMeta("partial occlusion", sly.TagValueType.NONE)
    heavy_occlusion = sly.TagMeta("heavy occlusion", sly.TagValueType.NONE)

    idx_to_occlusion = {0: no_occlusion, 1: partial_occlusion, 2: heavy_occlusion}

    no_truncation = sly.TagMeta("no truncation", sly.TagValueType.NONE)
    partial_truncation = sly.TagMeta("partial truncation", sly.TagValueType.NONE)

    idx_to_truncation = {0: no_truncation, 1: partial_truncation}

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(
        obj_classes=list(idx_to_class.values()),
        tag_metas=[
            challenge,
            dev,
            no_occlusion,
            partial_occlusion,
            heavy_occlusion,
            no_truncation,
            partial_truncation,
            seq_meta,
            target_meta,
        ],
    )
    api.project.update_meta(project.id, meta.to_json())

    for ds_name, data_path in ds_name_to_data.items():

        dataset = api.dataset.create(
            project.id, get_file_name(ds_name), change_name_if_conflict=True
        )

        for subfolder in os.listdir(data_path):

            seq_value = subfolder[3:-2]
            seq = sly.Tag(seq_meta, value=seq_value)

            curr_data_path = os.path.join(data_path, subfolder)

            curr_ann_path = curr_data_path.replace("sequences", "annotations") + ".txt"

            frame_to_data = defaultdict(list)

            with open(curr_ann_path) as f:
                content = f.read().split("\n")

                for curr_data in content:
                    if len(curr_data) != 0:
                        curr_data = curr_data.split(",")
                        im_name = curr_data[0].zfill(7)
                        frame_to_data[im_name].append(curr_data[1:])

            images_names = os.listdir(curr_data_path)

            progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

            for images_names_batch in sly.batched(images_names, batch_size=batch_size):
                im_names_batch = []
                img_pathes_batch = []
                for image_name in images_names_batch:
                    im_names_batch.append(seq_value + "_" + image_name)
                    img_pathes_batch.append(os.path.join(curr_data_path, image_name))

                img_infos = api.image.upload_paths(dataset.id, im_names_batch, img_pathes_batch)
                img_ids = [im_info.id for im_info in img_infos]

                anns = [create_ann(image_path) for image_path in img_pathes_batch]
                api.annotation.upload_anns(img_ids, anns)

                progress.iters_done_report(len(images_names_batch))

    for subfolder in os.listdir(test_challenge_path):
        seq_value = subfolder[3:-2]
        seq = sly.Tag(seq_meta, value=seq_value)

        curr_data_path = os.path.join(test_challenge_path, subfolder)

        curr_ann_path = curr_data_path.replace("sequences", "annotations") + ".txt"
        images_names = os.listdir(curr_data_path)

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        for images_names_batch in sly.batched(images_names, batch_size=batch_size):
            im_names_batch = []
            img_pathes_batch = []
            for image_name in images_names_batch:
                im_names_batch.append(seq_value + "_" + image_name)
                img_pathes_batch.append(os.path.join(curr_data_path, image_name))

            img_infos = api.image.upload_paths(dataset.id, im_names_batch, img_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns = [create_ann(image_path) for image_path in img_pathes_batch]
            api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(images_names_batch))

    return project

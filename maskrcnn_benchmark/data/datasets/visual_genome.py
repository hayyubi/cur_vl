import json
import numpy as np
import torch
import torchvision
import torch.utils.data as data
import os
from pycocotools.coco import COCO
from PIL import Image
from maskrcnn_benchmark.structures.bounding_box import BoxList
from visual_genome.visual_genome import VisualGenome
from maskrcnn_benchmark.data.datasets.abstract import AbstractDataset
from collections import defaultdict

def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj.bbox[2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    return True


class VisualGenomeDataset(AbstractDataset):
    def __init__(
        self, img_ann_file, det_ann_file, root, remove_images_without_annotations,
        phase_objects_file=None, label_emb_file=None, transforms=None, extra_args=None, 
        img_caps_id=None, is_train=True, 
    ):
        self.root = os.path.expanduser(root)
        self.vg = VisualGenome(img_ann_file, det_ann_file)

        # sort indices for reproducible results
        self.ids = sorted(list(self.vg.imgs.keys()))

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                anno = self.vg.loadAnns(img_id)[0]
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        if extra_args["DEBUG"]:
            self.ids = self.ids[:10] if is_train else self.ids[:10]

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

        # Variables for Det
        self.categories = self.vg.objects
        self.cat_name_to_id = {v: k for k, v in self.categories.items()}
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.vg.getObjectIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        self.contiguous_class_list = [None for _ in 
                                    range(max(self.contiguous_category_id_to_json_id.keys()) + 1)]
        for i, id in self.contiguous_category_id_to_json_id.items():
            self.contiguous_class_list[i] = self.categories[id]
        self.contiguous_class_list[0] = 'background'

        if phase_objects_file:
            with open(phase_objects_file, 'r') as f:
                object_names_by_phases = json.load(f)
            self.objects_by_phases = defaultdict(list)
            for phase, objects in object_names_by_phases.items():
                self.objects_by_phases[phase].extend(
                    [self.json_category_id_to_contiguous_id[self.cat_name_to_id[obj]]
                    for obj in objects]
                )

        # Word Embeddings for zeros shot class recognition
        if getattr(extra_args, 'LOAD_EMBEDDINGS', False):
            self.class_embeddings = {}
            with open(label_emb_file, 'r') as fin:
                label_embs = json.load(fin)
                for label, embds in label_embs.items():
                    self.class_embeddings[self.cat_name_to_id[label]] = np.asarray(
                                            embds[extra_args['EMB_KEY']], dtype=np.float32)
            self.class_emb_mtx = np.zeros(
                (len(self.contiguous_category_id_to_json_id) + 1, extra_args['EMB_DIM']),
                dtype=np.float32)
            for i, cid in self.contiguous_category_id_to_json_id.items():
                self.class_emb_mtx[i, :] = self.class_embeddings[cid]

        # Required for AbstractDataset Type, mandatory for coco type evaluation
        self.CLASSES = list(self.categories.values())
        self.CLASSES.insert(0, "__background__")
        self.initMaps()

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        # Load Image.
        path = self.vg.loadImgs(img_id)[0].filename
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        # Load obj detections anns
        det_anno = self.vg.loadAnns(img_id)[0]

        boxes = [obj.bbox for obj in det_anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        det_target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [self.vg.getIdOfObject(obj) for obj in det_anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        det_target.add_field("labels", classes)

        det_target = det_target.clip_to_image(remove_empty=True)

        # Final Transformation
        if self._transforms is not None:
            img, det_target = self._transforms(img, det_target)

        target = det_target

        return img, target, idx

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self._transforms.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.vg.imgs[img_id]
        img_info = {'id': img_data.id, 'url': img_data.url, 'width': img_data.width,
                     'height': img_data.height, 'file_name': str(img_data.id)+'.jpg',
                     'coco_id': img_data.coco_id, 'flickr_id': img_data.flickr_id}
        return img_info

    def set_class_labels(self, categories, json_category_id_to_contiguous_id):
        '''
        For multi-label mode only
        Should be called to register the list of categories before calling __getitem__()
        '''
        self.categories = categories
        self.json_category_id_to_contiguous_id = json_category_id_to_contiguous_id
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.num_categories = max(list(self.contiguous_category_id_to_json_id.keys())) + 1
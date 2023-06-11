import json
import random
import numpy as np
import torch
import torchvision
import torch.utils.data as data
import os
from pycocotools.coco import COCO
from PIL import Image
from maskrcnn_benchmark.structures.bounding_box import BoxList
from collections import defaultdict

def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    return True


class COCOCapDetDataset(data.Dataset):
    def __init__(
        self, cap_ann_file, det_ann_file, root, remove_images_without_annotations,
        phase_objects_file=None, transforms=None, extra_args=None, img_caps_id=None,
        is_train=True
    ):
        self.root = os.path.expanduser(root)
        self.coco_cap = COCO(cap_ann_file)
        self.coco_det = COCO(det_ann_file)

        # check for curriculum learning
        self.is_curriculum = True if img_caps_id else False

        # sort indices for reproducible results
        self.ids = sorted(list(self.coco_cap.imgs.keys()))

        if self.is_curriculum:
            img_caps_id = sorted(img_caps_id, key=lambda l: l[0])
            img_ids, cap_ids = zip(*img_caps_id)
            img_ids, cap_ids = list(img_ids), list(cap_ids)
            self.ids = img_ids
            self.cap_ids = cap_ids

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            cap_ids = []
            for idx, img_id in enumerate(self.ids):
                cap_ann_ids = self.coco_cap.getAnnIds(imgIds=img_id)
                cap_anno = self.coco_cap.loadAnns(cap_ann_ids)
                det_ann_ids = self.coco_det.getAnnIds(imgIds=img_id, iscrowd=None)
                det_anno = self.coco_det.loadAnns(det_ann_ids)
                if len(cap_anno) > 0 and has_valid_annotation(det_anno):
                    ids.append(img_id)
                    if self.is_curriculum:
                        cap_ids.append(self.cap_ids[idx])
            self.ids = ids
            if self.is_curriculum:
                self.cap_ids = cap_ids

        # If debugging limit data size to 10 images
        if extra_args["DEBUG"]:
           self.ids = self.ids[:10] if is_train else self.ids[:10]

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

        # Variables for Cap
        self.multilabel_mode = extra_args.get('MULTI_LABEL_MODE', False)

        # Variables for Det
        self.categories = {cat['id']: cat['name'] for cat in self.coco_det.cats.values()}
        self.cat_name_to_id = {v: k for k, v in self.categories.items()}
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco_det.getCatIds())
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
        self.class_splits = {}
        if getattr(extra_args, 'LOAD_EMBEDDINGS', False):
            self.class_embeddings = {}
            with open(det_ann_file, 'r') as fin:
                ann_data = json.load(fin)
                for item in ann_data['categories']:
                    emb = item['embedding'][extra_args['EMB_KEY']]
                    self.class_embeddings[item['id']] = np.asarray(emb, dtype=np.float32)
                    if 'split' in item:
                        if item['split'] not in self.class_splits:
                            self.class_splits[item['split']] = []
                        self.class_splits[item['split']].append(item['id'])
            self.class_emb_mtx = np.zeros(
                (len(self.contiguous_category_id_to_json_id) + 1, extra_args['EMB_DIM']),
                dtype=np.float32)
            for i, cid in self.contiguous_category_id_to_json_id.items():
                self.class_emb_mtx[i, :] = self.class_embeddings[cid]

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        # Load Image. It will be same for cap and det.
        path = self.coco_cap.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        # Load captions
        # Cap Ann Ids come from the img_id tp caption map for curriculum learning
        if self.is_curriculum:
            cap_ann_ids = [self.cap_ids[idx]]
        else:
            cap_ann_ids = self.coco_cap.getAnnIds(imgIds=img_id)

        anns = self.coco_cap.loadAnns(cap_ann_ids)
        captions = [ann['caption'] for ann in anns]

        if self.multilabel_mode:
            if self.is_curriculum:
                raise NotImplementedError("Multilabel mode with curriculum not implemented")
            caption = self.convert_to_multilabel_anno(captions)
        else:
            # captions is a list of sentences. Pick one randomly.
            # TODO use a more deterministic approach, especially for validation
            # For curriculum learning there's only one, so pick it
            caption = np.random.choice(captions) if len(captions) > 0 else captions[0]

        # Load obj detections anns
        det_ann_ids = self.coco_det.getAnnIds(imgIds=img_id)
        dets = self.coco_det.loadAnns(det_ann_ids)
        det_anno = [det for det in dets if det["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in det_anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        det_target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in det_anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        det_target.add_field("labels", classes)

        det_target = det_target.clip_to_image(remove_empty=True)

        # Final Transformation
        if self._transforms is not None:
            img, det_target = self._transforms(img, det_target)

        target = (caption, det_target)

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
        img_data = self.coco_cap.imgs[img_id]
        return img_data

    def convert_to_multilabel_anno(self, sentence_list):
        anno = np.zeros((self.num_categories), dtype=np.float32)
        for cid, cind in self.json_category_id_to_contiguous_id.items():
            cname = self.categories[cid].lower()
            for sent in sentence_list:
                if cname in sent.lower():
                    anno[cind] = 1
        return anno


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
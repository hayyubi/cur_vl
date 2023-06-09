import json
import numpy as np
import torch
import torchvision

class COCOCaptionsDataset(torchvision.datasets.coco.CocoCaptions):
    def __init__(
        self, ann_file, root, remove_images_without_annotations,
        transforms=None, extra_args=None, img_caps_id=None,
    ):
        super(COCOCaptionsDataset, self).__init__(root, ann_file)
        # sort indices for reproducible results

        self.img_caps_id = img_caps_id
        self.ids = sorted(self.ids)

        if self.img_caps_id:
            img_caps_id = sorted(img_caps_id, key=lambda l: l[0])
            img_ids, cap_ids = zip(*img_caps_id)
            img_ids, cap_ids = list(img_ids), list(cap_ids)
            self.ids = img_ids
            self.cap_ids = cap_ids
            self.id_to_cap_map = {k: v for k, v in enumerate(self.cap_ids)}

        # filter images without detection annotations
        if remove_images_without_annotations and not self.img_caps_id:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anno = self.coco.loadAnns(ann_ids)
                if len(anno) > 0:
                    ids.append(img_id)
            self.ids = ids

        # If debugging limit data size to 10 images
        if extra_args["DEBUG"]:
            self.ids = self.ids[:10]

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms
        self.multilabel_mode = extra_args.get('MULTI_LABEL_MODE', False)


    def __getitem__(self, idx):
        img, anno = super(COCOCaptionsDataset, self).__getitem__(idx)
        if self.multilabel_mode:
            anno = self.convert_to_multilabel_anno(anno)
        else:
            # anno is a list of sentences. Pick one randomly.
            # TODO use a more deterministic approach, especially for validation
            anno = np.random.choice(anno)

        if self._transforms is not None:
            img, _ = self._transforms(img, None)

        if self.img_caps_id:
            anno = self.coco.loadAnns([self.id_to_cap_map[idx]])[0]
            anno = anno['caption']

        return img, anno, idx


    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
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
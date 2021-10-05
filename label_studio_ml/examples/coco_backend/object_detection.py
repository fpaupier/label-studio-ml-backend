import typing as tp
import json

from label_studio_ml.model import LabelStudioMLBase
from yolo.detect import run


class ObjectDetector(LabelStudioMLBase):
    """A YOLO object detector"""

    def __init__(self, **kwargs):
        super(ObjectDetector, self).__init__(**kwargs)
        # TODO: FPR: model initialization should be here

    def predict(self, tasks: tp.List[tp.Dict], **kwargs) -> tp.List[tp.Dict]:
        """
        From a collections of task, generate the matching predictions.


        Note:
            For information on the Task and Prediction JSON data format see the LabelStudio doc at
             https://labelstud.io/guide/export.html#Relevant-JSON-property-descriptions


        """
        if len(tasks) == 0:
            return []

        from_name, schema = list(self.parsed_label_config.items())[0]
        to_name = schema['to_name'][0]
        # Iterate over each task to generate its matching prediction
        preds = []
        for task in tasks:
            # Sanity check to recover the image
            if 'data' not in task:
                continue
            if 'image' not in task['data']:
                continue

            img_url: str = task['data']['image']

            pred = run(source=img_url, weights='/Users/fpaupier/projects/label-studio-ml-backend/label_studio_ml/examples/coco_backend/weights/yolov5s.pt')
            print(pred)
            pred = json.loads(pred)
            preds.append(pred)
            print(pred)
            num_preds: int = len(pred['result'])
            # for i in range(num_preds):
            #     print(pred['result'][i])
            #     pred['result'][i] = from_name
            #     pred['result'][i] = to_name
            #     print(pred['result'][i])

        print('RETURNING:', preds)
        return preds

    def fit(self, completions, workdir=None, **kwargs):
        """"Not required if you don't need to trigger model training"""
        raise NotImplementedError

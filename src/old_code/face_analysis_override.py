from insightface.app import FaceAnalysis
import onnxruntime, glob
from insightface import model_zoo
import os.path as osp
from insightface.app.common import Face
import numpy as np
from insightface.model_zoo.model_zoo import ModelRouter, get_default_providers, get_default_provider_options

def get_model(model_file, **kwargs):
    router = ModelRouter(model_file)
    providers = kwargs.get('providers', get_default_providers())
    provider_options = kwargs.get('provider_options', get_default_provider_options())
    model = router.get_model(providers=providers, provider_options=provider_options)
    return model


class FaceAnalysisOverride(FaceAnalysis):
    def __init__(self, model_path, allowed_modules=None, **kwargs):
        onnxruntime.set_default_logger_severity(3)
        self.models = {}
        onnx_files = [model_path]
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            model = get_model(onnx_file, **kwargs)
            if model is None:
                print('model not recognized:', onnx_file)
            elif allowed_modules is not None and model.taskname not in allowed_modules:
                print('model ignore:', onnx_file, model.taskname)
                del model
            elif model.taskname not in self.models and (allowed_modules is None or model.taskname in allowed_modules):
                print('find model:', onnx_file, model.taskname, model.input_shape, model.input_mean, model.input_std)
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']


    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
            else:
                model.prepare(ctx_id)

    def get(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric='default')
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.models.items():
                if taskname=='detection':
                    continue
                model.get(img, face)
            ret.append(face)
        return ret

    def draw_on(self, img, faces):
        import cv2
        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(np.int32)
            color = (0, 0, 255)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            if face.kps is not None:
                kps = face.kps.astype(np.int32)
                #print(landmark.shape)
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color,
                               2)
            if face.gender is not None and face.age is not None:
                cv2.putText(dimg,'%s,%d'%(face.sex,face.age), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)

            #for key, value in face.items():
            #    if key.startswith('landmark_3d'):
            #        print(key, value.shape)
            #        print(value[0:10,:])
            #        lmk = np.round(value).astype(np.int)
            #        for l in range(lmk.shape[0]):
            #            color = (255, 0, 0)
            #            cv2.circle(dimg, (lmk[l][0], lmk[l][1]), 1, color,
            #                       2)
        return dimg


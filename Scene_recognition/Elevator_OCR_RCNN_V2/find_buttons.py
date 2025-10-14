# Scene_recognition/Elevator_OCR_RCNN_V2/find_buttons.py
# Unified detector+OCR runner (TF2 compat.v1), utils 없이 동작
# - TensorFlow 임포트/로드/추론 로그 조용히
# - 모델 그래프/세션은 최초 1회만 로드하여 재사용(전역 캐시)
# - CLI와 모듈 API(find_buttons) 겸용

import os
# ===== 환경변수는 어떤 import보다 위에서 설정 =====
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # TF C++ 로그: ERROR만
os.environ["GLOG_minloglevel"] = "2"       # glog 대비
os.environ["PYTHONWARNINGS"] = "ignore"    # 파이썬 경고 무시(선택)

import sys
import shutil
import argparse
import numpy as np
import cv2
from contextlib import redirect_stderr, redirect_stdout, contextmanager

# OpenCV 로그 억제 (가능한 경우)
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.ERROR)
except Exception:
    pass

# ===== TensorFlow를 '무음 임포트'로 로드 =====
def _quiet_import_tf():
    with open(os.devnull, "w") as devnull, redirect_stderr(devnull), redirect_stdout(devnull):
        import tensorflow as tf  # noqa
    import tensorflow as tf
    # 파이썬 로거도 조용히
    try:
        tf.get_logger().setLevel("ERROR")
    except Exception:
        pass
    try:
        from absl import logging as absl_logging
        absl_logging.set_verbosity(absl_logging.FATAL)
    except Exception:
        pass
    return tf

tf = _quiet_import_tf()
tf.compat.v1.disable_eager_execution()  # frozen graph(v1) 사용

# ===== 무음 컨텍스트 (필요 구간만 STDERR 잠금) =====
@contextmanager
def silence_stderr():
    with open(os.devnull, "w") as devnull, redirect_stderr(devnull):
        yield

# ======================
# HERE 상수 (스크립트 폴더)
# ======================
HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------
# Defaults & charset (47)
# ------------------------
DEFAULT_DET_NAMES = [
    'detection_graph.pb',               # variable input
    'detection_graph_640x480.pb',       # fixed 640x480
    'detection_graph_640x480_optimized.pb',
]
DEFAULT_OCR_NAMES = [
    'ocr_graph.pb',
    'ocr_graph_optimized.pb',
]

CHARSET = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
           'A':10,'B':11,'C':12,'D':13,'E':14,'F':15,'G':16,'H':17,'I':18,
           'J':19,'K':20,'L':21,'M':22,'N':23,'O':24,'P':25,'R':26,'S':27,
           'T':28,'U':29,'V':30,'X':31,'Z':32,'<':33,'>':34,'(' :35,')':36,
           '$':37,'#':38,'^':39,'s':40,'-':41,'*':42,'%' :43,'?':44,'!' :45,
           '+' :46}
IDX2CHAR = {v:k for k,v in CHARSET.items()}

# ------------------------
# Tensor helpers
# ------------------------
def load_graph(pb_path):
    g = tf.Graph()
    with silence_stderr():  # 그래프 import 시 떠드는 로그 컷
        with tf.io.gfile.GFile(pb_path, 'rb') as f:
            gd = tf.compat.v1.GraphDef()
            gd.ParseFromString(f.read())
        with g.as_default():
            tf.import_graph_def(gd, name='')
    return g

def get_tensor(graph, name):
    try:
        return graph.get_tensor_by_name(name)
    except Exception:
        return None

# ------------------------
# Detection
# ------------------------
DET_INPUT_TENSOR   = 'image_tensor:0'
DET_BOXES_TENSOR   = 'detection_boxes:0'
DET_SCORES_TENSOR  = 'detection_scores:0'
DET_CLASSES_TENSOR = 'detection_classes:0'
DET_NUM_TENSOR     = 'num_detections:0'

def run_detection(img_bgr, sess, tdict, score_thr=0.5, top_k=100):
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inp = np.expand_dims(img_rgb, 0)  # uint8
    # sess.run에서만 조용히
    with silence_stderr():
        boxes, scores, classes, num = sess.run(
            [tdict['boxes'], tdict['scores'], tdict['classes'], tdict['num']],
            feed_dict={tdict['input']: inp}
        )
    n = int(num[0])
    boxes, scores, classes = boxes[0][:n], scores[0][:n], classes[0][:n]
    order = np.argsort(-scores)[:min(top_k, n)]
    out = []
    for i in order:
        if scores[i] < score_thr:
            continue
        ymin, xmin, ymax, xmax = boxes[i]
        x1, y1 = int(xmin*W), int(ymin*H)
        x2, y2 = int(xmax*W), int(ymax*H)
        out.append({'bbox':(x1,y1,x2,y2), 'score':float(scores[i]), 'cls':int(classes[i])})
    return out

# ------------------------
# OCR
# ------------------------
OCR_INPUT_NAMES   = ['ocr_input:0']
OCR_LOGPROB_NAMES = ['chars_log_prob:0']
OCR_PRED_CHARS    = ['predicted_chars:0']
OCR_PRED_SCORES   = ['predicted_scores:0']

def preprocess_ocr_input(crop_bgr, size=180):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    arr = rgb.astype(np.uint8)
    return arr[np.newaxis, ...], rgb  # (1,H,W,3), RGB

def decode_from_logprob(logp):
    probs = np.exp(logp)
    seq = np.argmax(probs, axis=2)[0]
    confs = np.max(probs, axis=2)[0]
    chars, conf_list = [], []
    for idx, c in zip(seq, confs):
        ch = IDX2CHAR.get(int(idx))
        if ch is None or ch == '+':
            continue
        chars.append(ch)
        conf_list.append(float(c))
    text = ''.join(chars) if chars else ''
    conf = float(np.mean(conf_list)) if conf_list else 0.0
    return text, conf

def decode_from_predicted(codes, scores):
    chars, conf_list = [], []
    for code, s in zip(codes, scores):
        ch = IDX2CHAR.get(int(code))
        if ch is None or ch == '+':
            continue
        chars.append(ch)
        conf_list.append(float(s))
    text = ''.join(chars) if chars else ''
    conf = float(np.mean(conf_list)) if conf_list else 0.0
    return text, conf

# ------------------------
# 전역 캐시(싱글톤) : 최초 1회 로드 후 재사용
# ------------------------
_DET = {"graph": None, "sess": None, "tdict": None, "path": None}
_OCR = {"graph": None, "sess": None, "tdict": None, "path": None, "mode": None}

def _init_detector(model_path):
    if _DET["graph"] is None or _DET["path"] != model_path:
        g = load_graph(model_path)
        with g.as_default():
            sess = tf.compat.v1.Session(graph=g)
            tdict = {
                'input'  : get_tensor(g, DET_INPUT_TENSOR),
                'boxes'  : get_tensor(g, DET_BOXES_TENSOR),
                'scores' : get_tensor(g, DET_SCORES_TENSOR),
                'classes': get_tensor(g, DET_CLASSES_TENSOR),
                'num'    : get_tensor(g, DET_NUM_TENSOR),
            }
            if any(v is None for v in tdict.values()):
                print('[!] Detection 텐서를 못 찾음. 그래프 텐서명이 표준이 아님.')
                for k,v in tdict.items():
                    print('   ', k, '->', v)
                sys.exit(1)
        _DET.update(graph=g, sess=sess, tdict=tdict, path=model_path)
    return _DET["sess"], _DET["tdict"]

def _init_ocr(model_path, prefer_mode='auto'):
    if _OCR["graph"] is None or _OCR["path"] != model_path:
        g = load_graph(model_path)
        with g.as_default():
            sess = tf.compat.v1.Session(graph=g)
            # 입력 텐서
            ocr_in = None
            for nm in OCR_INPUT_NAMES:
                ocr_in = get_tensor(g, nm)
                if ocr_in is not None:
                    break
            if ocr_in is None:
                print('[!] OCR 입력 텐서(ocr_input:0)를 찾지 못함.')
                sys.exit(1)

            # 출력 모드 탐색
            mode = prefer_mode
            ocr_out = None
            if mode in ('auto','predicted'):
                t_chars = t_scores = None
                for nm in OCR_PRED_CHARS:
                    t_chars = get_tensor(g, nm)
                    if t_chars is not None:
                        break
                for nm in OCR_PRED_SCORES:
                    t_scores = get_tensor(g, nm)
                    if t_scores is not None:
                        break
                if t_chars is not None and t_scores is not None:
                    mode = 'predicted'
                    ocr_out = (t_chars, t_scores)
            if ocr_out is None and mode in ('auto','logprob'):
                t_logp = None
                for nm in OCR_LOGPROB_NAMES:
                    t_logp = get_tensor(g, nm)
                    if t_logp is not None:
                        break
                if t_logp is not None:
                    mode = 'logprob'
                    ocr_out = t_logp
            if ocr_out is None:
                print('[!] OCR 출력 텐서를 찾지 못함. (predicted_* 혹은 chars_log_prob)')
                sys.exit(1)

        _OCR.update(graph=g, sess=sess,
                    tdict={"in": ocr_in, "out": ocr_out},
                    path=model_path, mode=mode)
    return _OCR["sess"], _OCR["tdict"], _OCR["mode"]

def preload_models(det_path, ocr_path, do_warmup=True):
    """실행 초기에 한 번 호출해서 모델/세션을 미리 준비(옵션: 워밍업)"""
    det_sess, det_t = _init_detector(det_path)
    ocr_sess, ocr_t, _ = _init_ocr(ocr_path, 'auto')
    if do_warmup:
        # 디텍터 워밍업(더미 1x480x640x3)
        dummy = np.zeros((1, 480, 640, 3), dtype=np.uint8)
        with silence_stderr():
            det_sess.run(
                [det_t['boxes'], det_t['scores'], det_t['classes'], det_t['num']],
                feed_dict={det_t['input']: dummy}
            )
        # OCR 워밍업(작은 더미)
        dummy_crop = np.zeros((1, 180, 180, 3), dtype=np.uint8)
        with silence_stderr():
            out = ocr_t["out"]
            if isinstance(out, tuple):
                ocr_sess.run(list(out), feed_dict={ocr_t["in"]: dummy_crop})
            else:
                ocr_sess.run(out, feed_dict={ocr_t["in"]: dummy_crop})
    return {"det": (det_sess, det_t), "ocr": (ocr_sess, ocr_t)}

# ------------------------
# Main
# ------------------------
def find_buttons(argv=None, ctx=None):
    ap = argparse.ArgumentParser(description='Unified Button Finder (det + ocr)')
    ap.add_argument('--image', default=os.path.join(HERE, '../Test_data/sample.png'),
                    help='입력 이미지 경로 (기본: HERE/../sample.png)')
    ap.add_argument('--frozen_dir', default=os.path.join(HERE, 'frozen_model'),
                    help='모델 폴더 (기본: HERE/frozen_model)')
    ap.add_argument('--det_graph', default='auto',
                    help="검출 그래프 파일명 또는 'auto'")
    ap.add_argument('--ocr_graph', default='auto',
                    help="OCR 그래프 파일명 또는 'auto'")
    ap.add_argument('--ocr_output', choices=['auto','predicted','logprob'], default='auto',
                    help="OCR 출력 형식")
    ap.add_argument('--score_thr', type=float, default=0.5, help='검출 점수 임계값')
    ap.add_argument('--max_dets', type=int, default=100, help='최대 디텍션 수')
    ap.add_argument('--force_resize_640x480', action='store_true',
                    help='검출 입력을 640x480으로 강제 리사이즈')
    ap.add_argument('--process_dir', default=os.path.join(HERE, 'process'),
                    help='과정 이미지 저장 폴더 (있으면 지움)')
    ap.add_argument('--output', default=os.path.join(HERE, 'OCR_RCNN_V2_out.png'),
                    help='최종 시각화 저장 경로')
    ap.add_argument('--thickness', type=int, default=1,
                    help='라벨 텍스트/선 굵기')
    ap.add_argument('--no_vis', action='store_true',
                    help='disable visualization (no drawing, no output image)')
    ap.add_argument('--no_save', action='store_true',
                help='최종 합성 이미지를 파일로 저장하지 않음')
    args = ap.parse_args(argv)

    # 준비: process 폴더 재생성
    if os.path.exists(args.process_dir):
        shutil.rmtree(args.process_dir, ignore_errors=True)
    os.makedirs(args.process_dir, exist_ok=True)

    # 모델 경로 결정
    def pick_model(name_or_auto, defaults, frozen_dir):
        if name_or_auto != 'auto':
            p = name_or_auto if os.path.isabs(name_or_auto) else os.path.join(frozen_dir, name_or_auto)
            if not os.path.exists(p):
                print(f'[!] 지정한 모델이 없음: {p}')
                sys.exit(1)
            return p
        for fname in defaults:
            cand = os.path.join(frozen_dir, fname)
            if os.path.exists(cand):
                return cand
        print(f'[!] auto로 탐색했지만 모델을 못 찾음: {defaults}')
        sys.exit(1)

    det_pb = pick_model(args.det_graph, DEFAULT_DET_NAMES, args.frozen_dir)
    ocr_pb = pick_model(args.ocr_graph, DEFAULT_OCR_NAMES, args.frozen_dir)

    # print(f'[i] DET: {det_pb}')
    # print(f'[i] OCR: {ocr_pb}')

    # 세션/텐서 준비(컨텍스트가 있으면 재사용, 없으면 전역 캐시 사용)
    if ctx and "det" in ctx and "ocr" in ctx:
        det_sess, det_tensors = ctx["det"]
        ocr_sess, ocr_tensors = ctx["ocr"]
        ocr_mode = _OCR.get("mode", "auto")
    else:
        det_sess, det_tensors = _init_detector(det_pb)
        ocr_sess, ocr_tensors, ocr_mode = _init_ocr(ocr_pb, args.ocr_output)

    # print(f'[i] OCR mode: {ocr_mode}')

    # 이미지 읽기
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        print(f'[!] 이미지 못 읽음: {args.image}')
        sys.exit(1)

    # 옵션: 640x480 강제
    det_input_img = img
    if args.force_resize_640x480:
        det_input_img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)

    dets = run_detection(det_input_img, det_sess, det_tensors,
                         score_thr=args.score_thr, top_k=args.max_dets)

    vis = img.copy()
    result_list = []

    for k, d in enumerate(dets, start=1):
        x1, y1, x2, y2 = d['bbox']

        # 리사이즈 모드면 bbox를 원본 스케일로 환산
        if args.force_resize_640x480 and det_input_img.shape != img.shape:
            H0, W0 = det_input_img.shape[:2]
            H1, W1 = img.shape[:2]
            scale_x = W1 / float(W0)
            scale_y = H1 / float(H0)
            x1 = int(x1 * scale_x); x2 = int(x2 * scale_x)
            y1 = int(y1 * scale_y); y2 = int(y2 * scale_y)

        x1 = max(0, min(x1, img.shape[1]-1))
        x2 = max(0, min(x2, img.shape[1]-1))
        y1 = max(0, min(y1, img.shape[0]-1))
        y2 = max(0, min(y2, img.shape[0]-1))
        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # 저장: 원본 crop
        crop_path = os.path.join(args.process_dir, f'button_{k:02d}_crop.png')
        cv2.imwrite(crop_path, crop)

        # OCR 입력 준비 + 저장
        inp, inp_rgb = preprocess_ocr_input(crop, size=180)
        inp_path = os.path.join(args.process_dir, f'button_{k:02d}_inp.png')
        cv2.imwrite(inp_path, cv2.cvtColor(inp_rgb, cv2.COLOR_RGB2BGR))

        # OCR 추론 + 디코딩 (무음)
        out = ocr_tensors["out"]
        with silence_stderr():
            if isinstance(out, tuple):  # predicted 모드
                t_chars, t_scores = out
                codes, scores = ocr_sess.run([t_chars, t_scores], feed_dict={ocr_tensors["in"]: inp})
                codes, scores = np.squeeze(codes), np.squeeze(scores)
                text, conf = decode_from_predicted(codes, scores)
            else:  # logprob 모드
                logp = ocr_sess.run(out, feed_dict={ocr_tensors["in"]: inp})  # (1,T,47)
                text, conf = decode_from_logprob(logp)

        # 각 crop 위에 라벨 저장
        if not args.no_vis:
            vis_crop = crop.copy()
            cv2.putText(vis_crop, f'{text} ({conf:.2f})', (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),
                        args.thickness, cv2.LINE_AA)
            vis_crop_path = os.path.join(args.process_dir, f'button_{k:02d}_vis.png')
            cv2.imwrite(vis_crop_path, vis_crop)

        # 전체 이미지에도 그리기
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(vis, f'{text} ({conf:.2f})', (x1, max(0,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),
                    args.thickness, cv2.LINE_AA)

        result = (f"[{d['score']:.2f}] {d['bbox']} -> {text} ({conf:.2f})")
        result_list.append(result)

    if not args.no_save:
        cv2.imwrite(args.output, vis)
        print('saved:', args.output + '\n')

    return result_list

# ===== CLI 진입점 =====
if __name__ == '__main__':
    # uv run ... 2>/dev/null 로도 실행 가능
    # 또는 사전 프리로드:
    # preload_models(os.path.join(HERE,'frozen_model','detection_graph.pb'),
    #                os.path.join(HERE,'frozen_model','ocr_graph.pb'), do_warmup=True)
    find_buttons()

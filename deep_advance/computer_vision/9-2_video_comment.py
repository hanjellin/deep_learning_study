import numpy as np
import cv2 as cv
import sys

###############################################
#  YOLOv3 구성: 가중치, cfg 파일, 클래스 이름 로드
###############################################
def construct_yolo_v3():

    # COCO 데이터셋의 클래스 이름을 읽어옴
    # (예: person, car, dog 등 80개)
    with open('coco_names.txt', 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines()]

    # YOLOv3 모델(가중치 + 구조 파일) 로딩
    model = cv.dnn.readNet('yolov3.weights', 'yolov3.cfg')

    # 전체 계층(layer) 이름을 가져옴
    layer_names = model.getLayerNames()

    # 실제 출력층(Output layers)만 가져오기
    # YOLOv3는 3개의 output layer를 가짐
    out_layers = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]

    return model, out_layers, class_names


###################################################
#   YOLO 검출 함수: 입력 이미지 → bounding box 추출
###################################################
def yolo_detect(img, yolo_model, out_layers):

    # 입력 이미지의 세로, 가로 크기
    height, width = img.shape[:2]

    # 이미지 전처리: blob 생성
    # - 1/256 스케일링
    # - (448, 448) 크기로 리사이즈
    # - BGR → RGB 변환(swapRB=True)
    test_img = cv.dnn.blobFromImage(img, 1.0/256, (448, 448),
                                    (0, 0, 0), swapRB=True)

    # YOLO 입력으로 blob 설정
    yolo_model.setInput(test_img)

    # Forward 실행 → YOLO output 얻기 (3개의 feature map)
    output3 = yolo_model.forward(out_layers)

    # 결과 저장용 리스트 생성
    boxes = []      # bounding box 위치
    confidences = [] # confidence score(신뢰도)
    class_ids = []   # 클래스 인덱스(0~79)

    # YOLO output parsing
    for output in output3:
        for vec85 in output:

            # vector 85: [center_x, center_y, width, height, obj_score, class_scores(80개)]
            scores = vec85[5:]                   # class score 부분만 분리
            class_id = np.argmax(scores)         # 가장 확률 높은 클래스 선택
            confidence = scores[class_id]        # 해당 클래스의 confidence 값

            # confidence threshold 적용 (0.5 이상만 선택)
            if confidence > 0.5:

                # YOLO 출력은 비율(relative value)이므로 실제 pixel 좌표로 변환
                center_x = int(vec85[0] * width)
                center_y = int(vec85[1] * height)
                w = int(vec85[2] * width)
                h = int(vec85[3] * height)

                # bounding box 왼쪽 상단 좌표 구하기
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, x + w, y + h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 박스가 하나도 없으면 그대로 반환
    if len(boxes) == 0:
        return []

    # NMS(Non-Maximum Suppression)
    # - IoU가 일정 이상 겹치는 박스들을 제거
    # - confidence 높은 박스만 남김
    indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # 최종 detected object만 모아서 출력 리스트 구성
    objects = [boxes[i] + [confidences[i]] + [class_ids[i]]
               for i in range(len(boxes)) if i in indices]

    return objects


###############################################
#   메인 코드 시작: 비디오 파일 입력 → 검출
###############################################

# YOLO 모델 로드
model, out_layers, class_names = construct_yolo_v3()

# 각 클래스별로 다른 색상 랜덤 생성 (bounding box용)
colors = np.random.uniform(0, 255, size=(len(class_names), 3))

# ------------------------------------------------------
# 동영상 파일 로드 (웹캠 대신)
# ------------------------------------------------------
video_path = 'images/walking.avi'    # ← 여기를 원하는 동영상 파일로 변경
cap = cv.VideoCapture(video_path)

# 비디오 파일이 정상적으로 열렸는지 확인
if not cap.isOpened():
    sys.exit('❌ 동영상 파일을 열 수 없습니다: ' + video_path)

# ------------------------------------------------------
# 비디오 프레임 반복 처리
# ------------------------------------------------------
while True:

    # 프레임 읽기
    ret, frame = cap.read()

    # ret=False → 동영상 끝 또는 읽기 실패
    if not ret:
        print('📌 동영상 재생이 종료되었습니다.')
        break

    # YOLO 검출 수행
    results = yolo_detect(frame, model, out_layers)

    # 검출된 객체들을 화면에 그리기
    for x1, y1, x2, y2, conf, class_id in results:

        # Bounding box 텍스트 (class + confidence)
        text = f'{class_names[class_id]} {conf:.3f}'

        # 박스 그리기
        cv.rectangle(frame, (x1, y1), (x2, y2),
                     colors[class_id], 2)

        # 클래스 이름 쓰기
        cv.putText(frame, text, (x1, y1 + 30),
                   cv.FONT_HERSHEY_PLAIN, 1.5,
                   colors[class_id], 2)

    # 결과 프레임 출력
    cv.imshow("YOLO v3 Object Detection", frame)

    # 'q' 키 입력 시 종료
    # 비디오 재생은 보통 25~33ms 정도로 waitKey 설정하면 자연스러움
    if cv.waitKey(30) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv.destroyAllWindows()

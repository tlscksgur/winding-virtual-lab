import cv2
import ffmpeg
import numpy as np

COLOR_PALETTE = [
    # (255, 0, 0),     # Red
    # (0, 255, 0),     # Green
    # (0, 0, 255),     # Blue
    (180, 180, 180),
    (180, 180, 180),
    (180, 180, 180),
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 128),  # Purple
    (255, 165, 0),  # Orange
    (0, 128, 255),  # Sky Blue
]


def get_color(class_idx):
    return COLOR_PALETTE[class_idx % len(COLOR_PALETTE)]


def pad_image_to_width(image: np.ndarray, target_width: int, color: tuple = (0, 0, 0)) -> np.ndarray:
    width = image.shape[1]
    if width >= target_width:
        return image  # 이미 충분히 넓으면 패딩 불필요

    pad_left = (target_width - width) // 2
    pad_right = target_width - width - pad_left
    padded = cv2.copyMakeBorder(image, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=color)
    return padded


def draw_dotted_line(img, pt1, pt2, color, thickness=1, gap=5):
    dist = np.linalg.norm(np.array(pt2) - np.array(pt1))
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r))
        y = int((pt1[1] * (1 - r) + pt2[1] * r))
        pts.append((x, y))

    for i in range(0, len(pts) - 1, 2):  # 짝수 인덱스만 그려서 점선처럼 보이게
        cv2.line(img, pts[i], pts[i + 1], color, thickness)

    return img


def overlay_contours(image, contours, color=(0, 255, 0), alpha=0.5, border_thickness=5):
    if contours is None:
        return image

    if not isinstance(contours, list):
        if contours.ndim == 2 and contours.shape[1] == 2:
            contours = contours.reshape(-1, 1, 2)
        contours = [contours]

    result = image.copy()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            continue

        # ROI 참조
        roi = result[y : y + h, x : x + w]

        mask_roi = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask_roi, [cnt], -1, 255, thickness=cv2.FILLED, offset=(-x, -y))

        overlay = np.zeros_like(roi)
        overlay[:] = color
        blended = cv2.addWeighted(roi, 1 - alpha, overlay, alpha, 0)

        cv2.copyTo(blended, mask_roi, roi)

        # 흰색 테두리
        cv2.drawContours(result, [cnt], -1, (255, 255, 255), thickness=border_thickness)

    return result


def compute_template_match_diff(frame1, frame2, template_size=100):
    h, w = frame1.shape
    cx, cy = w // 2, h // 2
    half = template_size // 2

    template = frame1[cy - half : cy + half, cx - half : cx + half]
    result = cv2.matchTemplate(frame2, template, cv2.TM_SQDIFF_NORMED)
    min_val, _, min_loc, _ = cv2.minMaxLoc(result)
    return min_val  # 작을수록 유사


def compute_diffs(source, target, apply_blur=True, blur_ksize=5):
    source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    if apply_blur:
        source = cv2.GaussianBlur(source, (blur_ksize, blur_ksize), 0)
        target = cv2.GaussianBlur(target, (blur_ksize, blur_ksize), 0)
    return compute_template_match_diff(source, target)


def draw_dashed_curve(
    input_image, curve_points, color=(0, 0, 255), thickness=2, dash_length=5, gap_length=10
):
    image_copy = input_image.copy()
    curve_points = curve_points[::gap_length]

    for i in range(1, len(curve_points)):
        pt1 = np.array(curve_points[i - 1], dtype=np.float32)
        pt2 = np.array(curve_points[i], dtype=np.float32)
        direction = pt2 - pt1
        distance = np.linalg.norm(direction)

        if distance == 0:
            continue

        direction = direction / distance  # 단위 벡터

        # 대시와 간격의 총 길이 계산
        total_length = dash_length + gap_length

        # 점선 그리기: 최소 대시 1개 이상 그리기
        num_dashes = int(distance // total_length)
        if num_dashes == 0:
            num_dashes = 1

        # 점선 그리기
        for j in range(num_dashes):
            start = pt1 + direction * (j * total_length)
            end = pt1 + direction * (j * total_length + dash_length)

            # 대시 그리기
            cv2.line(image_copy, tuple(start.astype(int)), tuple(end.astype(int)), color, thickness)

    return image_copy


def draw_text_with_background(
    image,
    text,
    position,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=1.5,
    text_color=(255, 255, 255),
    bg_color=(0, 0, 0),
    thickness=3,
    padding=10,
    alpha=0.6,
):
    """
    텍스트에 배경을 추가하여 이미지에 그리는 함수
    """
    overlay = image.copy()
    x, y = position
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # 배경 사각형 좌표 계산
    top_left = (x - padding, y - text_height - padding)
    bottom_right = (x + text_width + padding, y + baseline + padding)

    # 배경 사각형 그리기 (반투명)
    cv2.rectangle(overlay, top_left, bottom_right, bg_color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # 텍스트 그리기
    cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    return image


def overlay_text_with_background(image, bbox, text="Initializing..."):
    x1, y1, x2, y2 = bbox

    roi = image[y1:y2, x1:x2].copy()
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), (0, 0, 0), -1)

    alpha = 0.7
    blended = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)

    image[y1:y2, x1:x2] = blended

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    text_x = x1 + (x2 - x1 - text_size[0]) // 2
    text_y = y1 + (y2 - y1 + text_size[1]) // 2

    cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return image


def image_dir_to_mp4(image_dir: str, output_file: str, fps: int = 20, image_extension: str = "jpg"):
    """
    Convert a directory of images to an MP4 file using ffmpeg.

    Args:
        image_dir (str): Path to the directory containing images.
        output_file (str): Path to the output MP4 file.
        fps (int): Frames per second for the output video.
        image_extension (str): File extension of the input images.
    """
    try:
        (
            ffmpeg.input(f"{image_dir}/*.{image_extension}", pattern_type="glob", framerate=fps)
            .output(output_file, vcodec="libx264", pix_fmt="yuv420p", loglevel="quiet")
            .run(overwrite_output=True)
        )
    except ffmpeg.Error as e:
        print(f"Error converting images in {image_dir} to {output_file}: {e}")

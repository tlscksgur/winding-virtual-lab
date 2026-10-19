import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from .visualization_tools import (
    draw_dotted_line,
    overlay_contours,
    pad_image_to_width,
)
from ..data.data_class import SideSegmentationResults
from ..modules.line_tracker import MultiLineTracker
from ..engine.monitoring import ShiftMonitor, DirectionMonitor
from ..engine.config import CommonSettings

# COLOR
ROI_LINE_COLOR = (0, 255, 0)  # Green
SHIFT_BOX_COLOR = (255, 0, 0)  # Red
DRUM_BODY_LINE_COLOR = (49, 148, 248)  # Orange
WINDING_CABLE_LINE_COLOR = (71, 55, 212)  # Red
ANOMALY_BOX_COLOR = (255, 0, 255)  # Magenta

# VISUALIZATION CONSTANTS
DEFAULT_PADDING = 20
POINT_PADDING = 100
SUB_FIGURE_WIDTH = 1200
UPPER_SUB_FIGURE_HEIGHT_RATIO = 0.6
DESIRED_WIDTH_PX = 1500
DPI = 200
FIG_WIDTH_IN = DESIRED_WIDTH_PX / DPI


class ResultVisualizer:
    def __init__(self):
        self.frame_height = None
        self.frame_width = None

        self.progress_history = []
        self.ng_frame_list = []
        self.drum_shift_frame_list = []
        self.sponge_frame_list = []
        self._init_winding_figure()

    def __call__(
        self,
        frame: np.ndarray,
        side_view_frame: np.ndarray,
        prediction: dict,
        side_segmentation_results: SideSegmentationResults,
        direction_monitor: DirectionMonitor | None = None,
        shift_monitor: ShiftMonitor | None = None,
    ):
        # Prepare frame and monitoring region
        original = frame.copy()
        self._set_frame_dimensions(original)
        monitoring_vis, left_x, right_x = self._create_monitoring_visual(
            original, prediction, shift_monitor
        )

        # Initialize canvas and place monitoring
        self._initialize_canvas()
        self.canvas[: self.frame_height, : self.frame_width] = monitoring_vis

        # Prepare segmentation and defect visualizations
        seg_vis = self._create_segmentation_visual(
            frame, prediction, left_x, right_x, direction_monitor, shift_monitor
        )
        side_vis = self._create_side_view_visual(
            side_view_frame, shift_monitor, side_segmentation_results
        )

        # Place sub-figures on canvas
        self._place_sub_figures(seg_vis, side_vis)

        # Update progress plot and monitoring board
        self._update_progress(prediction)
        self._plot_winding_progress()

        # Finalize and return
        self.canvas = cv2.cvtColor(self.canvas, cv2.COLOR_RGB2BGR)
        self.canvas = cv2.resize(
            self.canvas, (self.canvas.shape[1] // 2, self.canvas.shape[0] // 2)
        )  # resize to half
        return self.canvas

    def _set_frame_dimensions(self, image: np.ndarray):
        self.frame_height, self.frame_width = image.shape[:2]

    def _create_monitoring_visual(
        self, image: np.ndarray, prediction: dict, shift_monitor: ShiftMonitor | None
    ):
        contours = prediction.get("segmentation_contours", {})

        left_drum = contours["left_drum"]
        right_drum = contours["right_drum"]

        if left_drum is not None and right_drum is not None:
            x_min = max(left_drum[:, 0].min() - POINT_PADDING, 0)
            x_max = min(right_drum[:, 0].max() + POINT_PADDING, self.frame_width)
        else:
            body = contours.get("drum_body")
            if body is not None:
                x_min = max(body[:, 0].min() - POINT_PADDING, 0)
                x_max = min(body[:, 0].max() + POINT_PADDING, self.frame_width)
            else:
                x_min, x_max = 0, self.frame_width
                vis = image.copy()

        roi = (
            x_min,
            DEFAULT_PADDING,
            x_max - x_min,
            self.frame_height - 2 * DEFAULT_PADDING,
        )
        vis = self.draw_dotted_box(image.copy(), roi, ROI_LINE_COLOR)

        # Draw shifting box if present
        shift = prediction["cable_shifting_box"]
        if shift is not None:
            vis = self.draw_dotted_box(vis, shift, SHIFT_BOX_COLOR)

        self.put_status_text(vis, prediction, shift_monitor)

        return vis, x_min, x_max

    def _initialize_canvas(self):
        width = self.frame_width + SUB_FIGURE_WIDTH + 2 * DEFAULT_PADDING
        height = self.frame_height + DEFAULT_PADDING
        self.canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    def _create_segmentation_visual(
        self,
        image: np.ndarray,
        prediction: dict,
        left_x: int,
        right_x: int,
        direction_monitor: DirectionMonitor | None = None,
        shift_monitor: ShiftMonitor | None = None,
    ) -> np.ndarray:
        contours = prediction["segmentation_contours"]

        vis = image.copy()
        if contours["left_drum"] is not None:
            vis = overlay_contours(vis, contours["left_drum"], color=(20, 58, 184), alpha=0.6)
        if contours["right_drum"] is not None:
            vis = overlay_contours(vis, contours["right_drum"], color=(43, 116, 243), alpha=0.6)
        if contours["drum_body"] is not None:
            vis = overlay_contours(vis, contours["drum_body"], color=(255, 127, 0), alpha=0.6)

        if direction_monitor is not None:
            vis = self._draw_progress_range(
                vis,
                direction_monitor.winding_cable_moving_range_left_point,
                direction_monitor.winding_cable_moving_range_right_point,
                direction_monitor.winding_cable_point,
            )  # Draw progress range based on body points
        if shift_monitor is not None:
            vis = self._draw_side_shift_range(
                vis,
                shift_monitor.tightening_score_calculator.side_drum_shift_x_range,
            )
            vis = self._draw_swing_range(
                vis,
                shift_monitor.swing_score_calculator.swing_range,
            )

        if contours["winding_cable"] is not None:
            vis = overlay_contours(
                vis, contours["winding_cable"], color=WINDING_CABLE_LINE_COLOR, alpha=0.6
            )
            vis = self._draw_body_lines(vis, contours["drum_body"])

        # Add NG boxes to the existing drum shift visualization
        ng_boxes = prediction.get("ng_boxes")
        vis = self._draw_ng_boxes(vis, ng_boxes)

        # Crop and resize
        title_width = int((right_x - left_x) * 0.95)
        title_height = title_width // 9
        cropped = vis[:, left_x:right_x]
        target_size = (SUB_FIGURE_WIDTH // 2, int(self.frame_height * UPPER_SUB_FIGURE_HEIGHT_RATIO))

        if cropped.size < 1:
            return cv2.resize(vis, target_size)

        cv2.rectangle(
            cropped,
            (DEFAULT_PADDING, DEFAULT_PADDING),
            (DEFAULT_PADDING + title_width, DEFAULT_PADDING + title_height),
            color=(10, 10, 10),
            thickness=-1,
        )
        cv2.putText(
            cropped,
            "Monitoring Drum Shift...",
            (DEFAULT_PADDING * 2, int(DEFAULT_PADDING * 4.5)),
            cv2.FONT_HERSHEY_TRIPLEX,
            2,
            (180, 180, 180),
            3,
        )

        return cv2.resize(cropped, target_size)

    def _draw_ng_boxes(
        self,
        image: np.ndarray,
        ng_boxes: np.ndarray | None,
        border_thickness: int = 5,
        text_margin: int = 30,
    ) -> np.ndarray:
        """Draw NG boxes on the image."""
        if ng_boxes is None:
            return image

        vis = image.copy()
        for x1, y1, x2, y2 in ng_boxes:
            # Draw bounding box
            cv2.rectangle(
                vis,
                (x1, y1),
                (x1 + x2, y1 + y2),
                ANOMALY_BOX_COLOR,
                thickness=border_thickness,
            )
            cv2.putText(
                vis,
                "NG",
                (x1, max(y1 - text_margin, text_margin)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                ANOMALY_BOX_COLOR,
                2,
            )

        return vis

    def _draw_progress_range(self, image, left_point, right_point, current_point, range_height=800):
        for point in (left_point, right_point):
            if point is None:
                continue

            x, y = point
            image = draw_dotted_line(
                image,
                (x, y - range_height // 2),
                (x, y + range_height // 2),
                color=(0, 250, 250),
                thickness=5,
                gap=10,
            )
            cv2.circle(image, (x, y), 15, (0, 250, 250), -1)

        if current_point is not None:
            image = draw_dotted_line(
                image,
                (current_point[0], current_point[1] - range_height // 2),
                (current_point[0], current_point[1] + range_height // 2),
                color=(255, 0, 0),
                thickness=5,
                gap=10,
            )
            cv2.circle(image, (current_point[0], current_point[1]), 15, (255, 0, 0), -1)

        return image

    def _draw_side_shift_range(self, image, shift_x_range: list[int] | None):
        if shift_x_range is not None:
            left_x, right_x = shift_x_range
            image = draw_dotted_line(
                image,
                (left_x, 400),
                (left_x, 500),
                color=(0, 0, 255),
                thickness=5,
                gap=10,
            )
            image = draw_dotted_line(
                image,
                (right_x, 400),
                (right_x, 500),
                color=(0, 0, 255),
                thickness=5,
                gap=10,
            )
            image = draw_dotted_line(
                image,
                (left_x, 450),
                (right_x, 450),
                color=(0, 0, 255),
                thickness=5,
                gap=10,
            )
        return image

    def _draw_swing_range(self, image, swing_range: list[int] | None):
        if swing_range is not None:
            left_x, right_x = swing_range
            image = draw_dotted_line(
                image,
                (left_x, 850),
                (left_x, 950),
                color=(0, 0, 255),
                thickness=5,
                gap=10,
            )
            image = draw_dotted_line(
                image,
                (right_x, 850),
                (right_x, 950),
                color=(0, 0, 255),
                thickness=5,
                gap=10,
            )
            image = draw_dotted_line(
                image,
                (left_x, 900),
                (right_x, 900),
                color=(0, 0, 255),
                thickness=5,
                gap=10,
            )
        return image

    def _create_side_view_visual(
        self,
        side_view_frame: np.ndarray,
        shift_monitor: ShiftMonitor,
        side_segmentation_results: SideSegmentationResults,
    ):
        # side view drum box crop 사진과 line detection 시각화 (위아래로 붙이기)
        drum_vis = SideViewResultVisualizer.visualize_side_drum_prediction(
            side_view_frame, side_segmentation_results
        )

        target_size = (
            SUB_FIGURE_WIDTH // 2,
            int(self.frame_height * UPPER_SUB_FIGURE_HEIGHT_RATIO),
        )

        if drum_vis is None:
            return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

        # Get line tracker from shift monitor
        line_tracker = getattr(shift_monitor, "multi_line_tracker", None)

        line_vis = SideViewResultVisualizer.visualize_line_detection(
            image=side_view_frame,
            line_tracker=line_tracker,
            side_drum_box=side_segmentation_results.side_drum_box,
        )

        # fallback: line detection이 유효하지 않으면 드럼 예측만 사용
        if line_vis.size == 0:
            return cv2.resize(drum_vis, target_size)

        drum_width = drum_vis.shape[1]
        line_width = line_vis.shape[1]
        max_width = max(drum_width, line_width)

        drum_vis = pad_image_to_width(drum_vis, max_width)
        line_vis = pad_image_to_width(line_vis, max_width)

        side_view_visual = np.vstack((drum_vis, line_vis))
        side_view_visual = cv2.resize(side_view_visual, target_size)
        return side_view_visual

    def _place_sub_figures(self, seg_vis, defect_vis):
        h2 = int(self.frame_height * UPPER_SUB_FIGURE_HEIGHT_RATIO)
        w = self.frame_width
        pad = DEFAULT_PADDING
        half = SUB_FIGURE_WIDTH // 2

        self.canvas[:h2, w + pad : w + pad + half] = seg_vis
        self.canvas[:h2, w + pad * 2 + half : w + pad * 2 + half * 2] = defect_vis

    def _update_progress(self, prediction):
        rate = prediction.get("progress_rate", 0) * 100
        self.progress_history.append(rate)

        frame_idx = len(self.progress_history) - 1
        if prediction.get("is_ng", False):
            self.ng_frame_list.append(frame_idx)
        if prediction.get("require_sponge_wrapping", False):
            self.sponge_frame_list.append(frame_idx)
        if prediction["state_index"] == CommonSettings.state_index["shift_drum"]:
            self.drum_shift_frame_list.append(frame_idx)

    def _draw_body_lines(self, image, contour):
        x, y, w, h = cv2.boundingRect(contour)
        y_mid = y + h // 2
        for px in (x, x + w):
            image = cv2.line(image, (px, y_mid), (px, self.frame_height), DRUM_BODY_LINE_COLOR, 5)
            image = cv2.circle(image, (px, y_mid), 8, DRUM_BODY_LINE_COLOR, -1)
        return image

    def draw_dotted_box(self, image, box, color, thickness=8):
        x_left, y_top, w, h = box
        vertical_points = [x_left, x_left + w]
        for point in vertical_points:
            image = draw_dotted_line(
                image,
                (point, y_top),
                (point, y_top + h),
                color=color,
                thickness=thickness,
            )
        horizontal_points = [y_top, y_top + h]
        for point in horizontal_points:
            image = draw_dotted_line(
                image,
                (vertical_points[0], point),
                (vertical_points[1], point),
                color=color,
                thickness=thickness,
            )

        return image

    def put_status_text(self, image: np.ndarray, prediction: dict, shift_monitor: ShiftMonitor):
        def _draw_section(title, value, y_offset, value_formatter=str):
            """Draw a section with a title and value."""
            top_left = (
                DEFAULT_PADDING,
                DEFAULT_PADDING * y_offset + winding_status_height * (y_offset - 1),
            )
            bottom_right = (
                DEFAULT_PADDING + winding_status_width,
                DEFAULT_PADDING * y_offset + winding_status_height * y_offset,
            )
            cv2.rectangle(image, top_left, bottom_right, color=(10, 10, 10), thickness=-1)
            cv2.putText(
                image,
                title,
                (
                    DEFAULT_PADDING * 3,
                    int(DEFAULT_PADDING * (y_offset + 3)) + winding_status_height * (y_offset - 1),
                ),
                cv2.FONT_HERSHEY_TRIPLEX,
                2,
                (180, 180, 180),
                3,
            )
            cv2.putText(
                image,
                value_formatter(value),
                (
                    DEFAULT_PADDING * 3,
                    int(DEFAULT_PADDING * (y_offset + 7)) + winding_status_height * (y_offset - 1),
                ),
                cv2.FONT_HERSHEY_TRIPLEX,
                2,
                (255, 255, 255),
                3,
            )

        def _format_direction(direction):
            if direction == "left_to_right":
                return "Left -> Right"
            elif direction == "right_to_left":
                return "Right -> Left"
            else:
                return str(direction)

        def _format_float_value(value):
            return f"{value * 100:.2f}"

        def _format_angle(angle):
            return f"{angle:.2f} deg"

        # Dimensions for the status box
        winding_status_width = int(SUB_FIGURE_WIDTH * 0.6)
        winding_status_height = winding_status_width // 4

        # Section data
        sections = [
            ("Direction", prediction["cable_moving_direction"], 1, _format_direction),
            ("Progress rate", prediction["progress_rate"], 2, _format_float_value),
            ("Shifting score", prediction["drum_shifting_score"], 3, _format_float_value),
            ("Side shifting score", prediction["tightening_score"], 4, _format_float_value),
            ("Drum direction", prediction.get("drum_moving_direction", None), 5, _format_direction),
            ("Drum swing score", prediction["drum_swing_score"], 6, _format_float_value),
            ("Rotation angle", shift_monitor.cumulative_rotation_angle or 0.0, 7, _format_angle),
            ("Current State", shift_monitor.state.name, 8, str),
        ]

        # Draw all sections
        for title, value, y_offset, formatter in sections:
            _draw_section(title, value, y_offset, formatter)

    def initialize_canvas(self):
        self.canvas = np.ones(
            (
                self.frame_height + DEFAULT_PADDING,
                self.frame_width + SUB_FIGURE_WIDTH + DEFAULT_PADDING * 2,
                3,
            ),
            dtype=np.uint8,
        )
        self.canvas *= 255

    def _plot_winding_progress(self):
        ax = self._progress_ax
        canvas = self._progress_canvas
        line = self._progress_line

        data = self.progress_history
        N = len(data)
        x = list(range(N))

        # 1) Line 데이터 갱신
        line.set_data(x, data)

        # 2) 이전 fill_between 제거
        if self._progress_poly is not None:
            self._progress_poly.remove()
            self._progress_poly = None

        # 3) 새 fill_between 생성
        if N > 0:
            self._progress_poly = ax.fill_between(x, data, color="khaki", alpha=0.5)

        # 4) Y축 범위 설정
        if N == 0:
            y_min, y_max = 0, 1
        else:
            y_min, y_max = min(data), max(data)
        padding = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.set_ylim(-5, 105)

        # 5) X축 범위 & 눈금 업데이트
        ax.set_xlim(0, max(1, N - 1))
        step = max(1, N // 10)
        xticks = list(range(0, N, step))
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)

        # 6) 오류 발생한 프레임을 빨간색 vertical 선으로 표시
        for ng_frame_idx in self.ng_frame_list:
            ax.axvline(
                x=ng_frame_idx,
                color="red",
                linestyle="--",
                linewidth=1.5,
            )

        # 7) 스폰지 알람 발생 프레임을 초록색 vertical 선으로 표시
        for sponge_frame_idx in self.sponge_frame_list:
            ax.axvline(
                x=sponge_frame_idx,
                color="green",
                linestyle="--",
                linewidth=1.5,
            )

        # 8) 드럼 이동 알람 발생 프레임을 파란색 vertical 선으로 표시
        for drum_shift_frame_idx in self.drum_shift_frame_list:
            ax.axvline(
                x=drum_shift_frame_idx,
                color="blue",
                linestyle="--",
                linewidth=1.5,
            )

        canvas.draw()

        # 9) RGB 배열로 변환 후 리사이즈 & BGR 변환
        w, h = canvas.get_width_height()
        buf = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
        plot_img = buf.reshape((h, w, 3))
        plot_img = cv2.resize(
            plot_img, (SUB_FIGURE_WIDTH, int(self.frame_height * (1 - UPPER_SUB_FIGURE_HEIGHT_RATIO)))
        )
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)

        # 10) 메인 캔버스에 붙이기
        monitoring_x = self.frame_width + DEFAULT_PADDING
        monitoring_y = int(self.frame_height * UPPER_SUB_FIGURE_HEIGHT_RATIO) + DEFAULT_PADDING
        ph, pw = plot_img.shape[:2]
        self.canvas[monitoring_y : monitoring_y + ph, monitoring_x : monitoring_x + pw] = plot_img

    def _init_winding_figure(self):
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 0.4), dpi=DPI)
        canvas = FigureCanvas(fig)

        # 빈 Line2D 객체 생성
        (line,) = ax.plot([], [], color="goldenrod", linewidth=2, label="Progress")
        poly = None

        # static setup
        ax.set_title("Progress Over Time", fontsize=12)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Progress")
        ax.grid(True)
        ax.set_yticks(np.arange(0, 110, 10))

        self._progress_fig = fig
        self._progress_ax = ax
        self._progress_canvas = canvas
        self._progress_line = line
        self._progress_poly = poly


class SideViewResultVisualizer:
    def __init__(self, plot_graph: bool = False):
        self.visual_height = 2000
        self.visual_width = 1000

        self.plot_graph = plot_graph
        self.rotation_angle_history = []

        if plot_graph:
            self._init_rotation_figure()

    def __call__(
        self,
        image: np.ndarray,
        side_segmentation_results: SideSegmentationResults,
        line_tracker=None,
        rotation_angle: float = 0.0,
    ):
        original = image.copy()

        drum_vis = self.visualize_side_drum_prediction(original, side_segmentation_results)

        # Line detection 시각화 (라인 ID + 각도 표시)
        if line_tracker is None:
            line_vis = np.zeros_like(drum_vis)
        else:
            line_vis = self.visualize_line_detection(
                image=original,
                line_tracker=line_tracker,
                side_drum_box=side_segmentation_results.side_drum_box,
                draw_angles=True,
            )

        # Pad the images to have the same width by adding black borders to the narrower one
        drum_width = drum_vis.shape[1]
        line_width = line_vis.shape[1]
        max_width = max(drum_width, line_width)

        drum_vis = pad_image_to_width(drum_vis, max_width)
        line_vis = pad_image_to_width(line_vis, max_width)

        # Update rotation angle history and plot
        if self.plot_graph:
            self.rotation_angle_history.append(rotation_angle)
            rotation_plot = self._plot_rotation_angle()

            # Resize rotation plot to match width
            plot_height = rotation_plot.shape[0]
            rotation_plot = cv2.resize(rotation_plot, (max_width, plot_height))

            # 위 아래로 붙이기 (rotation angle plot 추가)
            vis = np.vstack((drum_vis, line_vis, rotation_plot))

        else:
            vis = np.vstack((drum_vis, line_vis))

        # visualize text - rotation angle(float)
        cv2.putText(
            vis,
            f"Rotation Angle: {rotation_angle:.2f} deg",
            (70, 70),
            cv2.FONT_HERSHEY_TRIPLEX,
            2.5,
            (0, 0, 0),
            3,
        )

        # XXX: 영상으로 만들기 위해서, 고정된 크기로 리사이즈
        vis = cv2.resize(vis, (self.visual_width, self.visual_height))
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

        return vis

    @staticmethod
    def visualize_side_drum_prediction(
        image: np.ndarray, side_segmentation_results: SideSegmentationResults
    ):
        side_drum_box = side_segmentation_results.side_drum_box
        if side_drum_box is None or image is None:
            return image

        vis = image.copy()
        vis = overlay_contours(
            vis, side_segmentation_results.side_drum_contour, color=(49, 148, 248), alpha=0.6
        )
        # draw bbox
        x, y, w, h = side_drum_box
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)

        vis = SideViewResultVisualizer.crop_with_margin(vis, side_drum_box, margin=50)
        return vis

    @staticmethod
    def visualize_line_detection(
        image: np.ndarray,
        line_tracker: MultiLineTracker,
        side_drum_box: tuple | None = None,
        draw_angles: bool = False,
        font_scale: float = 2.0,
        thickness: int = 3,
    ):
        """Line detection 결과를 시각화

        Args:
            image: 원본 이미지
            line_tracker: MultiLineTracker 인스턴스
            side_drum_box: 드럼 박스 (x, y, w, h) - cropping에 사용
            draw_angles: 각도 정보 표시 여부

        Returns:
            시각화된 이미지
        """
        if line_tracker is None or image is None:
            return np.zeros((600, 600, 3), dtype=np.uint8)

        # Create visualization on original image
        vis = image.copy()

        # Get tracked lines information
        lines_info = line_tracker.get_tracked_lines_info()

        # Draw drum center
        if line_tracker.center is not None:
            center_x, center_y = line_tracker.center
            cv2.circle(vis, (int(center_x), int(center_y)), 10, (255, 0, 0), -1)
            cv2.circle(vis, (int(center_x), int(center_y)), 12, (255, 255, 255), 2)

        # Draw detected lines
        for idx, info in enumerate(lines_info):
            if info["line_coords"]:
                x1, y1, x2, y2 = info["line_coords"]
                line_id = info["line_id"]
                center = info["center"]

                # Draw line
                cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

                # Draw line center point
                cv2.circle(vis, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)

                # Draw line ID with background
                text = f"#{line_id}"

                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )

                # Position text slightly offset from center
                text_x = int(center[0]) + 10
                text_y = int(center[1]) - 10

                # Draw background rectangle for text
                padding = 3
                cv2.rectangle(
                    vis,
                    (text_x - padding, text_y - text_height - padding),
                    (text_x + text_width + padding, text_y + baseline + padding),
                    (0, 0, 0),
                    -1,
                )

                # Draw text
                cv2.putText(
                    vis,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 0),  # Yellow color for visibility
                    thickness,
                )

                # Draw angle text if requested (below the line ID)
                if draw_angles and info["angle"] is not None:
                    angle = info["angle"]
                    angle_text_pos = (int(center[0]) + 30, int(center[1]) + 25)
                    cv2.putText(
                        vis,
                        f"{angle:.1f}",
                        angle_text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 0),
                        2,
                    )

        # Crop to drum box first if provided
        if side_drum_box is not None:
            vis = SideViewResultVisualizer.crop_with_margin(vis, side_drum_box, margin=50)

        # Draw info text
        num_lines = line_tracker.num_lines
        rotation = line_tracker.get_result()
        direction = line_tracker.get_rotation_direction_str()

        info_texts = [
            f"Lines: {num_lines}",
            f"Rotation: {rotation:.2f}deg",
            f"Direction: {direction}",
        ]

        if len(line_tracker.average_frame_angle_changes) > 0:
            info_texts.append(f"Change: {line_tracker.average_frame_angle_changes[-1]:+.2f}deg")

        # Draw text with background
        y_offset = 80
        padding = 5
        for text in info_texts:
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            # Draw filled background rectangle
            cv2.rectangle(
                vis,
                (padding, y_offset - text_height - padding),
                (padding * 2 + text_width, y_offset + baseline + padding),
                (0, 0, 0),
                -1,
            )
            # Draw border rectangle
            cv2.rectangle(
                vis,
                (padding, y_offset - text_height - padding),
                (padding * 2 + text_width, y_offset + baseline + padding),
                (255, 255, 255),
                2,
            )
            # Draw text
            cv2.putText(
                vis,
                text,
                (padding * 2, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
            )
            y_offset += text_height + baseline + padding * 4

        return vis

    @staticmethod
    def crop(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = roi
        return image[y : y + h, x : x + w]

    @staticmethod
    def crop_with_margin(
        array: np.ndarray, roi: tuple[int, int, int, int], margin: int = 50
    ) -> np.ndarray:
        x, y, w, h = roi
        x_start = max(x - margin, 0)
        y_start = max(y - margin, 0)
        x_end = min(x + w + margin, array.shape[1])
        y_end = min(y + h + margin, array.shape[0])
        return array[y_start:y_end, x_start:x_end]

    def _init_rotation_figure(self):
        """Initialize matplotlib figure for rotation angle plot."""
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 0.4), dpi=DPI)
        canvas = FigureCanvas(fig)

        # 빈 Line2D 객체 생성
        (line,) = ax.plot([], [], color="steelblue", linewidth=2, label="Rotation Angle")
        poly = None

        # static setup
        ax.set_title("Rotation Angle Over Time", fontsize=12)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Angle (degrees)")
        ax.grid(True)

        self._rotation_fig = fig
        self._rotation_ax = ax
        self._rotation_canvas = canvas
        self._rotation_line = line
        self._rotation_poly = poly

    def _plot_rotation_angle(self) -> np.ndarray:
        """Plot rotation angle history and return as image."""
        ax = self._rotation_ax
        canvas = self._rotation_canvas
        line = self._rotation_line

        data = self.rotation_angle_history
        N = len(data)
        x = list(range(N))

        # 1) Line 데이터 갱신
        line.set_data(x, data)

        # 2) 이전 fill_between 제거
        if self._rotation_poly is not None:
            self._rotation_poly.remove()
            self._rotation_poly = None

        # 3) 새 fill_between 생성
        if N > 0:
            self._rotation_poly = ax.fill_between(x, data, color="lightblue", alpha=0.5)

        # 4) Y축 범위 설정
        if N == 0:
            y_min, y_max = -10, 10
        else:
            y_min, y_max = min(data), max(data)
        padding = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
        ax.set_ylim(y_min - padding, y_max + padding)

        # 5) X축 범위 & 눈금 업데이트
        ax.set_xlim(0, max(1, N - 1))
        step = max(1, N // 10)
        xticks = list(range(0, N, step))
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)

        canvas.draw()

        # 6) RGB 배열로 변환 후 BGR 변환
        w, h = canvas.get_width_height()
        buf = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
        plot_img = buf.reshape((h, w, 3))
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)

        return plot_img

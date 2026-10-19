import numpy as np

from .direction import Direction
from .shift_state import BaseState, FirstSwingState, SwingTermState, SecondSwingState, TighteningState
from ..data.data_class import SegmentationResults
from ..utils.geometry import find_winding_top_x, find_top_edge_points
from ..utils.smoothing import calculate_smooth_slopes
from ..utils.roi import extract_top_roi, get_roi_side


class SwingScoreCalculator:
    def __init__(
        self,
        swing_range_factors: list[float],
        default_swing_score: float,
        cable_width_percentile: float = 5.0,
        swing_skip_threshold_factor: float = 3.0,
        side_top_percentage: float = 0.3,
        roi_margin: int = 15,
        flat_slope_threshold: float = 0.1,
    ):
        self.default_swing_score = default_swing_score
        self.swing_range_factors = swing_range_factors
        self._cable_width_history: list[int] = []
        self.cable_width_percentile = cable_width_percentile
        self.swing_skip_threshold_factor = swing_skip_threshold_factor
        self.side_top_percentage = side_top_percentage
        self.roi_margin = roi_margin
        self.flat_slope_threshold = flat_slope_threshold

        # Cumulative average tracking for auto swing skip
        self._cumulative_top_x_sum: float = 0.0
        self._cumulative_top_x_count: int = 0
        self._frozen_avg_top_x: float | None = None
        self._is_avg_frozen: bool = False

        self.reset()

    def calculate_score_and_excess_travel(
        self,
        winding_cable_mask: np.ndarray,
        drum_body_box: list[int],
        cable_moving_direction: Direction,
        state: BaseState,
        bounding_boxes: dict[str, list[int]] = None,
        segmentation_masks: dict[str, np.ndarray] = None,
    ) -> tuple[float, float]:
        if not isinstance(state, (FirstSwingState, SwingTermState, SecondSwingState)):
            return self.default_swing_score, 0.0

        winding_x = self._get_current_cable_x(winding_cable_mask, cable_moving_direction)

        # 권취 중인 전선이 첫번째 바퀴 옆으로 넘어가기 위한 swing range 설정.
        roi_side = get_roi_side(state.name, cable_moving_direction)

        # Swing range는 현재 프레임의 drum body 끝점 기준으로 매 프레임 다시 계산한다.
        self.set_swing_range(
            winding_cable_mask,
            drum_body_box,
            roi_side,
            bounding_boxes,
            segmentation_masks,
            cable_moving_direction,
        )
        if self._swing_range is None:
            return self.default_swing_score, 0.0

        start_x, target_x = self._swing_range

        # Current_x는 방향에 따른 winding cable top edge를 사용한다.
        current_x = winding_x

        # Swing score 계산: current_x가 target_x에 가까워질수록 0으로 감소
        if cable_moving_direction.is_left_to_right():
            # 왼쪽에서 오른쪽: current_x가 start_x에서 target_x(오른쪽)으로 이동
            total_distance = abs(target_x - start_x)
            traveled = max(0, current_x - start_x)
        elif cable_moving_direction.is_right_to_left():
            # 오른쪽에서 왼쪽: current_x가 start_x에서 target_x(왼쪽)으로 이동
            total_distance = abs(start_x - target_x)
            traveled = max(0, start_x - current_x)
        else:
            assert False, "Invalid cable moving direction"

        if total_distance <= 0:
            return 0.0, 0.0

        # Score: 목적지에 가까워질수록 0으로 감소 (1 - traveled/total_distance)
        score = 1.0 - min(traveled, total_distance) / total_distance

        # excess_travel: current_x가 target_x를 넘은 거리
        excess_travel = current_x - target_x

        # XXX: 부호 변환
        if cable_moving_direction == Direction.RIGHT_TO_LEFT:
            # 오른쪽 극단에서는 목표 지점 대비 왼쪽으로 넘었는지를 보기 때문.
            excess_travel = -excess_travel

        return score, excess_travel

    def set_swing_range(
        self,
        winding_cable_mask: np.ndarray,
        drum_body_box: list[int],
        roi_side: str,
        bounding_boxes: dict[str, list[int]] = None,
        segmentation_masks: dict[str, np.ndarray] = None,
        cable_moving_direction: Direction = None,
    ) -> None:
        """
        Set the horizontal swing range for swing scoring.
        start_x: 현재 프레임의 drum body 하단 2% 영역에서 방향에 따라 선택한 끝점
            - RIGHT_TO_LEFT: x max
            - LEFT_TO_RIGHT: x min
        target_x: 목표 지점 (start_x ± cable_width * factor)

        Args:
            winding_cable_mask (np.ndarray): Binary mask of the cable wrap (H, W).
            drum_body_box (list[int]): Bounding box of the drum.
            roi_side (str): Region of interest side ("left" or "right").
            bounding_boxes (dict[str, list[int]]): Bounding boxes for all segmented objects.
            segmentation_masks (dict[str, np.ndarray]): Segmentation masks for all objects.
            cable_moving_direction (Direction): Direction of cable movement.

        Returns:
            list[int]: [start_x, target_x] - 시작 위치와 목표 위치
        """
        # Winding cable mask의 상위 {self.cable_width_percentile}% Y 좌표를 기준으로 전선의 두께 추정
        cable_width = self.calculate_cable_width_from_mask(winding_cable_mask, use_history=True)
        if cable_width is None:
            return

        # start_x는 기본적으로 현재 방향에 맞는 winding cable edge를 사용하되,
        # drum body 하단 2% 끝점을 얻을 수 있으면 그 값을 우선한다.
        start_x = self._get_current_cable_x(winding_cable_mask, cable_moving_direction)

        if (
            bounding_boxes is not None
            and segmentation_masks is not None
            and cable_moving_direction is not None
        ):
            drum_body_start_x = self._get_drum_body_start_x(segmentation_masks, cable_moving_direction)
            if drum_body_start_x is not None:
                start_x = drum_body_start_x

            if cable_moving_direction.is_right_to_left():
                # RIGHT_TO_LEFT: start_x(오른쪽) 기준으로 왼쪽으로 이동
                swing_range_factor = self.swing_range_factors[1]
                target_x = round(start_x - cable_width * swing_range_factor)
            else:
                # LEFT_TO_RIGHT: start_x(왼쪽) 기준으로 오른쪽으로 이동
                swing_range_factor = self.swing_range_factors[0]
                target_x = round(start_x + cable_width * swing_range_factor)
        else:
            # Fallback: drum_body_box 기준 계산 (기존 방식)
            if roi_side == "left":
                swing_range_factor = self.swing_range_factors[0] + 1
                target_x = round(start_x + cable_width * swing_range_factor)
            else:
                swing_range_factor = self.swing_range_factors[1] + 1
                target_x = round(start_x - cable_width * swing_range_factor)

        self._swing_range = [start_x, target_x]

    @property
    def swing_range(self):
        return self._swing_range

    def reset(self):
        self._swing_range: list[int] | None = None  # 두번째 바퀴 이동 범위 [시작 x, 목표 x]
        self._cable_width_history.clear()

        # Reset cumulative average tracking
        self._cumulative_top_x_sum = 0.0
        self._cumulative_top_x_count = 0
        self._frozen_avg_top_x = None
        self._is_avg_frozen = False

    def get_cable_width(self) -> None | int:
        if len(self._cable_width_history) == 0:
            return None
        return int(np.median(self._cable_width_history).item())

    def calculate_cable_width_from_mask(
        self, winding_cable_mask: np.ndarray, use_history: bool = True
    ) -> int | None:
        """
        Winding cable mask의 상위 {self.cable_width_percentile}% Y 좌표를 기준으로 전선의 두께 계산.

        Args:
            winding_cable_mask: Winding cable의 마스크
            use_history: 히스토리를 사용하여 median 값을 반환할지 여부

        Returns:
            전선 두께 (픽셀)
        """
        ys, xs = np.where(winding_cable_mask == 1)
        if len(ys) == 0:
            return None

        cutoff = np.percentile(ys, self.cable_width_percentile)
        top_xs = xs[ys <= cutoff]

        if len(top_xs) == 0:
            return None

        cable_width = int(top_xs.max() - top_xs.min())

        if use_history:
            self._cable_width_history.append(cable_width)
            return int(np.median(self._cable_width_history).item())

        return cable_width

    def update_cumulative_avg_top_x(self, winding_cable_mask: np.ndarray) -> None:
        """
        Update cumulative average of winding cable top x position.
        This is used for auto swing skip logic.

        Args:
            winding_cable_mask: Winding cable의 마스크
        """
        if self._is_avg_frozen:
            return

        top_x = find_winding_top_x(winding_cable_mask)
        if top_x is not None and top_x > 0:
            self._cumulative_top_x_sum += top_x
            self._cumulative_top_x_count += 1

    def freeze_cumulative_avg(self) -> None:
        """
        Freeze the cumulative average at current value.
        Called when entering waiting_for_first_swing state.
        """
        if self._cumulative_top_x_count > 0 and not self._is_avg_frozen:
            self._frozen_avg_top_x = self._cumulative_top_x_sum / self._cumulative_top_x_count
            self._is_avg_frozen = True

    def get_frozen_avg_top_x(self) -> float | None:
        """
        Get the frozen cumulative average top x.

        Returns:
            Frozen average top x, or None if not frozen yet
        """
        return self._frozen_avg_top_x

    def get_current_avg_top_x(self) -> float | None:
        """
        Get the current cumulative average (not frozen).

        Returns:
            Current average or None if no data
        """
        if self._cumulative_top_x_count > 0:
            return self._cumulative_top_x_sum / self._cumulative_top_x_count
        return None

    def get_cable_top_x_coordinates(
        self, winding_cable_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Winding cable mask의 상위 {self.cable_width_percentile}% Y 좌표 영역의 x, y 좌표를 반환.

        Args:
            winding_cable_mask: Winding cable의 마스크

        Returns:
            (top_xs, top_ys): 상위 percentile 영역의 x, y 좌표 배열
        """
        ys, xs = np.where(winding_cable_mask == 1)
        if len(ys) == 0:
            return None

        cutoff = np.percentile(ys, self.cable_width_percentile)
        top_mask = ys <= cutoff
        top_xs = xs[top_mask]
        top_ys = ys[top_mask]

        if len(top_xs) == 0:
            return None

        return top_xs, top_ys

    def _get_drum_body_start_x(
        self,
        masks: dict[str, np.ndarray],
        direction: Direction,
    ) -> int | None:
        """Return the direction-aware endpoint of the bottom 2% drum body mask."""

        drum_body_mask = masks.get("drum_body")
        if drum_body_mask is None:
            return None

        ys, xs = np.where(drum_body_mask > 0)
        if len(ys) == 0:
            return None

        # 숫자가 큰 y 좌표, 즉 화면 하단 2% 영역만 사용한다.
        cutoff = np.percentile(ys, 98)
        bottom_mask = ys >= cutoff
        bottom_xs = xs[bottom_mask]
        if len(bottom_xs) == 0:
            return None

        if direction.is_right_to_left():
            return int(np.max(bottom_xs))
        return int(np.min(bottom_xs))

    def _get_current_cable_x(
        self,
        winding_cable_mask: np.ndarray,
        direction: Direction | None,
    ) -> int:
        """Return the direction-aware top edge x of the winding cable."""
        if direction is None:
            return find_winding_top_x(winding_cable_mask)
        edge = "max" if direction.is_right_to_left() else "min"
        return find_winding_top_x(winding_cable_mask, percentage=1.0, edge=edge)


class ShiftScoreCalculator:
    def __init__(
        self,
        default_shifting_score: float,
        shifting_box_ratio: float,
        right_over_left_expansion_ratio: float,
    ):
        self.default_shifting_score = default_shifting_score
        self._shifting_box_ratio = shifting_box_ratio
        self._right_over_left_expansion_ratio = right_over_left_expansion_ratio
        self.reset()

    def calculate_score(self, winding_cable_box: list[int], cable_moving_direction: Direction) -> float:
        """좌/우로 얼마나 벗어났는지 비율로 계산합니다."""
        if (
            self._shifting_box_in is None
            or self._shifting_box_out is None
            or cable_moving_direction is None
        ):
            return self.default_shifting_score  # 드럼 이동 유효 영역이 설정되지 않은 경우

        box_x, _, box_width, _ = winding_cable_box

        shifting_score = self.default_shifting_score
        if cable_moving_direction.is_right_to_left():
            total_length = self._shifting_box_in[0] - self._shifting_box_out[0]
            out_length = max(0, self._shifting_box_in[0] - box_x)
            shifting_score = min(out_length, total_length) / total_length if total_length > 0 else 0.0

        elif cable_moving_direction.is_left_to_right():
            point_in = self._shifting_box_in[0] + self._shifting_box_in[2]
            point_out = self._shifting_box_out[0] + self._shifting_box_out[2]
            total_length = point_out - point_in
            out_length = max(0, box_x + box_width - point_in)
            shifting_score = min(out_length, total_length) / total_length if total_length > 0 else 0.0
        return shifting_score

    def set_shifting_box(self, box: list[int]) -> None:
        """
        드럼 이동 유효 영역 박스를 설정합니다.
        권취 중인 전선이 왼쪽으로의 움직임 보다 오른쪽 움직임이 더 크기 때문에 shifting_box_ratio를 왼쪽 보다 오른쪽으로 더 확장시킵니다.
        """
        if box is not None:
            self._shifting_box_in = [
                box[0],
                box[1],
                box[2],
                box[3],
            ]

            # 남는 여유 폭을 좌:우 = 1:ratio 로 배분하여 왼쪽 시작점을 이동
            total_margin = (self._shifting_box_ratio - 1) * box[2]
            left_margin = total_margin / (self._right_over_left_expansion_ratio + 1)
            out_x = int(box[0] - left_margin)  # 오른쪽으로 더 넓게 확장되도록 왼쪽 여유 폭을 줄임
            out_w = int(box[2] * self._shifting_box_ratio)
            self._shifting_box_out = [
                out_x,
                box[1],
                out_w,
                box[3],
            ]

    @property
    def shifting_box_ratio(self):
        return self._shifting_box_ratio

    @shifting_box_ratio.setter
    def shifting_box_ratio(self, value: float):
        self._shifting_box_ratio = value

    @property
    def shifting_box_in(self):
        return self._shifting_box_in

    @property
    def shifting_box_out(self):
        return self._shifting_box_out

    def reset(self):
        """드럼 이동 유효 영역 박스와 이탈 영역 박스를 초기화합니다."""
        self._shifting_box_in: list[int] = None  # 드럼 이동 유효 영역 박스 (권취 전선의 타이트 박스)
        self._shifting_box_out: list[int] = None  # 드럼 이동 이탈 영역 박스 (ratio로 확장된 박스)


class TighteningScoreCalculator:
    def __init__(
        self,
        default_tightening_score: float,
        tightening_range_factor: float = 1.0,
        side_top_percentage: float = 0.3,
        roi_margin: int = 15,
        flat_slope_threshold: float = 0.1,
    ):
        self.default_tightening_score = default_tightening_score
        self.tightening_range_factor = tightening_range_factor
        self.side_top_percentage = side_top_percentage
        self.roi_margin = roi_margin
        self.flat_slope_threshold = flat_slope_threshold
        self.reset()

    def calculate_score(
        self,
        segmentation_results: SegmentationResults,
        cable_moving_direction: Direction,
        state: BaseState,
    ):
        """Return the normalized side-drum shift score, or reset as needed."""

        if not isinstance(state, TighteningState):
            return self.default_tightening_score

        winding_cable_mask = segmentation_results.winding_cable_mask
        if self._side_drum_shift_x_range is None:
            original_target = find_winding_top_x(winding_cable_mask)
            start_drum_x = self._get_side_drum_current_x(
                segmentation_results.bounding_boxes,
                segmentation_results.segmentation_masks,
                cable_moving_direction,
            )

            # tightening_range_factor를 적용하여 목표 지점 조정
            adjusted_target = self._calculate_adjusted_target(
                original_target,
                start_drum_x,
                cable_moving_direction,
            )

            self._side_drum_shift_x_range = [adjusted_target, start_drum_x]

        # Compute how far we’ve moved from that initial top.
        start_x, init_drum_x = self._side_drum_shift_x_range
        current_x = self._get_side_drum_current_x(
            segmentation_results.bounding_boxes,
            segmentation_results.segmentation_masks,
            cable_moving_direction,
        )

        if cable_moving_direction.is_right_to_left():
            total_distance = start_x - init_drum_x
            traveled = start_x - current_x
        elif cable_moving_direction.is_left_to_right():
            total_distance = init_drum_x - start_x
            traveled = current_x - start_x
        else:
            assert False, "Invalid cable moving direction"

        if total_distance <= 0:
            return 0.0

        return min(max(traveled, 0), total_distance) / total_distance

    def _get_side_drum_current_x(
        self,
        boxes: dict[str, list[int]],
        masks: dict[str, np.ndarray],
        direction: Direction,
    ) -> int:
        """Return the current X coordinate of the moving drum edge."""

        def _get_side_drum_key(direction: Direction) -> str:
            return "right_drum" if direction.is_left_to_right() else "left_drum"

        side_drum_key = _get_side_drum_key(direction)
        side_drum_box = boxes[side_drum_key]
        side_drum_mask = masks[side_drum_key].copy()

        top_roi, roi_x = extract_top_roi(
            side_drum_box, side_drum_mask, self.side_top_percentage, self.roi_margin
        )
        top_edge_points = find_top_edge_points(top_roi, roi_x)
        x_vals, slopes = calculate_smooth_slopes(top_edge_points)

        zero_slope_indices = np.where(
            (slopes > -self.flat_slope_threshold) & (slopes < self.flat_slope_threshold)
        )[0]
        gradient_zero_x_points = x_vals[zero_slope_indices]

        try:
            if direction.is_left_to_right():
                output = int(np.max(gradient_zero_x_points))
            else:
                output = int(np.min(gradient_zero_x_points))
        except:
            output = int(np.mean(x_vals))

        return output

    def _calculate_adjusted_target(
        self,
        original_target: int,
        start_x: int,
        direction: Direction,
    ) -> int:
        """tightening_range_factor를 적용하여 목표 지점을 조정합니다.

        Args:
            original_target: 기존 목표 지점 (winding cable top x)
            start_x: 시작 지점 (side drum x)
            direction: 케이블 이동 방향

        Returns:
            조정된 목표 지점

        Example:
            LEFT_TO_RIGHT (roi_side=right): original_target=100, start_x=110, factor=1.1
            -> new_target = 110 - (110 - 100) * 1.1 = 99
        """
        # factor가 1.0이면 원본값 그대로 반환
        if self.tightening_range_factor == 1.0:
            return original_target

        distance = abs(start_x - original_target)
        adjusted_distance = distance * self.tightening_range_factor

        if direction.is_left_to_right():
            # LEFT_TO_RIGHT + roi_side=right: target이 작아짐 (더 왼쪽으로)
            new_target = start_x - adjusted_distance
        else:  # RIGHT_TO_LEFT
            # RIGHT_TO_LEFT + roi_side=left: target이 커짐 (더 오른쪽으로)
            new_target = start_x + adjusted_distance

        return int(round(new_target))

    @property
    def side_drum_shift_x_range(self):
        return self._side_drum_shift_x_range

    def reset(self):
        self._side_drum_shift_x_range: list[int] = None  # 극단 드럼 이동 x 좌표 범위 [목표 x, 시작 x]

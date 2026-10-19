class IntervalTimer:
    """주어진 간격(interval)마다 True를 반환하는 타이머 클래스입니다.

    - tick(): 프레임마다 호출. 내부 카운터를 1씩 증가시키고, interval에 도달하면 True를 반환하며 0으로 리셋합니다.
    - set_interval(v): interval 변경.
    - reset(): 카운터를 0으로 초기화합니다.
    """

    def __init__(self, interval: int, prime: bool = False):
        assert isinstance(interval, int) and interval > 0, "Interval must be greater than 0"
        self._interval = interval
        # prime=True이면 첫 tick() 호출에서 바로 True가 되도록 interval-1에서 시작
        self._count = (self._interval - 1) if prime else 0

    @property
    def interval(self) -> int:
        return self._interval

    def set_interval(self, interval: int) -> None:
        assert isinstance(interval, int) and interval > 0, "Interval must be greater than 0"
        self._interval = interval
        self.reset()

    def tick(self) -> bool:
        self._count += 1
        if self._count >= self._interval:
            self._count = 0
            return True
        return False

    def reset(self) -> None:
        self._count = 0

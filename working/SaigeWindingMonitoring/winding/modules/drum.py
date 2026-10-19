class Drum:
    def __init__(self):
        # ----------                                                ----------  #
        # |        |                                                |        |  #
        # |        |          ----------------------                |        |  #
        # |        |          |                     \---            |        |  #
        # |        |          |                        |            |        |  #
        # |   head_right   body_left  body_center  body_right  tail_left     |  #
        # |        |          |                        |            |        |  #
        # |        |          |                     /---            |        |  #
        # |        |          ----------------------                |        |  #
        # |        |                                                |        |  #
        # ----------                                                ----------  #

        self.body_center_point = None
        self.body_left_point = None
        self.body_right_point = None
        self.head_right_point = None
        self.tail_left_point = None

    def set_body_center_point(self, center_point: list[int]):
        self.body_center_point = center_point

    def set_body_left_point(self, left_point: list[int]):
        self.body_left_point = left_point

    def set_body_right_point(self, right_point: list[int]):
        self.body_right_point = right_point

    def set_head_right_point(self, center_point: list[int]):
        self.head_right_point = center_point

    def set_tail_left_point(self, center_point: list[int]):
        self.tail_left_point = center_point

    def __str__(self):
        return (
            "Drum (x, y):\n"
            f"  ├─ Body center : {self.body_center_point}\n"
            f"  ├─ Body left   : {self.body_left_point}\n"
            f"  ├─ Body right  : {self.body_right_point}\n"
            f"  ├─ Head center : {self.head_right_point}\n"
            f"  └─ Tail center : {self.tail_left_point}"
        )

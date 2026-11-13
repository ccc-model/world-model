"""pytorch3d 的替身：Duolando 只 import 了 matrix_to_axis_angle，實際用到的程式碼已被註解掉。
pytorch3d 在 macOS 上難以安裝，所以用這個空殼讓 import 成功。"""


def matrix_to_axis_angle(*args, **kwargs):
    raise NotImplementedError("pytorch3d shim: matrix_to_axis_angle is not available")

import numpy as np

from .quantized_numeric_type import QuantizedNumericType


class QuantizedInteger(QuantizedNumericType):
    """经过量化的整数类型，以二进制补码表示。

    参数:
        bit_width: 位宽，包括符号位在内。
        signed: 是否有符号。
    """
    @staticmethod
    def get_numeric_type_class_id() -> str:
        return "int"

    def __init__(self, bit_width: int, signed: bool):
        super().__init__()
        bit_width_ = int(bit_width)
        signed_ = bool(signed)
        if bit_width_ <= 0:
            raise ValueError("bit_width must be a positive integer.")
        self._add_parameter("bit_width", bit_width_)
        self._add_parameter("signed", signed_)

    def get_bit_width(self) -> int:
        """获取位宽。

        返回值:
            位宽，包括符号位在内。
        """
        return self.get_parameter()["bit_width"]

    def is_signed(self) -> bool:
        """判断是否有符号。

        返回值:
            是否有符号。
        """
        return self.get_parameter()["signed"]

    def quantize(self, value: np.ndarray) -> np.ndarray:
        """量化给定数值。

        参数:
            value: 数值。

        返回值:
            经过量化的数值。

        提示:
            本方法返回数组的每个元素是 `value` 对应元素的量化结果。
        """
        value_ = np.asarray(value)
        parameter = self.get_parameter()
        bit_width = parameter["bit_width"]
        BIT_WIDTH_NUMPY_INTEGER_MAX = 64
        BIT_WIDTH_BYTE = 8
        # 计算返回的NumPy数组元素的位宽numpy_integer_bit_width。
        bit_width_bit_length = int.bit_length(bit_width)
        bit_width_low_bits = bit_width & \
            ((1 << (bit_width_bit_length - 1)) - 1)
        numpy_integer_bit_width = max(
            1 << (bit_width_bit_length + int(bool(bit_width_low_bits)) - 1),
            BIT_WIDTH_BYTE)
        unsigned_type = getattr(np, "uint" + str(numpy_integer_bit_width)) \
            if bit_width <= BIT_WIDTH_NUMPY_INTEGER_MAX \
            else object
        if not np.issubsctype(value_.dtype, np.integer):
            value_ = self.__convert_to_int(value_)
            # 若bit_width足够小，能用NumPy的整数类型表示，则截取低位，避免数值超出NumPy整
            # 数类型的表示范围。
            value_ &= (1 << bit_width) - 1
        # 若bit_width足够小，能用NumPy的整数类型表示，则将输入数值转换为无符号整数。
        value_ = value_.astype(unsigned_type)
        if parameter["signed"]:
            # sign_mask必须是无符号整数，否则可能无法表示(1 << (bit_width - 1))的值。
            sign_mask = np.array(1 << (bit_width - 1), dtype=unsigned_type)
            low_mask = np.array(~(-1 << (bit_width - 1)), dtype=unsigned_type)
            signed_type = getattr(np, "int" + str(numpy_integer_bit_width)) \
                if bit_width <= BIT_WIDTH_NUMPY_INTEGER_MAX \
                else object
            return ((value_ & low_mask) | -(value_ & sign_mask)).astype(
                signed_type)
        else:
            mask = np.array((1 << bit_width) - 1, dtype=unsigned_type)
            return value_ & mask

    def __eq__(self, other: QuantizedNumericType) -> bool:
        """判断本实例与给定实例是否相等。

        参数:
            other: 给定实例。

        返回值:
            本实例与给定实例是否相等。
        """
        if not isinstance(other, QuantizedInteger):
            return False
        parameter = self.get_parameter()
        parameter_other = other.get_parameter()
        return parameter["bit_width"] == parameter_other["bit_width"] and \
            parameter["signed"] == parameter_other["signed"]

    def __hash__(self) -> int:
        """计算本实例的哈希值。

        返回值:
            本实例的哈希值。
        """
        parameter = self.get_parameter()
        return hash(QuantizedInteger) ^ hash(parameter["bit_width"]) ^ \
            hash(parameter["signed"])

    __convert_to_int = np.frompyfunc(int, 1, 1)


QuantizedNumericType.register(QuantizedInteger)

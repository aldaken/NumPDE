"""
LLDB Eigen Pretty Printer
Source: https://github.com/fantaosha/LLDB-Eigen-Pretty-Printer
Supports: Matrix, Array, SparseMatrix, Quaternion
"""
import lldb
import re
from functools import partial


def __lldb_init_module(debugger, dict):
    for type_name, func_name in [
        ("Matrix", "eigen_matrix_print"),
        ("Array", "eigen_array_print"),
        ("Quaternion", "eigen_quaternion_print"),
        ("SparseMatrix", "eigen_sparsematrix_print"),
    ]:
        debugger.HandleCommand(
            'type summary add -x "^Eigen::%s<.*?>$" '
            "-F lldb_eigen_printer.%s -p -r -w Eigen" % (type_name, func_name)
        )
    debugger.HandleCommand("type category enable Eigen")


def evaluate_at_index(valobj, index):
    return valobj.GetValueForExpressionPath("[" + str(index) + "]")


class Printer:
    def __init__(self, data):
        self.data = data

    def evaluate_real(self, index):
        return "%1.8e" % float(evaluate_at_index(self.data, index).GetValue())

    def evaluate_bool(self, index):
        return "%d" % evaluate_at_index(self.data, index).GetValueAsUnsigned()

    def evaluate_complex_double(self, index):
        val = list(
            self.data.GetValueForExpressionPath(
                "[" + str(index) + "]._M_value"
            ).GetValue()
        )
        val[-1] = "j"
        for n in range(0, len(val)):
            if val[n] == " ":
                del val[n]
                del val[n + 1]
                if val[n + 1] == "-":
                    del val[n]
                break
        val = complex("".join(val))
        return "{0:1.5e} {1} {2:1.5e}i".format(
            val.real, "+-"[val.imag < 0], abs(val.imag)
        )

    def evaluate_complex_int(self, index):
        val = evaluate_at_index(self.data, index)
        real = val.GetValueForExpressionPath("._M_real").GetValueAsSigned()
        imag = val.GetValueForExpressionPath("._M_imag").GetValueAsSigned()
        val = real + imag * 1j
        return "{0:1.5e} {1} {2:1.5e}i".format(
            val.real, "+-"[val.imag < 0], abs(val.imag)
        )

    def evaluate_complex_bool(self, index):
        val = evaluate_at_index(self.data, index)
        real = val.GetValueForExpressionPath("._M_real").GetValueAsUnsigned()
        imag = val.GetValueForExpressionPath("._M_imag").GetValueAsUnsigned()
        return "{0:d} + {1:d}i".format(real, imag)


def _detect_getter(printer, data, begin, type_str):
    """Detect the right getter function based on scalar type."""
    complex_scalar = "std::complex<"
    is_complex = type_str.find(complex_scalar) >= 0
    is_bool = type_str.find(begin + ("std::complex<bool" if is_complex else "bool")) >= 0

    if is_complex:
        val = data.GetValueForExpressionPath("[0]")
        if val.GetValueForExpressionPath("._M_value").IsValid():
            return partial(Printer.evaluate_complex_double, printer)
        elif val.GetValueForExpressionPath("._M_real").IsValid():
            if is_bool:
                return partial(Printer.evaluate_complex_bool, printer)
            return partial(Printer.evaluate_complex_int, printer)
        return None
    elif is_bool:
        return partial(Printer.evaluate_bool, printer)
    return partial(Printer.evaluate_real, printer)


class Matrix(Printer):
    def __init__(self, variety, val):
        try:
            type_str = val.GetType().GetDirectBaseClassAtIndex(0).GetName()
            begin = "Eigen::" + variety + "<"
            if type_str.find("std::complex<") >= 0:
                regex = re.compile(begin + "std::complex<.*?>,.*?>")
            else:
                regex = re.compile(begin + ".*?>")

            self.variety = regex.findall(type_str)[0]
            m = self.variety[len(begin):-1]
            params = [x.replace(" ", "") for x in m.split(",")]

            self.rows = int(params[1])
            if self.rows == -1:
                self.rows = val.GetValueForExpressionPath(".m_storage.m_rows").GetValueAsSigned()

            self.cols = int(params[2])
            if self.cols == -1:
                self.cols = val.GetValueForExpressionPath(".m_storage.m_cols").GetValueAsSigned()

            self.options = int(params[3]) if len(params) > 3 else 0
            self.rowMajor = self.options & 0x1

            if int(params[1]) == -1 or int(params[2]) == -1:
                data = val.GetValueForExpressionPath(".m_storage.m_data")
            else:
                data = val.GetValueForExpressionPath(".m_storage.m_data.array")

            Printer.__init__(self, data)
            self.get = _detect_getter(self, data, begin, type_str)
            if self.get is None:
                self.variety = -1
        except:
            self.variety = -1

    def to_string(self):
        n = self.rows * self.cols
        padding = max(len(str(self.get(i))) for i in range(n)) if n else 1

        lines = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                idx = i * self.cols + j if self.rowMajor else i + j * self.rows
                row.append(self.get(idx).rjust(padding + (2 if j else 1)))
            lines.append("".join(row))

        return "rows: %d, cols: %d\n[%s ]\n" % (
            self.rows, self.cols,
            ";\n ".join(lines) if lines else "",
        )


class SparseMatrix(Printer):
    def __init__(self, val):
        try:
            type_str = val.GetType().GetDirectBaseClassAtIndex(0).GetName()
            begin = "Eigen::SparseMatrix<"
            if type_str.find("std::complex<") >= 0:
                regex = re.compile(begin + "std::complex<.*?>,.*?>")
            else:
                regex = re.compile(begin + ".*?>")

            self.variety = regex.findall(type_str)[0]
            m = self.variety[len(begin):-1]
            params = [x.replace(" ", "") for x in m.split(",")]

            self.options = int(params[1]) if len(params) > 1 else 0
            self.rowMajor = self.options & 0x1

            if self.rowMajor:
                self.rows = val.GetValueForExpressionPath(".m_outerSize").GetValueAsSigned()
                self.cols = val.GetValueForExpressionPath(".m_innerSize").GetValueAsSigned()
            else:
                self.rows = val.GetValueForExpressionPath(".m_innerSize").GetValueAsSigned()
                self.cols = val.GetValueForExpressionPath(".m_outerSize").GetValueAsSigned()

            self.outerStarts = val.GetValueForExpressionPath(".m_outerIndex")
            innerNNZs = val.GetValueForExpressionPath(".m_innerNonZeros")
            outer_count = self.rows if self.rowMajor else self.cols

            if innerNNZs.GetValueAsSigned():
                self.innerNNZs = [
                    evaluate_at_index(innerNNZs, k).GetValueAsSigned()
                    for k in range(outer_count)
                ]
            else:
                self.innerNNZs = [
                    evaluate_at_index(self.outerStarts, k + 1).GetValueAsSigned()
                    - evaluate_at_index(self.outerStarts, k).GetValueAsSigned()
                    for k in range(outer_count)
                ]

            self.nonzeros = sum(self.innerNNZs)
            self.size = val.GetValueForExpressionPath(".m_data.m_size").GetValueAsSigned()
            self.indices = val.GetValueForExpressionPath(".m_data.m_indices")

            data = val.GetValueForExpressionPath(".m_data.m_values")
            Printer.__init__(self, data)
            self.get = _detect_getter(self, data, begin, type_str)
            if self.get is None:
                self.variety = -1
        except:
            self.variety = -1

    def to_string(self):
        padding = max((len(str(self.get(i))) for i in range(self.size)), default=1)
        entries = []

        if self.rowMajor:
            for i in range(self.rows):
                idx = evaluate_at_index(self.outerStarts, i).GetValueAsSigned()
                for c in range(self.innerNNZs[i]):
                    j = evaluate_at_index(self.indices, idx + c).GetValueAsSigned()
                    entries.append("[%d, %d] =%s" % (i, j, self.get(idx + c).rjust(padding + 1)))
        else:
            raw = []
            for j in range(self.cols):
                idx = evaluate_at_index(self.outerStarts, j).GetValueAsSigned()
                for c in range(self.innerNNZs[j]):
                    i = evaluate_at_index(self.indices, idx + c).GetValueAsSigned()
                    raw.append((i, j, self.get(idx + c)))
            raw.sort(key=lambda x: x[0])
            entries = ["[%d, %d] =%s" % (i, j, v.rjust(padding + 1)) for i, j, v in raw]

        return "rows: %d, cols: %d, nonzeros: %d\n{ %s }\n" % (
            self.rows, self.cols, self.nonzeros, ", ".join(entries)
        )


class Quaternion(Printer):
    def __init__(self, val):
        try:
            type_str = val.GetType().GetDirectBaseClassAtIndex(0).GetName()
            regex = re.compile("Eigen::Quaternion<.*?>")
            self.variety = regex.findall(type_str)[0]
            data = val.GetValueForExpressionPath(".m_coeffs.m_storage.m_data.array")
            Printer.__init__(self, data)
            self.get = partial(Printer.evaluate_real, self)
        except:
            self.variety = -1

    def to_string(self):
        return "{ [x] = %s, [y] = %s, [z] = %s, [w] = %s }\n" % tuple(
            self.get(i) for i in range(4)
        )


def eigen_matrix_print(valobj, internal_dict):
    m = Matrix("Matrix", valobj)
    return m.to_string() if m.variety != -1 else ""

def eigen_array_print(valobj, internal_dict):
    a = Matrix("Array", valobj)
    return a.to_string() if a.variety != -1 else ""

def eigen_quaternion_print(valobj, internal_dict):
    q = Quaternion(valobj)
    return q.to_string() if q.variety != -1 else ""

def eigen_sparsematrix_print(valobj, internal_dict):
    s = SparseMatrix(valobj)
    return s.to_string() if s.variety != -1 else ""

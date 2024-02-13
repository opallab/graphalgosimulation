from . import layers
from ltfs import common
from ltfs.common.proto import BaseLayer, BaseAttention, BaseMLP, BaseTransformer


class Model(BaseLayer):
    def __init__(self, d: int, m: int, index: dict, grad: bool = False):
        super().__init__()
        self._build_model(d, m, index, grad)
        self.termination = index["TERM"]
        
    def forward(self, X, A):
        stop = False
        while not stop:
            X = self.forward_step(X, A)
            stop = X[0, self.termination]
        return X
    
    def forward_step(self, X, A, step: str = None):
            X = self._min_initialize(X, A)
            if step == "min_initialize":
                return X

            X = self._mask_visited(X, A)
            if step == "mask_visited":
                return X
            
            X = self._min_increment(X, A)
            if step == "min_increment":
                return X
            
            X = self._round_pe(X, A)
            if step == "round_pe":
                return X
            
            X = self._min_read(X, A)
            if step == "min_read":
                return X
            
            X = self._min_less_than(X, A)
            if step == "min_less_than":
                return X

            X = self._round_bin_1(X, A)
            if step == "round_bin_1":
                return X
            
            X = self._min_if_else(X, A)
            if step == "min_if_else":
                return X
            
            X = self._min_update(X, A)
            if step == "min_update":
                return X

            X = self._min_terminate(X, A)
            if step == "min_terminate":
                return X

            X = self._round_bin_2(X, A)
            if step == "round_bin_2":
                return X
            
            X = self._read_only_check(X, A)
            if step == "read_only_check":
                return X
            
            X = self._round_bin_3(X, A)
            if step == "round_bin_3":
                return X

            X = self._write(X, A)
            if step == "write":
                return X
            
            X = self._read_A(X, A)
            if step == "read_A":
                return X
        
            X = self._read(X, A)
            if step == "read":
                return X

            X = self._update(X, A)
            if step == "update":
                return X
            
            X = self._round_bin_4(X, A)
            if step == "round_bin_4":
                return X

            X = self._mask_write(X, A)
            if step == "mask_write":
                return X
            
            X = self._round_bin_5(X, A)
            if step == "round_bin_5":
                return X
            
            X = self._if_else(X, A)
            if step == "if_else":
                return X

            X = self._round_bin_6(X, A)
            if step == "round_bin_6":
                return X

            X = self._terminate(X, A)
            if step == "terminate":    
                return X
            
            X = self._round_bin_7(X, A)
            if step == "round_bin_7":
                return X

            X = self._keep_termination(X, A)
            if step == "keep_termination":
                return X
        
            if step is not None:
                raise ValueError(f"Unknown step {step}")

            return X

    def _build_model(self, d: int, m: int, index: dict, grad: bool):

        empty_attention = lambda: BaseAttention(d, m, grad=grad)
        empty_mlp = lambda: BaseMLP(d, grad=grad)

        # step 1
        self._min_initialize = BaseTransformer.from_pretrained(
            empty_attention(),
            common.min.initialize(d, index, grad=grad)[0]
        )

        # step 2
        self._mask_visited = BaseTransformer.from_pretrained(
            empty_attention(),
            common.mask.mask_visited(d, index, "D", "bin_visit", "M_D", grad=grad)[0],
        )

        # step 3
        self._min_increment = BaseTransformer.from_pretrained(
            empty_attention(),
            common.write.increment_pe(d, index, ["M_P_cur"], ["M_P_cur"], grad=grad)[0]
        )

        # step 4 and 5 (2 layers)
        self._round_pe = common.round.round_positional_encoding(d, m, index, grad=grad)

        # step 6
        self._min_read = common.min.read_min(d, m, index, grad=grad)

        # step 7
        self._min_less_than = BaseTransformer.from_pretrained(
            empty_attention(),
            common.logic.less_than(d, index, "M_val_cur", "M_val_best", "M_is_less", grad=grad)[0],
        )

        # step 8
        self._round_bin_1 = BaseTransformer.from_pretrained(
            empty_attention(),
            common.round.round_binary_fields(d, index, grad=grad)[0],
        )

        # step 9
        self._min_if_else = BaseTransformer.from_pretrained(
            empty_attention(),
            common.logic.if_else(d, index, 
                ["M_val_cur", "M_P_cur", "M_P_int_cur"],
                ["M_val_best", "M_P_best", "M_P_int_best"], "M_is_less", 
                ["M_val_best", "M_P_best", "M_P_int_best"], grad=grad)[0],
        )

        # step 10
        self._min_update = common.min.update_min(d, m, index, grad=grad)

        # step 11
        self._min_terminate = common.min.termination_min(d, m, index, grad=grad)

        # step 12
        self._round_bin_2 = BaseTransformer.from_pretrained(
            empty_attention(),
            common.round.round_binary_fields(d, index, grad=grad)[0],
        )

        # step 13
        self._read_only_check = BaseTransformer.from_pretrained(
            empty_attention(),
            common.read.read_only_check(d, index, grad=grad),
        )

        # step 14
        self._round_bin_3 = BaseTransformer.from_pretrained(
            empty_attention(),
            common.round.round_binary_fields(d, index, grad=grad)[0],
        )

        # step 15
        self._write = BaseTransformer.from_pretrained(
            empty_attention(),
            common.logic.if_else(d, index,
                ["M_P_best", "M_P_int_best"],
                ["P_i", "P_i_int"], "bin_write",
                ["P_i", "P_i_int"], grad=grad)[0]
        )

        # step 16
        self._read_A = BaseTransformer.from_pretrained(
            common.read.read_A(d, m, index, grad=grad),
            empty_mlp(),
        )

        # step 17
        self._read = BaseTransformer.from_pretrained(
            layers.read(d, m, index, grad=grad),
            empty_mlp(),
        )

        # step 18
        self._update = layers.update(d, m, index, grad=grad)

        # step 19
        self._round_bin_4 = BaseTransformer.from_pretrained(
            empty_attention(),
            common.round.round_binary_fields(d, index, grad=grad)[0],
        )

        # step 20
        self._mask_write = BaseTransformer.from_pretrained(
            empty_attention(),
            layers.mask_write(d, index, grad=grad),
        )

        # step 21
        self._round_bin_5 = BaseTransformer.from_pretrained(
            empty_attention(),
            common.round.round_binary_fields(d, index, grad=grad)[0],
        )

        # step 22
        self._if_else = BaseTransformer.from_pretrained(
            empty_attention(),
            common.logic.if_else(d, index,
                ["order", "P_n"], ["D", "OUT"],
                "S_change", ["D", "OUT"], grad=grad)[0]
        )

        # step 23
        self._round_bin_6 = BaseTransformer.from_pretrained(
            empty_attention(),
            common.round.round_binary_fields(d, index, grad=grad)[0],
        )

        # step 24
        self._terminate = layers.termination(d, m, index, grad=grad)

        # step 25
        self._round_bin_7 = BaseTransformer.from_pretrained(
            empty_attention(),
            common.round.round_binary_fields(d, index, grad=grad)[0],
        )

        # step 26
        self._keep_termination = BaseTransformer.from_pretrained(
            empty_attention(),
            common.logic.if_else(d, index,
                ["TERM"], ["S_bin_term"],
                "TERM", ["TERM"], grad=grad)[0]
        )
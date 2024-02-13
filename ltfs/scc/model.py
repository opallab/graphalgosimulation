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

            X = self._min_if_else_fields(X, A)
            if step == "min_if_else_field":
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

            X = self._round_bin_2(X, A)
            if step == "round_bin_2":
                return X

            X = self._min_terminate(X, A)
            if step == "min_terminate":
                return X

            X = self._round_bin_3(X, A)
            if step == "round_bin_3":
                return X
            
            X = self._read_only_check(X, A)
            if step == "read_only_check":
                return X
            
            X = self._round_bin_4(X, A)
            if step == "round_bin_4":
                return X

            X = self._write(X, A)
            if step == "write":
                return X
            
            X = self._read_A(X, A)
            if step == "read_A":
                return X

            X = self._update(X, A)
            if step == "update":
                return X

            X = self._round_bin_5(X, A)
            if step == "round_bin_5":
                return X

            X = self._repeat_n1(X, A)
            if step == "repeat_n1":
                return X
            
            X = self._mark_visited(X, A)
            if step == "mark_visited":
                return X

            X = self._round_bin_6(X, A)
            if step == "round_bin_6":
                return X

            X = self._all_visited(X, A)
            if step == "all_visited":
                return X

            X = self._is_prev(X, A)
            if step == "is_prev":
                return X
    
            X = self._round_bin_7(X, A)
            if step == "round_bin_7":
                return X
            
            X = self._if_else_ref(X, A)
            if step == "if_else_ref":
                return X

            X = self._repeat_n2(X, A)
            if step == "repeat_n2":
                return X

            X = self._build_flags(X, A)
            if step == "build_flags":
                return X
            
            X = self._round_bin_8(X, A)
            if step == "round_bin_8":
                return X

            X = self._if_else_queue1(X, A)
            if step == "if_else_queue1":
                return X

            X = self._if_else_queue2(X, A)
            if step == "if_else_queue2":
                return X

            X = self._if_else_scc(X, A)
            if step == "if_else_pr":
                return X
            
            X = self._round_bin_9(X, A)
            if step == "round_bin_9":
                return X

            X = self._terminate(X, A)
            if step == "terminate":    
                return X

            X = self._round_bin_10(X, A)
            if step == "round_bin_10":
                return X
            
            X = self._keep_termination1(X, A)
            if step == "keep_termination1":
                return X
            
            X = self._keep_termination2(X, A)
            if step == "keep_termination2":
                return X

            X = self._repeat_n3(X, A)
            if step == "repeat_n3":
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
        self._min_if_else_fields = BaseTransformer.from_pretrained(
            empty_attention(),
            common.logic.if_else(d, index,
                ["Q2", "bin_visit3"], ["Q1", "bin_visit2"], "bin_termn1",
                ["M_D", "S_bin_visit_mask"], grad=grad)[0]
        )

        # step 3
        self._mask_visited = BaseTransformer.from_pretrained(
            empty_attention(),
            common.mask.mask_visited(d, index, "M_D", "S_bin_visit_mask", "M_D", grad=grad)[0],
        )

        # step 4
        self._min_increment = BaseTransformer.from_pretrained(
            empty_attention(),
            common.write.increment_pe(d, index, ["M_P_cur"], ["M_P_cur"], grad=grad)[0]
        )

        # step 5 and 6 (2 layers)
        self._round_pe = common.round.round_positional_encoding(d, m, index, grad=grad)

        # step 7
        self._min_read = layers.read_min(d, m, index, grad=grad)

        # step 8
        self._min_less_than = BaseTransformer.from_pretrained(
            empty_attention(),
            common.logic.less_than(d, index, "M_val_cur", "M_val_best", "M_is_less", grad=grad)[0],
        )

        # step 9
        self._round_bin_1 = BaseTransformer.from_pretrained(
            empty_attention(),
            common.round.round_binary_fields(d, index, grad=grad)[0],
        )

        # step 10
        self._min_if_else = BaseTransformer.from_pretrained(
            empty_attention(),
            common.logic.if_else(d, index, 
                ["M_val_cur", "M_P_cur", "M_SCC_cur", "M_SCC_cur_int"],
                ["M_val_best", "M_P_best", "M_SCC_best", "M_SCC_best_int"], "M_is_less",
                ["M_val_best", "M_P_best", "M_SCC_best", "M_SCC_best_int"], grad=grad)[0]
        )

        # step 11
        self._min_update = common.min.update_min(d, m, index, grad=grad)

        # step 12
        self._round_bin_2 = BaseTransformer.from_pretrained(
            empty_attention(),
            common.round.round_binary_fields(d, index, grad=grad)[0],
        )

        # step 13
        self._min_terminate = common.min.termination_min(d, m, index, grad=grad)

        # step 14
        self._round_bin_3 = BaseTransformer.from_pretrained(
            empty_attention(),
            common.round.round_binary_fields(d, index, grad=grad)[0],
        )

        # step 15
        self._read_only_check = BaseTransformer.from_pretrained(
            empty_attention(),
            layers.read_only_check(d, index, grad=grad),
        )

        # step 16
        self._round_bin_4 = BaseTransformer.from_pretrained(
            empty_attention(),
            common.round.round_binary_fields(d, index, grad=grad)[0],
        )

        # step 17
        self._write = BaseTransformer.from_pretrained(
            empty_attention(),
            common.logic.if_else(d, index, 
                ["M_P_best", "M_SCC_best", "M_SCC_best_int"],
                ["P_i", "SCC_i", "SCC_i_int"], "bin_write_all",
                ["P_i", "SCC_i", "SCC_i_int"], grad=grad)[0]
        )

        # step 18
        self._read_A = BaseTransformer.from_pretrained(
            layers.read_A(d, m, index, grad=grad),
            common.round.round_binary_fields(d, index, grad=grad)[0],
        )
        
        # step 19
        self._update = layers.update(d, m, index, grad=grad)

        # step 20
        self._round_bin_5 = BaseTransformer.from_pretrained(
            empty_attention(),
            common.round.round_binary_fields(d, index, grad=grad)[0],
        )

        # step 21
        self._repeat_n1 = BaseTransformer.from_pretrained(
            common.write.repeat_n(d, m, index, 
                ["bin_write1", "bin_write2", "M_bin_keep"],
                ["bin_writen1", "bin_writen2", "M_bin_keep"], grad=grad)[0],
            common.round.round_binary_fields(d, index, grad=grad)[0],
        )

        # step 22
        self._mark_visited = BaseTransformer.from_pretrained(
            empty_attention(),
            layers.mark_visited(d, index, grad=grad),
        )

        # step 23
        self._round_bin_6 = BaseTransformer.from_pretrained(
            empty_attention(),
            common.round.round_binary_fields(d, index, grad=grad)[0],
        )

        # step 24
        self._all_visited = layers.all_neighbors_visited(d, m, index, grad=grad)

        # step 25
        self._is_prev = layers.is_prev_visited(d, m, index, grad=grad)

        # step 26
        self._round_bin_7 = BaseTransformer.from_pretrained(
            empty_attention(),
            common.round.round_binary_fields(d, index, grad=grad)[0],
        )

        # step 27
        self._if_else_ref = BaseTransformer.from_pretrained(
            empty_attention(),
            common.logic.if_else(d, index, 
                ["SCC_i", "SCC_i_int"], ["P_ref", "P_ref_int"], "bin_ref",
                ["P_ref", "P_ref_int"], grad=grad)[0]
        )

        # step 28
        self._repeat_n2 = BaseTransformer.from_pretrained(
            common.write.repeat_n(d, m, index, 
                ["P_ref", "P_ref_int", "bin_all"],
                ["P_ref_n", "P_ref_int_n", "bin_all"], grad=grad)[0],
            common.round.round_binary_fields(d, index, grad=grad)[0],
        )

        # step 29
        self._build_flags = BaseTransformer.from_pretrained(
            empty_attention(),
            layers.build_flags(d, index, grad=grad),
        )

        # step 30
        self._round_bin_8 = BaseTransformer.from_pretrained(
            empty_attention(),
            common.round.round_binary_fields(d, index, grad=grad)[0],
        )

        # step 31
        self._if_else_queue1 = BaseTransformer.from_pretrained(
            empty_attention(),
            common.logic.if_else(d, index, 
                ["Dec"], ["Q1"], "bin_Q1",
                ["Q1"], grad=grad)[0]
        )

        # step 32
        self._if_else_queue2 = BaseTransformer.from_pretrained(
            empty_attention(),
            common.logic.if_else(d, index, 
                ["Dec", "bin_Q2"], ["Q2", "bin_visit2"], "bin_Q2",
                ["Q2", "bin_visit2"], grad=grad)[0]
        )

        # step 33
        self._if_else_scc = BaseTransformer.from_pretrained(
            empty_attention(),
            common.logic.if_else(d, index, 
                ["Dec", "P_ref_n", "P_ref_int_n"], ["Q2", "SCC", "OUT"], "bin_Q3",
                ["Q2", "SCC", "OUT"], grad=grad)[0]
        )

        # step 34
        self._round_bin_9 = BaseTransformer.from_pretrained(
            empty_attention(),
            common.round.round_binary_fields(d, index, grad=grad)[0],
        )

        # step 35
        self._terminate = layers.termination(d, m, index, grad=grad)

        # step 36
        self._round_bin_10 = BaseTransformer.from_pretrained(
            empty_attention(),
            common.round.round_binary_fields(d, index, grad=grad)[0],
        )

        # step 37
        self._keep_termination1 = BaseTransformer.from_pretrained(
            empty_attention(),
            common.logic.if_else(d, index, 
                ["bin_term1"], ["S_bin_term1"], "bin_term1",
                ["bin_term1"], grad=grad)[0]
        )

        # step 38
        self._keep_termination2 = BaseTransformer.from_pretrained(
            empty_attention(),
            common.logic.if_else(d, index, 
                ["TERM"], ["S_bin_term2"], "TERM",
                ["TERM"], grad=grad)[0]
        )

        # step 39
        self._repeat_n3 = BaseTransformer.from_pretrained(
            common.write.repeat_n(d, m, index, 
                ["bin_term1"],
                ["bin_termn1"], grad=grad)[0],
            common.round.round_binary_fields(d, index, grad=grad)[0],
        )




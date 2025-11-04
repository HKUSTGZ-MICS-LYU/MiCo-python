class DimTransform:
    def __init__(self, dim: int):
        self.out_dim = dim
        self.in_dim = dim
        return

    def __call__(self, scheme: list) -> list:
        return scheme

class SameWQAQTransform(DimTransform):
    def __init__(self, dim: int):
        super().__init__(dim)
        self.in_dim = dim // 2

    def __call__(self, scheme: list) -> list:
        assert len(scheme) == self.in_dim
        out_scheme = scheme * 2
        assert len(out_scheme) == self.out_dim
        return out_scheme
    

'''
Transform for LLaMa models to apply quantization scheme per Transformer block + Output Layer.
'''
class LLaMaBlockWiseTransform(DimTransform):
    def __init__(self, dim, llama_layers: int = 7):
        super().__init__(dim)
        assert (dim - 2) % llama_layers == 0
        self.in_dim = (dim - 2) // llama_layers + 2
        self.llama_layers = llama_layers
        assert self.in_dim % 2 == 0

    def __call__(self, scheme: list) -> list:
        assert len(scheme) == self.in_dim
        in_wq = scheme[:self.in_dim // 2]
        in_aq = scheme[self.in_dim // 2:]
        wq_scheme = []
        aq_scheme = []
        for i in range(self.in_dim // 2 - 1):
            wq_scheme += [in_wq[i]] * self.llama_layers
            aq_scheme += [in_aq[i]] * self.llama_layers
        wq_scheme += [in_wq[-1]]
        aq_scheme += [in_aq[-1]]
        out_scheme = wq_scheme + aq_scheme
        assert len(out_scheme) == self.out_dim, f"{len(out_scheme)} != {self.out_dim}"
        return out_scheme

'''
Transform for LLaMa models to apply quantization scheme inside a Transformer block + Output Layer.
Basically: WQ, WK, WV, WO + FF1, FF2, FF3 + Output (8 parts)
'''
class LLaMaInBlockTransformer(DimTransform):
    def __init__(self, dim, llama_layers: int = 7):
        super().__init__(dim)
        assert (dim - 2) % llama_layers == 0
        self.in_dim = (llama_layers + 1) * 2
        self.llama_layers = llama_layers
        self.num_blocks = (self.out_dim // 2 - 1) // llama_layers
        assert self.in_dim % 2 == 0

    def __call__(self, scheme: list) -> list:
        assert len(scheme) == self.in_dim
        wq_scheme = []
        aq_scheme = []
        in_wq = scheme[:self.in_dim // 2]
        in_aq = scheme[self.in_dim // 2:]
        wq_scheme += in_wq[:-1] * self.num_blocks
        aq_scheme += in_aq[:-1] * self.num_blocks
        wq_scheme += [in_wq[-1]]
        aq_scheme += [in_aq[-1]]
        out_scheme = wq_scheme + aq_scheme
        assert len(out_scheme) == self.out_dim, f"{len(out_scheme)} != {self.out_dim}"
        return out_scheme
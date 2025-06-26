from transformers import BertConfig
from typing import List


class SpladeFuseMaxPConfig(BertConfig):
    '''
    Config Class for Splade Fuse Max Pooling
    '''
    model_type = "spladeFuseMaxP"

    def __init__(
        self,hidden_dense:int = 4096, threshold_method:str = None, **kwargs,
    ):
        self.threshold_method = threshold_method
        self.hidden_dense = hidden_dense
        super().__init__(**kwargs)
    
    def setHiddenDense(self,hidden_dense:int) -> None:
        self.hidden_dense = hidden_dense
    
    def setThresholdMethod(self,threshold_method:str) -> None:
        self.threshold_method = threshold_method
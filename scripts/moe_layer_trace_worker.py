"""Only copies native layer inputs/outputs; no replacement computations."""

import functools

from scripts.moe_ep_correctness_worker import EPCorrectnessDiagnosticWorker


class LayerTraceDiagnosticWorker(EPCorrectnessDiagnosticWorker):
    def enable_layer_trace(self):
        self.layer_trace={}
        self.trace_layer=None
        from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter
        original=FusedMoERouter.select_experts

        @functools.wraps(original)
        def select(router,*args,**kwargs):
            weights,ids=original(router,*args,**kwargs)
            if self.trace_layer is not None and ids.shape[0]:
                self.layer_trace[self.trace_layer].update({"selected_ids":ids[-1].cpu().tolist(),
                    "selected_weights":weights[-1].float().cpu().tolist()})
            return weights,ids
        FusedMoERouter.select_experts=select
        model=self.model_runner.model
        for index,layer in enumerate(model.model.layers):
            block=layer.block_sparse_moe

            def before(module,inputs,key=index):
                self.trace_layer=key
                self.layer_trace[key]={"input":inputs[0].reshape(-1,1536)[-1].float().cpu().tolist()}

            def after(module,inputs,output,key=index):
                self.layer_trace[key]["moe_output"]=output.reshape(-1,1536)[-1].float().cpu().tolist()
                self.trace_layer=None

            def gate(module,inputs,output,key=index):
                self.layer_trace[key]["router_logits"]=output[0][-1].float().cpu().tolist()

            def decoder(module,inputs,output,key=index):
                self.layer_trace[key]["decoder_output"]=output.reshape(-1,1536)[-1].float().cpu().tolist()

            block.register_forward_pre_hook(before)
            block.register_forward_hook(after)
            block.gate.register_forward_hook(gate)
            layer.register_forward_hook(decoder)

    def reset_layer_trace(self):
        self.layer_trace={}

    def get_layer_trace(self):
        return {"rank":self.rank,"local_rank":self.local_rank,
                "layers":{str(key):value for key,value in self.layer_trace.items()},
                "scope":"last prefill query of the identical prefix, one request and one generated token"}

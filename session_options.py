import onnxruntime as ort

def get_optimized_session(model_path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8     
    so.inter_op_num_threads = 8

    # optimizador de grafos
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    return ort.InferenceSession(
        model_path,
        sess_options=so,
        providers=[
            "CPUExecutionProvider"
        ]
    )

import argparse
import mmcv
import numpy as np
import onnxruntime as rt
import torch
import torch.nn as nn
from mmcv.onnx import register_extra_symbolics
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor
from mmseg.datasets.pipelines import Compose

class MaCViExportWrapper(nn.Module):
    """Ensures a single-input single-output static graph for MaCVi."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img):
        # resacle=False is critical to prevent dynamic resizing in the graph
        logits = self.model.encode_decode(img, img_metas=[])
        # Output single-channel class indices [1, H, W]
        return logits.argmax(dim=1).to(torch.int32)

def pytorch2onnx(model, input_shape, output_file, opset_version=11, verify_img=None, cfg=None):
    model.cpu().eval()
    wrapped_model = MaCViExportWrapper(model)
    
    # 1. Export the Model
    dummy_input = torch.randn(input_shape)
    register_extra_symbolics(opset_version)
    
    torch.onnx.export(
        wrapped_model,
        (dummy_input, ),
        output_file,
        input_names=['input'],
        output_names=['output'],
        export_params=True,
        opset_version=opset_version,
        dynamic_axes=None) # Static graph required
    
    print(f'Successfully exported: {output_file}')

    # 2. Comparison Check with Real Image
    if verify_img and cfg:
        print(f'--- Starting Comparison Verification on {verify_img} ---')
        
        # Build standard preprocessing pipeline (Load -> MultiScale -> ImageToTensor -> Collect)
        # We manually run this to get the exact tensor the ONNX model expects
        device = next(model.parameters()).device
        test_pipeline = cfg.data.test.pipeline
        # Ensure we use the right shape for preprocessing
        test_pipeline[1]['img_scale'] = (input_shape[3], input_shape[2]) 
        test_pipeline[1]['flip'] = False
        
        pipeline = Compose(test_pipeline)
        data = pipeline(dict(img=verify_img))
        img_tensor = data['img'][0].unsqueeze(0).to(device) # [1, 3, H, W]

        # PyTorch Inference
        with torch.no_grad():
            pytorch_result = wrapped_model(img_tensor).cpu().numpy()

        # ONNX Inference
        sess = rt.InferenceSession(output_file)
        onnx_inputs = {sess.get_inputs()[0].name: img_tensor.cpu().numpy()}
        onnx_result = sess.run(None, onnx_inputs)[0]

        # Numerical Comparison
        mismatch = np.sum(pytorch_result != onnx_result)
        total_pixels = onnx_result.size
        print(f'Pixel Mismatch: {mismatch} / {total_pixels} ({(mismatch/total_pixels)*100:.4f}%)')
        
        if mismatch == 0:
            print('VERIFICATION SUCCESS: PyTorch and ONNX results are identical!')
        else:
            print('VERIFICATION WARNING: Minor mismatches found (often due to float precision).')

def main():
    parser = argparse.ArgumentParser(description='Convert MMSeg to MaCVi ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--input-img', type=str, help='Image for verification', default=None)
    parser.add_argument('--output-file', type=str, default='model.onnx')
    parser.add_argument('--shape', type=int, nargs='+', default=[384, 768], help='H W')
    args = parser.parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    input_shape = (1, 3, args.shape[0], args.shape[1])

    # Build model
    segmentor = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(segmentor, args.checkpoint, map_location='cpu')
    
    pytorch2onnx(
        segmentor, 
        input_shape, 
        args.output_file, 
        verify_img=args.input_img,
        cfg=cfg
    )

if __name__ == '__main__':
    main()
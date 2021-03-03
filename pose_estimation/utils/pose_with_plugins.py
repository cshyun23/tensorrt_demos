from __future__ import print_function

import ctypes

import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda

'''
try:
    ctypes.cdll.LoadLibrary('../plugins/libyolo_layer.so')
except OSError as e:
    raise SystemExit('ERROR: failed to load ./plugins/libyolo_layer.so.  '
                     'Did you forget to do a "make" in the "./plugins/" '
                     'subdirectory?') from e
'''
def _preprocess_pose(model_name, img, input_shape):
    """Preprocess an image before TRT POSE inferencing.
    # Args
        img: int8 numpy array of shape (img_h, img_w, 3)
        input_shape: a tuple of (H, W)
    # Returns
        preprocessed img: float32 numpy array of shape (3, H, W)
    """
    if 'resnet' in model_name:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    #print(img.shape)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img[0] = np.subtract(img[0], mean[0])/std[0]
    img[1] = np.subtract(img[1], mean[1])/std[1]
    img[2] = np.subtract(img[2], mean[2])/std[2]
    #print(img.shape)
    #print(type(img[1][20][15]))
    #print(img[1][20][15])
    img = np.expand_dims(img, axis=0)
    return img


class HostDeviceMem(object):
    """Simple helper data class that's a little nicer to use than a 2-tuple."""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    """Allocates all host/device in/out buffers required for an engine."""
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * \
               engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            # each grid has 3 anchors, each anchor generates a detection
            # output of 7 float32 values
            #print(size)
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference_v2(context, bindings, inputs, outputs, stream):
    """do_inference_v2 (for TensorRT 7.0+)

    This function is generalized for multiple inputs/outputs for full
    dimension networks.
    Inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


class TrtPOSE(object):
    """TrtYOLO class encapsulates things needed to run TRT YOLO."""

    def _load_engine(self):
        TRTbin = self.model
        with open(TRTbin, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def __init__(self, model_path, input_shape, cuda_ctx=None):
        """Initialize TensorRT plugins, engine and conetxt."""
        self.model = model_path
        self.input_shape = input_shape
        self.cuda_ctx = cuda_ctx
        if self.cuda_ctx:
            self.cuda_ctx.push()

        self.inference_fn = do_inference_v2
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine()

        try:
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = \
                allocate_buffers(self.engine)
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e
        finally:
            if self.cuda_ctx:
                self.cuda_ctx.pop()

    def __del__(self):
        """Free CUDA memories."""
        del self.outputs
        del self.inputs
        del self.stream

    def estimation(self, model_name, img):

        img_resized = _preprocess_pose(model_name, img, (self.input_shape[0], self.input_shape[1]))

        # Set host input to the image. The do_inference() function
        # will copy the input to the GPU before executing.
        self.inputs[0].host = np.ascontiguousarray(img_resized)

        if self.cuda_ctx:
            self.cuda_ctx.push()

        trt_outputs = self.inference_fn(
            context=self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream)

        if self.cuda_ctx:
            self.cuda_ctx.pop()

        if "384" in model_name:
            output = np.array(trt_outputs).reshape(1, 17, 96, 72)
        else:
            output = np.array(trt_outputs).reshape(1, 17, 64, 48)
        
        return output

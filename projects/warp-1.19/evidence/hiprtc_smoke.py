import ctypes as c

hip = c.CDLL('/opt/dtk/lib/libamdhip64.so')
rtc = c.CDLL('/opt/dtk/lib/libhiprtc.so')

def call(lib, name, types, *args):
    fn = getattr(lib, name)
    fn.argtypes = types
    fn.restype = c.c_int
    result = fn(*args)
    if result:
        raise RuntimeError(f'{name}: {result}')

ptr = c.c_void_p
integer = c.c_int
size = c.c_size_t
call(hip, 'hipSetDevice', [integer], 0)
name = c.create_string_buffer(256)
call(hip, 'hipDeviceGetName', [ptr, integer, integer], name, 256, 0)
source = b'extern "C" __global__ void fill(int* out) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<257) out[i]=i*3+7; }'
program = ptr()
call(rtc, 'hiprtcCreateProgram', [c.POINTER(ptr), c.c_char_p, c.c_char_p, integer, ptr, ptr], c.byref(program), source, b'smoke.hip', 0, None, None)
options = (c.c_char_p * 2)(b'--gpu-architecture=gfx936', b'-std=c++17')
try:
    call(rtc, 'hiprtcCompileProgram', [ptr, integer, ptr], program, 2, options)
except RuntimeError:
    length = size()
    call(rtc, 'hiprtcGetProgramLogSize', [ptr, c.POINTER(size)], program, c.byref(length))
    log = c.create_string_buffer(length.value)
    call(rtc, 'hiprtcGetProgramLog', [ptr, ptr], program, log)
    print(log.value.decode())
    raise
length = size()
call(rtc, 'hiprtcGetCodeSize', [ptr, c.POINTER(size)], program, c.byref(length))
code = c.create_string_buffer(length.value)
call(rtc, 'hiprtcGetCode', [ptr, ptr], program, code)
module, function, allocation = ptr(), ptr(), ptr()
call(hip, 'hipModuleLoadData', [c.POINTER(ptr), ptr], c.byref(module), code)
call(hip, 'hipModuleGetFunction', [c.POINTER(ptr), ptr, c.c_char_p], c.byref(function), module, b'fill')
call(hip, 'hipMalloc', [c.POINTER(ptr), size], c.byref(allocation), 257 * 4)
call(hip, 'hipMemset', [ptr, integer, size], allocation, 0xa5, 257 * 4)
arguments = (ptr * 1)(c.cast(c.byref(allocation), ptr))
call(hip, 'hipModuleLaunchKernel', [ptr] + [c.c_uint] * 7 + [ptr, ptr, ptr], function, 2, 1, 1, 256, 1, 1, 0, None, arguments, None)
call(hip, 'hipDeviceSynchronize', [])
output = (integer * 257)()
call(hip, 'hipMemcpyDtoH', [ptr, ptr, size], output, allocation, 257 * 4)
assert list(output) == [i * 3 + 7 for i in range(257)]
call(hip, 'hipFree', [ptr], allocation)
call(hip, 'hipModuleUnload', [ptr], module)
call(rtc, 'hiprtcDestroyProgram', [c.POINTER(ptr)], c.byref(program))
print(f'PASS: {name.value.decode()}, gfx936, HIPRTC compile/load/launch, 257 exact integer outputs')

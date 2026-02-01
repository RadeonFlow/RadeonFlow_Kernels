import zlib
import base64

# 1. Read source files
with open('src/perf_gemm.cc', 'rb') as f:
    kernel_cc = f.read()

# 2. Concatenate C++ sources
kernel_cc = kernel_cc.replace(b'#include "gemm_rs_kernel.h"\n', b'')

# 3. Compress and encode sources
encoded_cpp = base64.b64encode(zlib.compress(b'', level=9))
encoded_cuda = base64.b64encode(zlib.compress(kernel_cc, level=9))

# 4. Read the template.py template
with open('template.py', 'r') as f:
    submission_template = f.read()

# 5. Format the final submission file content
submission_content = submission_template.replace(
    'CPP_WRAPPER = ""',
    f'CPP_WRAPPER = zlib.decompress(base64.b64decode({encoded_cpp!r})).decode("utf-8")'
).replace(
    'CUDA_SRC = ""',
    f'CUDA_SRC = zlib.decompress(base64.b64decode({encoded_cuda!r})).decode("utf-8")'
)

# 6. Write the new submission.py
with open('submission.py', 'w') as f:
    f.write(submission_content)

print("submission.py has been generated successfully.")

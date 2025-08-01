#include "lib/$fileTransformUv"

void mx_image_color4($texSamplerSignature, int layer, vec4 defaultval, vec2 texcoord, int uaddressmode, int vaddressmode, int filtertype, int framerange, int frameoffset, int frameendaction, vec2 uv_scale, vec2 uv_offset, out vec4 result)
{
    vec2 uv = mx_transform_uv(texcoord, uv_scale, uv_offset);
    result = texture($texSamplerSampler2D, uv);
}
